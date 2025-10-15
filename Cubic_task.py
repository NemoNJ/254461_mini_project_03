import time
import numpy as np
import math as m
import random as rand
import re

# ======================= Global constants =======================
pi = np.pi
d2r = pi / 180
r2d = 1 / d2r
Pe = np.array([0, 0, 0, 1], dtype=float)          # origin of current frame
Ende = np.array([0, 0, 0.19163, 1], dtype=float)  # ไม่ได้ใช้ใน FK หลักแล้ว แต่คงไว้ให้

# Joint accel limits (ถ้าจะใช้จำกัด joint ภายหลังได้)
a_lim = np.array([80, 80, 100, 150, 150, 200], dtype=float)

# Globals จะถูกเซ็ตใน sysCall_thread()
theta = None          # current joint angles [rad], length 6
DH_table = None       # DH rows: [alpha_deg, a, d, theta_offset_deg]

# ======================= DH & Transform =======================
def dh_transforms(alpha, a, d, theta):
    ca, sa = m.cos(alpha), m.sin(alpha)
    ct, st = m.cos(theta), m.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0, 0, 0, 1]
    ], dtype=float)

def Homogeneous_trans_matrix(_first, _last):
    """
    คูณ T จากแถว _first จนก่อน _last (last-exclusive)
    และ "ใส่มุมข้อต่อปัจจุบัน" ลงไปแทน theta ของ 6 joint แรก
    """
    global DH_table, theta
    Tn = np.identity(4)
    for t in range(_first, _last):
        alpha_deg, a, d, theta_off_deg = DH_table[t]
        alpha_rad = alpha_deg * d2r
        if t < 6:
            theta_rad = theta[t] + theta_off_deg * d2r  # joint i
        else:
            theta_rad = theta_off_deg * d2r             # tool frame (fixed)
        Tn = np.dot(Tn, dh_transforms(alpha_rad, a, d, theta_rad))
    return Tn

def rotation_matrix(_first, _last):
    return Homogeneous_trans_matrix(_first, _last)[0:3, 0:3]

def rotation_matrix_to_euler_angles(R):
    # ZYX euler (roll x, pitch y, yaw z)
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = m.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = m.atan2(-R[1, 2], R[2, 2])
        gamma = m.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = m.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma])  # [rad]

def foward_kinematic():
    # position of end-effector (base frame)
    fw = np.dot(Homogeneous_trans_matrix(0, len(DH_table)), Pe)
    return np.array(fw[:3])

def Euler():
    # return euler [rad] ของ end-effector
    R = rotation_matrix(0, len(DH_table))
    return rotation_matrix_to_euler_angles(R)

# ======================= Jacobian-based IK =======================
def Jacobian_matrix():
    """
    6x6 Jacobian (linear;angular) ใน base frame
    สูตรมาตรฐาน: Jv_i = z_i x (p_e - p_i),  Jw_i = z_i
    หมายเหตุ: ใช้ p_e จากโซ่ DH รวม tool แล้ว (@ Pe)
    """
    n = len(DH_table)  # 7 rows (6 joints + tool)
    Jcols = []

    # ตำแหน่งปลายหุ่นจากโซ่เต็ม (รวม tool)
    p_e = (Homogeneous_trans_matrix(0, n) @ Pe)[0:3]

    for i in range(6):
        Ti = Homogeneous_trans_matrix(0, i) if i > 0 else np.identity(4)
        p_i = Ti[0:3, 3]
        z_i = Ti[0:3, 2]
        Jv = np.cross(z_i, (p_e - p_i))
        Jw = z_i
        Jcols.append(np.concatenate([Jv, Jw]))

    J = np.stack(Jcols, axis=1)
    return J

def X():
    # สถานะปลายหุ่นจริง [x y z roll pitch yaw] (rad)
    pos = foward_kinematic()
    ori = Euler()
    return np.concatenate((pos, ori))

def inverse_kinematics(desire_pos, alpha, theta_deg):
    """
    desire_pos: [x y z roll pitch yaw] (rad)
    alpha: step size
    theta_deg: รับเป็นองศาตามสัญญาเดิม แล้วแปลงเป็น rad ภายใน
    """
    global theta
    Delta_X = desire_pos - X()
    invJ = np.linalg.pinv(Jacobian_matrix())
    Delta_theta = np.dot(invJ, Delta_X)  # [rad]
    theta = np.array(theta_deg) * d2r    # current theta [rad]
    Ntheta = theta + (alpha * Delta_theta)
    return Ntheta

# ======================= Cubic trajectory =======================
def cubic_position(u0, uf, v0, vf, tf, t):
    # v0=vf=0 case
    return u0 + ((3 / (tf**2)) * (uf - u0) * (t**2)) - ((2 / (tf**3)) * (uf - u0) * (t**3))

def cubic_velocity(u0, uf, v0, vf, tf, t):
    return ((6 / (tf**2)) * (uf - u0) * t) - ((6 / (tf**3)) * (uf - u0) * (t**2))

def cubic_acceleration(u0, uf, v0, vf, tf, t):
    return ((6 / (tf**2)) * (uf - u0)) - ((12 / (tf**3)) * (uf - u0) * t)

def cubic_debug(u0, uf, v0, vf, tf, t):
    print(f"time : {t:.3f}s | Pos : {cubic_position(u0,uf,v0,vf,tf,t):.4f} | "
          f"Vel : {cubic_velocity(u0,uf,v0,vf,tf,t):.4f} | "
          f"Acc : {cubic_acceleration(u0,uf,v0,vf,tf,t):.4f}")

# ======================= Random task generator =======================
def random_task_positions():
    # กำหนดช่วงสุ่มที่เหมาะสม (เมตร) ให้ไม่ชนโต๊ะ/ฐาน
    p0 = np.array([rand.uniform(-0.45, 0.45),
                   rand.uniform(-0.45, 0.45),
                   rand.uniform(0.15, 0.55)])
    pf = np.array([rand.uniform(-0.45, 0.45),
                   rand.uniform(-0.45, 0.45),
                   rand.uniform(0.15, 0.55)])
    return p0, pf

# ======================= Robust handle helpers =======================
def _natural_key(name: str):
    return [int(s) if s.isdigit() else s.lower() for s in re.findall(r'\d+|\D+', name)]

def _try_get(paths):
    for p in paths:
        try:
            return sim.getObject(p)
        except:
            pass
    return None

def _find_ur5_handles():
    """
    คืนค่า: (hdl_j_list[6], hdl_end)
    ค้นหาจาก model root (.)
    """
    model = sim.getObject('.')

    # 1) รวบรวม joint ใต้โมเดล
    joints = sim.getObjectsInTree(model, sim.object_joint_type, 0)  # 0 = children
    # จัดเรียงตามชื่อธรรมชาติ
    aliases = [(h, sim.getObjectAlias(h, 1)) for h in joints]
    aliases.sort(key=lambda x: _natural_key(x[1]))
    hdl_j_list = [h for h, _ in aliases[:6]]
    if len(hdl_j_list) != 6:
        # พยายามดึงด้วย path ที่พบบ่อย
        alt = []
        for i in range(1, 7):
            h = _try_get([f'./joint{i}', f'/UR5/joint{i}', f'/UR5_joint{i}'])
            if h is not None:
                alt.append(h)
        if len(alt) == 6:
            hdl_j_list = alt
        else:
            found = [sim.getObjectAlias(h, 1) for h in joints]
            raise RuntimeError(f'พบ joint {len(joints)} ตัว: {found} แต่จัดชุด 6 ตัวไม่สำเร็จ')

    # 2) หา end-effector / tip
    candidate_paths = [
        './EndPoint', '/UR5/EndPoint',
        './tip', '/UR5/tip', './Tip', '/UR5/Tip',
        './ee', '/UR5/ee', './toolTip', '/UR5/toolTip'
    ]
    hdl_end = _try_get(candidate_paths)
    if hdl_end is None:
        dummies = sim.getObjectsInTree(model, sim.object_dummy_type, 0)
        if dummies:
            scored = []
            for d in dummies:
                name = sim.getObjectAlias(d, 1).lower()
                score = 0
                if 'tip' in name: score += 2
                if 'end' in name: score += 1
                scored.append((score, d))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                hdl_end = scored[0][1]
    if hdl_end is None:
        # fallback: ใช้ joint สุดท้ายแทนตำแหน่งคร่าว ๆ
        hdl_end = hdl_j_list[-1]
    return hdl_j_list, hdl_end

# ======================= CoppeliaSim =======================
def sysCall_init():
    global sim
    sim = require('sim')  # ตามรูปแบบ CoppeliaSim

def sysCall_thread():
    global DH_table, theta

    print("\n========= Task Space Path Planning (Cubic + Jacobian IK) =========")

    # ===== Get joints & end-effector robustly (แก้บั๊ก path ไม่ตรง) =====
    try:
        hdl_j_list, hdl_end = _find_ur5_handles()
    except Exception as e:
        print(f"[error] หา joint/EndPoint ไม่สำเร็จ: {e}")
        return
    hdl_j = {i: hdl_j_list[i] for i in range(6)}

    # Draw path
    line_handle = sim.addDrawingObject(sim.drawing_lines, 3, 0, -1, 200, [0, 1, 0])

    # Generate random task
    p0, pf = random_task_positions()
    print(f"Start: {np.round(p0,3)} | Goal: {np.round(pf,3)}")

    # ===== User params =====
    tf_user = 8.0              # เวลาที่ผู้ใช้กำหนด
    a_lin_max = 0.5            # [m/s^2] ความเร่งเชิงเส้นสูงสุดที่กำหนด
    alpha = 0.05               # IK step size

    # ===== Initialize DH (UR5) =====
    theta = np.zeros(6)        # rad
    DH_table = np.array([
        [0,    0,       0.0892, -90],
        [90,   0,       0,       90],
        [0,    0.4251,  0,        0],
        [0,    0.3922,  0.110,   -90],
        [-90,  0,       0.0948,   0],
        [90,   0,       0.0,      0],
        [0,    0,       0.26658, 180]   # tool
    ], dtype=float)

    # ส่งค่าเริ่มต้นไปที่จอยท์
    for i in range(6):
        sim.setJointTargetPosition(hdl_j[i], float(theta[i]))

    # ===== Lock initial orientation =====
    ori0 = Euler()                   # [rad] orientation ณ ต้นทาง
    pos0_actual = foward_kinematic() # สำหรับ debug

    # ===== Respect a_lin_max by stretching tf if needed =====
    # สำหรับ cubic v0=vf=0, |a|max ≈ 6‖Δp‖ / tf^2  → tf_min = sqrt(6‖Δp‖ / a_max)
    dp_norm = np.linalg.norm(pf - p0)
    tf_min = m.sqrt(max(1e-9, 6.0 * dp_norm / max(1e-6, a_lin_max)))
    tf = max(tf_user, tf_min)
    if tf > tf_user + 1e-6:
        print(f"[info] TF stretched to {tf:.3f}s to satisfy a_lin_max={a_lin_max} m/s^2")

    # ===== Trajectory loop =====
    t0 = time.time()
    prev_ee = None

    while True:
        t = time.time() - t0
        if t > tf:
            break

        # Cubic interpolation (v0=vf=0)
        pos_t = np.array([cubic_position(p0[i], pf[i], 0, 0, tf, t) for i in range(3)])
        Xd = np.concatenate((pos_t, ori0))  # orientation คงเดิม = ori0

        # Inverse Kinematics step (คงสัญญา argument เดิม)
        new_theta = inverse_kinematics(Xd, alpha, theta * r2d)
        theta = new_theta

        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(theta[i]))

        # Draw path segment
        ee_pos = sim.getObjectPosition(hdl_end, -1)  # [x,y,z]
        if prev_ee is not None:
            sim.addDrawingObjectItem(line_handle, prev_ee + ee_pos)
        prev_ee = ee_pos

        sim.switchThread()

    print("Motion complete. Path drawn.")
