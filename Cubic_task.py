import time
import numpy as np
import math as m
import random as rand

# ======================= Global constants =======================
pi = np.pi
d2r = pi/180
r2d = 1/d2r

# ======================= Global variables =======================
DH_table = None
DH_size  = 0
Pe   = np.array([0, 0, 0, 1], dtype=float)
Ende = np.array([0, 0, 0.19163, 1], dtype=float)

# ======================= Core robot math (names unchanged) =======================
def degarr2radarr(deg_dict_or_arr):
    if isinstance(deg_dict_or_arr, dict):
        arr_deg = np.array([deg_dict_or_arr.get(i, 0.0) for i in range(6)], dtype=float)
    else:
        arr_deg = np.array(deg_dict_or_arr, dtype=float)
    return arr_deg * d2r

def dh_transforms(alpha, a, d, theta):
    theta = theta * d2r
    alpha = alpha * d2r
    return np.array([
        [m.cos(theta), -m.sin(theta), 0, a],
        [m.sin(theta)*m.cos(alpha), m.cos(theta)*m.cos(alpha), -m.sin(alpha), -m.sin(alpha)*d],
        [m.sin(theta)*m.sin(alpha), m.cos(theta)*m.sin(alpha), m.cos(alpha), m.cos(alpha)*d],
        [0, 0, 0, 1]
    ], dtype=float)

def Homogeneous_trans_matrix(first, last):
    global DH_table
    Tn = np.identity(4)
    for t in range(first, last):
        alpha, a, d, theta = DH_table[t]
        T = dh_transforms(alpha, a, d, theta)
        Tn = np.dot(Tn, T)
    return Tn

def rotation_matrix(first, last):
    T = Homogeneous_trans_matrix(first, last)
    return T[0:3, 0:3]

def Euler():
    R = rotation_matrix(0, DH_size)
    alpha = np.arctan2(-R[1,2], R[2,2]) * r2d
    beta  = np.arcsin(R[0,2]) * r2d
    gamma = np.arctan2(-R[0,1], R[0,0]) * r2d
    return np.array([alpha, beta, gamma])

def foward_kinematic():
    fw = np.dot(Homogeneous_trans_matrix(0, DH_size), Pe)
    return np.array(fw[:3]).round(6)

def Jacobian_matrix():
    J = []
    for i in range(1, DH_size):
        r  = np.dot(Homogeneous_trans_matrix(i, DH_size - 1), Ende)
        R0 = rotation_matrix(0, i)
        r0 = np.dot(R0, r[0:3])
        k0 = np.dot(R0, [0, 0, 1])
        Jn = np.concatenate((np.cross(k0, r0), k0))
        J.append(Jn)
    return np.transpose(np.array(J))

# =============== Pose helpers (????) ===============
def current_pose():
    """Return current p (3,) in meters and R (3x3)."""
    p = foward_kinematic().astype(float)
    R = rotation_matrix(0, DH_size)
    return p, R

def rotvec_error(R_cur, R_des):
    """
    Rotation error as rotation vector in radians:
    e_omega = Log(R_des * R_cur^T)  (axis-angle vector)
    """
    Re = np.dot(R_des, R_cur.T)
    tr = np.trace(Re)
    c = max(min((tr - 1.0) * 0.5, 1.0), -1.0)  # clamp
    angle = np.arccos(c)
    if angle < 1e-9:
        return np.zeros(3)
    denom = 2.0 * np.sin(angle)
    vx = (Re[2,1] - Re[1,2]) / denom
    vy = (Re[0,2] - Re[2,0]) / denom
    vz = (Re[1,0] - Re[0,1]) / denom
    axis = np.array([vx, vy, vz])
    return axis * angle  # radians

def pose_error6(p_des, R_des):
    """Return 6x1 pose error [dp; e_omega] with units [m, m, m, rad, rad, rad]."""
    p_cur, R_cur = current_pose()
    dp = (np.array(p_des, dtype=float) - p_cur)
    e_omega = rotvec_error(R_cur, R_des)
    return np.concatenate((dp, e_omega))

# =============== X(), inverse, IK, MSE (??????) ===============
def X():
    # ??????????????????????????? ??????????? IK ????
    Zero = [0.0, 0.0, 0.0]
    return np.concatenate((foward_kinematic(), Zero))

def inverse(M):
    return np.linalg.pinv(np.array(M))

def inverse_kinematic(p_des, R_des, alpha, theta_deg_dict_or_arr):
    """
    Resolved-rate IK: theta_{k+1} = theta_k + alpha * J^+ * e,
    where e = [dp; e_omega] in [m, m, m, rad, rad, rad]
    """
    e6 = pose_error6(p_des, R_des)
    invJ = inverse(Jacobian_matrix())
    Delta_theta = np.dot(invJ, e6)  # radians
    theta = degarr2radarr(theta_deg_dict_or_arr)
    return theta + alpha * Delta_theta  # radians

def MSE(p_des, R_des):
    e6 = pose_error6(p_des, R_des)
    return float(np.mean(e6**2))

# ======================= Trajectory (unchanged) =======================
def cubic_position(u0, uf, v0, vf, tf, t):
    return u0 + ((3/(tf**2))*(uf-u0)*(t**2)) - ((2/(tf**3))*(uf-u0)*(t**3))

# ======================= New helper functions =======================
def init_task_random(th0, d_xyz, deg=50, joint_n=6):
    for i in range(joint_n):
        th0[i] = rand.randrange(-deg, deg+1)
    for _ in range(2):
        d_xyz.append(rand.randrange(45, 55)*0.01)
    d_xyz.append(0.0375)

def pause_seconds(t):
    t0 = time.time()
    while (time.time() - t0) < t:
        pass

def banner(msg):
    print("\n" + "="*12 + f" {msg} " + "="*12)

# >>> Utilities ?????? DH ??????
def build_ur5_dh_from_deg(th_deg_dict):
    return np.array([
        [0,    0,       0.0892,  -90 + th_deg_dict.get(0)],
        [90,   0,       0,        90 + th_deg_dict.get(1)],
        [0,    0.4251,  0,         th_deg_dict.get(2)],
        [0,    0.3922,  0.110,    -90 + th_deg_dict.get(3)],
        [-90,  0,       0.0948,     th_deg_dict.get(4)],
        [90,   0,       0.07495,    th_deg_dict.get(5)],
        [0,    0,       0.19163,   180]
    ], dtype=float)

def set_DH_from_th(th_deg_dict):
    global DH_table, DH_size
    DH_table = build_ur5_dh_from_deg(th_deg_dict)
    DH_size  = len(DH_table)

def set_sim_joints_from_deg(sim, hdl_j, th_deg_dict):
    for i in range(6):
        sim.setJointTargetPosition(hdl_j[i], th_deg_dict.get(i)*d2r)

# >>> ???????????????????????? (????? a_max) ?????????
def compute_T_used(distance, a_max, T_req):
    if distance <= 1e-9:
        return max(T_req, 0.5)  # ????????????
    T_min = 2.0 * m.sqrt(distance / max(a_max, 1e-9))
    return max(T_req, T_min)

def tri_distance_traveled(t, T, a):
    # ????????????????? ? ???? t (?????????-??????), ???????????/??????????
    half = 0.5 * T
    if t <= half:
        return 0.5 * a * t**2
    else:
        t2 = t - half
        v_peak = a * half
        return (0.5 * a * half**2) + (v_peak * t2) - 0.5 * a * t2**2

# ======================= Main trial =======================
def run_trial(trial_id, sim, hdl_j, hdl_end, tf=20.0, tb=5.0, alpha=0.01, err_thresh=0.004, max_iters=1000,
              a_max=0.8):  # a_max [m/s^2]
    global DH_table, DH_size
    th, th0, thf, d_xyz = {}, {}, {}, []

    init_task_random(th0, d_xyz, 50, 6)
    thf = th0.copy()

    banner(f"TEST CASE #{trial_id}")
    print(f"[Start Angles] q0 (deg): {th0}")
    print(f"[Target Position] XYZ (m): {np.round(np.array(d_xyz),4).tolist()} | tf(req)={tf}s | tb={tb}s | a_max={a_max} m/s^2")

    # ????????????????? + ?????? DH
    set_sim_joints_from_deg(sim, hdl_j, th0)
    set_DH_from_th(th0)

    # ??? R_des = orientation ???????? (???????????????????????)
    p_start, R_start = current_pose()
    p_goal  = np.array(d_xyz, dtype=float)
    R_des   = R_start.copy()

    print("[IK Solver] Searching for final joint configuration (keep orientation fixed)...")
    t_search0 = time.time()
    for it in range(1, max_iters+1):
        set_DH_from_th(thf)
        new_theta_rad = inverse_kinematic(p_goal, R_des, alpha, thf)  # radians
        new_theta_deg = new_theta_rad * r2d
        thf = dict(enumerate(new_theta_deg))
        err = MSE(p_goal, R_des)

        if it % 20 == 0:
            print(f"  Iter {it:4d} | MSE6: {err:.6f}")
        if err <= err_thresh:
            break

    print(f"[IK Completed] Time: {time.time()-t_search0:.2f}s | Iterations: {it}")
    print("[Final Angles] qf (deg):", {i: round(thf[i], 2) for i in range(6)})

    # ?????????????????????? a_max ?????????? D (?????????????????)
    D = float(np.linalg.norm(p_goal - p_start))
    T_used = compute_T_used(D, a_max, tf)
    a_used = (4.0 * D) / (T_used**2) if D > 1e-9 else a_max  # a ??????????? D ?????????????????????????? T_used
    print(f"[Time Scaling] D={D:.4f} m | T_used={T_used:.3f} s | a_used={a_used:.4f} m/s^2")

    # ??????????????????????????????, orientation ??????
    print("[Motion] Executing Cartesian straight-line with triangular time-scaling (fixed orientation)...")
    t0 = time.time()
    t = 0.0
    th_cur = th0.copy()
    set_DH_from_th(th_cur)

    ik_inner_steps = 2
    alpha_track = max(alpha, 0.02)  # ??????????????????????????

    while t < T_used:
        s_abs = tri_distance_traveled(t, T_used, a_used)
        s = 0.0 if D < 1e-9 else np.clip(s_abs / D, 0.0, 1.0)
        p_des = p_start + (p_goal - p_start) * s

        # Resolved-rate IK ????????????? ? ?????????????
        for _ in range(ik_inner_steps):
            set_DH_from_th(th_cur)
            new_theta_rad = inverse_kinematic(p_des, R_des, alpha_track, th_cur)
            new_theta_deg = new_theta_rad * r2d
            th_cur = dict(enumerate(new_theta_deg))

        set_sim_joints_from_deg(sim, hdl_j, th_cur)

        if int(t) % 5 == 0:
            print(f"  t={t:6.2f}s | s={s:5.3f} | pos={np.round(p_des,4).tolist()} | orient=fixed")

        t = time.time() - t0
        sim.switchThread()

    # ????????????????????
    set_sim_joints_from_deg(sim, hdl_j, thf)
    print("[Motion Completed] Robot reached target pose with fixed orientation.")

# ======================= Simulation callbacks =======================
def sysCall_init():
    global sim
    sim = require('sim')

def sysCall_thread():
    hdl_j={}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    for k in range(1, 4):
        run_trial(
            k, sim, hdl_j, hdl_end,
            tf=20.0,        # ?????????????????????? (?????????????????? T_used ?????????)
            tb=5.0,
            alpha=0.01,
            err_thresh=0.004,
            max_iters=1000,
            a_max=0.8       # ?????????????????????? [m/s^2]
        )
        pause_seconds(2)

    banner("ALL TESTS COMPLETED SUCCESSFULLY")
