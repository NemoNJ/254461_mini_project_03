import time
import numpy as np
import math as m
import random as rand

pi = np.pi
d2r = pi/180
r2d = 1/d2r

DH_table = None
DH_size  = 0
Pe   = np.array([0, 0, 0, 1], dtype=float)
Ende = np.array([0, 0, 0.19163, 1], dtype=float)

def lspb_init(q0, qf, a_max, T):
    q0 = float(q0)
    qf = float(qf)
    dq = qf - q0
    s  = 1.0 if dq >= 0.0 else -1.0
    a  = float(max(abs(a_max), 1e-6))
    T  = float(max(T, 1e-6))
    # เงื่อนไขขั้นต่ำ: T >= 2*sqrt(|dq|/a)
    Tmin = 2.0*m.sqrt(abs(dq)/a) if a > 0 else T
    if T < Tmin:
        T = Tmin
    # tb = 0.5*(T - sqrt(T^2 - 4|dq|/a))
    disc = max(T*T - 4.0*abs(dq)/a, 0.0)
    tb = 0.5*(T - m.sqrt(disc))
    vmax = a * tb
    d_blend = 0.5 * a * tb * tb
    Tc = T - 2.0 * tb
    return {
        "q0": q0, "qf": qf, "dq": dq, "s": s, "a": a, "T": T,
        "tb": tb, "vmax": vmax, "d_blend": d_blend, "Tc": Tc
    }

def lspb_position(state, t):
    q0, qf = state["q0"], state["qf"]
    sgn, a, T, tb, vmax, d_blend = state["s"], state["a"], state["T"], state["tb"], state["vmax"], state["d_blend"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return q0 + sgn*(0.5*a*t*t)
    elif t <= (T - tb):
        return q0 + sgn*(d_blend + vmax*(t - tb))
    else:
        td = T - t
        return qf - sgn*(0.5*a*td*td)

def lspb_velocity(state, t):
    sgn, a, T, tb, vmax = state["s"], state["a"], state["T"], state["tb"], state["vmax"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return sgn*(a*t)
    elif t <= (T - tb):
        return sgn*(vmax)
    else:
        td = T - t
        return sgn*(a*td)

def lspb_acceleration(state, t):
    sgn, a, T, tb = state["s"], state["a"], state["T"], state["tb"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return sgn*a
    elif t <= (T - tb):
        return 0.0
    else:
        return -sgn*a

def choose_LSPB_given_T(distance, tf, a_max_linear):
    D = float(max(distance, 1e-9))
    tf = float(max(tf, 1e-6))
    a_s_needed = 4.0 / (tf*tf)          # ความเร่งของ s(t)
    a_path_needed = a_s_needed * D      # ความเร่งบนเส้นทางจริง
    exceeded = a_path_needed > a_max_linear + 1e-9
    return tf, a_s_needed, exceeded

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

# =============== Pose helpers ===============
def current_pose():
    p = foward_kinematic().astype(float)
    R = rotation_matrix(0, DH_size)
    return p, R

def rotvec_error(R_cur, R_des):
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
    p_cur, R_cur = current_pose()
    dp = (np.array(p_des, dtype=float) - p_cur)
    e_omega = rotvec_error(R_cur, R_des)
    return np.concatenate((dp, e_omega))

# =============== X(), inverse, IK, MSE ===============
def X():
    Zero = [0.0, 0.0, 0.0]
    return np.concatenate((foward_kinematic(), Zero))

def inverse(M):
    return np.linalg.pinv(np.array(M))

def inverse_kinematic(p_des, R_des, alpha, theta_deg_dict_or_arr):
    e6 = pose_error6(p_des, R_des)
    invJ = inverse(Jacobian_matrix())
    Delta_theta = np.dot(invJ, e6)  # radians
    theta = degarr2radarr(theta_deg_dict_or_arr)
    return theta + alpha * Delta_theta  # radians

def MSE(p_des, R_des):
    e6 = pose_error6(p_des, R_des)
    return float(np.mean(e6**2))

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

def build_ur5_dh_from_deg(th_deg_dict):
    return np.array([
        [0,    0,       0.0892,  -90 + th_deg_dict.get(0)],
        [90,   0,       0,        90 + th_deg_dict.get(1)],
        [0,    0.4251,  0,         th_deg_dict.get(2)],
        [0,    0.39215,  0.110,    -90 + th_deg_dict.get(3)],
        [-90,  0,       0.09475,     th_deg_dict.get(4)],
        [90,   0,       0.0,    th_deg_dict.get(5)],
        [0,    0,       0.26658,   180]
    ], dtype=float)

def set_DH_from_th(th_deg_dict):
    global DH_table, DH_size
    DH_table = build_ur5_dh_from_deg(th_deg_dict)
    DH_size  = len(DH_table)

def set_sim_joints_from_deg(sim, hdl_j, th_deg_dict):
    for i in range(6):
        sim.setJointTargetPosition(hdl_j[i], th_deg_dict.get(i)*d2r)

def sample_task_point():
    x = rand.uniform(0.25, 0.60)
    y = rand.uniform(-0.40, 0.60)
    z = rand.uniform(0.05, 0.35)
    return np.array([x, y, z], dtype=float)
  
def solve_IK_to_pose(p_des, R_des, th_seed_deg, alpha=0.03, iters=1200, tol=1e-4):
    th_work = dict(th_seed_deg)  # degrees
    for _ in range(iters):
        set_DH_from_th(th_work)
        e6 = pose_error6(p_des, R_des)
        if np.linalg.norm(e6) <= tol:
            break
        new_theta_rad = inverse_kinematic(p_des, R_des, alpha, th_work)  # radians
        th_work = dict(enumerate(new_theta_rad * r2d))  # back to degrees
    return th_work

def run_trial(trial_id, sim, hdl_j, hdl_end, tf=20.0, tb=5.0, alpha=0.01, err_thresh=0.004, max_iters=1000,
              a_max=0.8):  
    global DH_table, DH_size
    th, th0, thf, d_xyz = {}, {}, {}, []

    init_task_random(th0, d_xyz, 50, 6)
    thf = th0.copy()

    banner(f"TEST CASE #{trial_id}")
    print(f"[Start Joint Seed] q0_seed (deg): {th0}")

    set_sim_joints_from_deg(sim, hdl_j, th0)
    set_DH_from_th(th0)

    p_start_des = sample_task_point()
    p_goal_des  = sample_task_point()

    _, R_start_meas = current_pose()
    R_des = R_start_meas.copy()

    th0 = solve_IK_to_pose(p_start_des, R_des, th0, alpha=0.03, iters=1200, tol=1e-4)
    set_sim_joints_from_deg(sim, hdl_j, th0)
    set_DH_from_th(th0)

    p_start, _ = current_pose()
    p_goal = p_goal_des.copy()

    print(f"[Task Points] p_start={np.round(p_start,4).tolist()}  p_goal={np.round(p_goal,4).tolist()}")

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

    D = float(np.linalg.norm(p_goal - p_start))  # ระยะทาง
    T_used, a_s, exceeded = choose_LSPB_given_T(D, tf, a_max)
    s_traj = lspb_init(0.0, 1.0, a_s, T_used)    # LSPB ของ s(t)
    v_path_max = s_traj["vmax"] * D
    a_path = a_s * D

    print(f"[Time Scaling LSPB] D={D:.4f} m | T_used={T_used:.3f} s | s_tb={s_traj['tb']:.4f} s")
    print(f"  s_vmax={s_traj['vmax']:.4f} | path_vmax≈{v_path_max:.4f} m/s | path_accel≈{a_path:.4f} m/s^2")
    if exceeded:
        print("[WARN] Required path acceleration exceeds a_max to finish within tf.")

    print("[Motion] Executing Cartesian straight-line with LSPB time-scaling (fixed orientation)...")

    try:
        dt = sim.getSimulationTimeStep()
    except:
        dt = 0.01

    t = 0.0
    th_cur = th0.copy()
    set_DH_from_th(th_cur)

    ik_inner_steps = 6
    alpha_track = max(alpha, 0.06)
    max_step_rad = 5.0 * d2r  

    while t < T_used:
        s = lspb_position(s_traj, t)     
        p_des = p_start + (p_goal - p_start) * s

        # Resolved-rate IK ไล่ตาม p_des (orientation คงที่) พร้อม clip step
        for _ in range(ik_inner_steps):
            set_DH_from_th(th_cur)
            e6 = pose_error6(p_des, R_des)
            invJ = inverse(Jacobian_matrix())
            dtheta = np.dot(invJ, e6) * alpha_track  # radians
            dtheta = np.clip(dtheta, -max_step_rad, max_step_rad)
            theta_rad = degarr2radarr(th_cur) + dtheta
            th_cur = dict(enumerate(theta_rad * r2d))

        set_sim_joints_from_deg(sim, hdl_j, th_cur)

        if int(t) % 5 == 0:
            print(f"  t={t:6.2f}s | s={s:5.3f} | pos={np.round(p_des,4).tolist()} | orient=fixed")

        sim.switchThread()
        t += dt

    p_des = p_start + (p_goal - p_start) * 1.0
    for _ in range(ik_inner_steps):
        set_DH_from_th(th_cur)
        e6 = pose_error6(p_des, R_des)
        invJ = inverse(Jacobian_matrix())
        dtheta = np.dot(invJ, e6) * alpha_track
        dtheta = np.clip(dtheta, -max_step_rad, max_step_rad)
        theta_rad = degarr2radarr(th_cur) + dtheta
        th_cur = dict(enumerate(theta_rad * r2d))
    set_sim_joints_from_deg(sim, hdl_j, th_cur)

    # (ออปชัน) ส่งไป qf สุดท้ายที่แก้ด้วย IK ล่วงหน้า หากต้องการให้หยุดตรง joint pose นั้น
    set_sim_joints_from_deg(sim, hdl_j, thf)

    print("[Motion Completed] Robot reached target pose with fixed orientation.")

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
            tf=15.0,       
            tb=5.0,         
            alpha=0.01,
            err_thresh=0.004,
            max_iters=1000,
            a_max=1.2     
        )
        pause_seconds(1.5)

    banner("ALL TESTS COMPLETED SUCCESSFULLY")
