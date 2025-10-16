import time
import numpy as np
import math as m
import random as rand

# ======================= Global constants/state =======================
pi  = np.pi
d2r = pi/180
r2d = 1/d2r

# Robot global state
DH_table = None
DH_size  = 0
first    = 0
last     = 0

# End-effector points
Pe   = np.array([0, 0, 0, 1], dtype=float)
Ende = np.array([0, 0, 0.19163, 1], dtype=float)

# ======================= DH & Euler =======================
def dh_transforms(alpha, a, d, theta):
    ca = m.cos(alpha); sa = m.sin(alpha)
    ct = m.cos(theta); st = m.sin(theta)
    A = np.array([
        [   ct,   -st,    0,     a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa,  ca,  d*ca],
        [    0,     0,   0,     1]
    ], dtype=float)
    return A

def rotation_matrix_to_euler_angles(R):
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = m.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = m.atan2(-R[1, 2], R[2, 2])
        gamma = m.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = m.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma])

def degarr2radarr(deg):
    return np.array([d*d2r for d in deg], dtype=float)

def Homogeneous_trans_matrix(_first, _last):
    global DH_table
    Tn = np.identity(4)
    for t in range(_first, _last):
        alpha_deg, a, d, theta_deg = DH_table[t,0], DH_table[t,1], DH_table[t,2], DH_table[t,3]
        Tn = np.dot(Tn, dh_transforms(alpha_deg*d2r, a, d, theta_deg*d2r))
    return Tn

def rotation_matrix(_first, _last):
    return Homogeneous_trans_matrix(_first, _last)[0:3, 0:3]

def Euler():
    R = rotation_matrix(first, last)
    eul_rad = rotation_matrix_to_euler_angles(R)
    return eul_rad * r2d

def foward_kinematic():
    fw = np.dot(Homogeneous_trans_matrix(first, last), Pe)
    return np.array(fw[:3]).round(6)

def Jacobian_matrix():
    J = []
    for i in range(1, DH_size):  # 1..(n-1)
        r  = np.dot(Homogeneous_trans_matrix(i, DH_size - 1), Ende)
        R0i = rotation_matrix(0, i)
        r0  = np.dot(R0i, r[0:3])
        k   = np.array([0,0,1], dtype=float)
        k0  = np.dot(R0i, k)
        J_cross = np.cross(k0, r0)
        Jn = np.concatenate((J_cross, k0))
        J.append(Jn)
    return np.array(J).T

def X():
    Zero = np.array([0,0,0], dtype=float)
    _ = (Euler() * d2r)
    return np.concatenate((foward_kinematic(), Zero))

def inverse(M):
    return np.linalg.pinv(np.array(M, dtype=float))

def inverse_kinematic(desire_pos, alpha, theta_deg):
    desire_pos = np.array(desire_pos, dtype=float)
    Delta_X = np.subtract(desire_pos, X())
    invJ = inverse(Jacobian_matrix())
    Delta_theta = np.dot(invJ, Delta_X)
    theta = degarr2radarr(theta_deg)
    Ntheta = theta + (alpha * Delta_theta)
    return Ntheta

def debug_robot():
    print("_____________________________________________")
    print("End point position:    {}".format(foward_kinematic().round(4)))
    print("End point orientation: {}".format(Euler().round(2)))
    print("_____________________________________________")
    print("jacobian of robot arm : ")
    print(Jacobian_matrix().round(6))
    print("---------------------------------------------")

# ======================= LSPB (Linear interp. with parabolic blend) =======================
def lspb_init(q0, qf, a_max, T):
    """
    ?????? state ?????? LSPB (v0=vf=0)
    ?????? dict ????????????????????????????????????? position/velocity/acceleration
    """
    q0 = float(q0)
    qf = float(qf)
    dq = qf - q0
    s  = 1.0 if dq >= 0.0 else -1.0
    a  = float(max(abs(a_max), 1e-6))
    T  = float(max(T, 1e-6))
    # ???????????????: T >= 2*sqrt(|dq|/a)
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
    s, a, T, tb, vmax, d_blend = state["s"], state["a"], state["T"], state["tb"], state["vmax"], state["d_blend"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return q0 + s*(0.5*a*t*t)
    elif t <= (T - tb):
        return q0 + s*(d_blend + vmax*(t - tb))
    else:
        td = T - t
        return qf - s*(0.5*a*td*td)

def lspb_velocity(state, t):
    s, a, T, tb, vmax = state["s"], state["a"], state["T"], state["tb"], state["vmax"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return s*(a*t)
    elif t <= (T - tb):
        return s*(vmax)
    else:
        td = T - t
        return s*(a*td)

def lspb_acceleration(state, t):
    s, a, T, tb = state["s"], state["a"], state["T"], state["tb"]
    t = float(min(max(t, 0.0), T))
    if t < tb:
        return s*a
    elif t <= (T - tb):
        return 0.0
    else:
        return -s*a

# ======================= Utils =======================
# ?????????????????????????????? (deg/s^2)
a_lim = np.array([80, 80, 100, 150, 150, 200], dtype=float)

def start2stop(th0, thf, deg, joint_n):
    for i in range(joint_n):
        th0[i] = rand.randrange(-deg, deg+1)
        thf[i] = rand.randrange(-deg, deg+1)

def choose_T_from_acc_limit_LSPB(th0, thf, a_lim, T_min=2.0):
    """
    ???????? LSPB (v0=vf=0) ?????? T >= 2*sqrt(|dq|/a)
    ????? T ?????????????????? joint
    """
    Tmin_list = []
    for i in range(6):
        dq = float(thf[i] - th0[i])
        a  = float(max(a_lim[i], 1e-6))
        Tmin_i = 2.0*m.sqrt(abs(dq)/a)
        Tmin_list.append(Tmin_i)
    T = max(max(Tmin_list), float(T_min))
    return T

# ======================= CoppeliaSim callbacks =======================
def sysCall_init():
    global sim
    sim = require('sim')

def sysCall_thread():
    global DH_table, DH_size, first, last

    # --- handles ---
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    print("\n========= JOINT-SPACE: 3 cases (LSPB: v0=vf=0) =========")

    for ci in range(1, 4):
        # ------------ ???? start/stop joints ------------
        th0, thf, th = {}, {}, {}
        start2stop(th0, thf, 60, 6)

        # ------------ ????? tf ???????????? (LSPB) ------------
        tf = choose_T_from_acc_limit_LSPB(th0, thf, a_lim, T_min=4.0)

        # ???????????????? LSPB ??? joint
        lspb = {}
        tb_list, vmax_list = [], []
        for i in range(6):
            lspb[i] = lspb_init(th0[i], thf[i], a_lim[i], tf)
            tb_list.append(round(lspb[i]["tb"], 4))
            vmax_list.append(round(lspb[i]["vmax"], 3))

        print(f"\n---------- Case {ci}/3 ----------")
        print("Start :", th0)
        print("Stop  :", thf)
        print("Time tf:", round(tf,3), "s")
        print("tb per joint:", tb_list, "s")
        print("v_max magnitude per joint:", vmax_list, "deg/s")
        print("a_lim:", a_lim.tolist())

        # ???????????????
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], th0.get(i) * d2r)
        sim.switchThread()

        # ====== ??????????? "????????????? LSPB" ????????? ======
        print("Start moving (LSPB)...")
        t = 0.0
        t1 = time.time()
        while t < tf:
            for i in range(6):
                th[i] = lspb_position(lspb[i], t)     # ?????????????? .position()
                sim.setJointTargetPosition(hdl_j[i], th.get(i) * d2r)

            # ?????? DH_table ???????
            DH_table = np.array([
                [0,      0,        0.0892,     -90 + th.get(0)],
                [90,     0,        0,           90 + th.get(1)],
                [0,      0.4251,   0,            th.get(2)],
                [0,      0.39215,  0.110,       -90 + th.get(3)],
                [-90,    0,        0.09475,      th.get(4)],
                [90,     0,        0.0,          th.get(5)],
                [0,      0,        0.26658,      180]
            ], dtype=float)
            DH_size = len(DH_table); first = 0; last = DH_size

            t = time.time() - t1
            sim.switchThread()

        print(f"Stop (Case {ci})")

        # ------------ ????????????????? ------------
        wait_time = 3.0
        print(f"Waiting {wait_time}s before next case...")
        t_wait0 = time.time()
        while (time.time() - t_wait0) < wait_time:
            sim.switchThread()

    print("\nAll 3 cases done.")
    pass
