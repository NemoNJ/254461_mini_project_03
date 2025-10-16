import time
import numpy as np
import math as m
import random as rand

# ======================= Global constants/state =======================
pi  = np.pi
d2r = pi/180
r2d = 1/d2r

DH_table = None
DH_size  = 0
first    = 0
last     = 0

Pe   = np.array([0, 0, 0, 1], dtype=float)
Ende = np.array([0, 0, 0.19163, 1], dtype=float)

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
        r  = np.dot(Homogeneous_trans_matrix(i, DH_size - 1), Ende)    # iri
        R0i = rotation_matrix(0, i)                        # 0iR
        r0  = np.dot(R0i, r[0:3])                          # 0ri
        k   = np.array([0,0,1], dtype=float)               # iki
        k0  = np.dot(R0i, k)                               # 0ki
        J_cross = np.cross(k0, r0)                         # 0ki x 0ri
        Jn = np.concatenate((J_cross, k0))
        J.append(Jn)
    J = np.array(J).T
    return J

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

def cubic_position(u0, uf, v0, vf, tf, t):
    return u0 + ((3 / (tf**2)) * (uf - u0) * (t**2)) - ((2 / (tf**3)) * (uf - u0) * (t**3))

def cubic_velocity(u0, uf, v0, vf, tf, t):
    return ((6 / (tf**2)) * (uf - u0) * t) - ((6 / (tf**3)) * (uf - u0) * (t**2))

def cubic_acceleration(u0, uf, v0, vf, tf, t):
    return ((6 / (tf**2)) * (uf - u0)) - ((12 / (tf**3)) * (uf - u0) * t)

def cubic_debug(u0, uf, v0, vf, tf, t):
    print(f"time : {t} , Position : {cubic_position(u0,uf,v0,vf,tf,t)} , "
          f"Velocity : {cubic_velocity(u0,uf,v0,vf,tf,t)} , "
          f"Accelerate : {cubic_acceleration(u0,uf,v0,vf,tf,t)} ")


a_lim = np.array([80, 80, 100, 150, 150, 200], dtype=float) 

def start2stop(th0, thf, deg, joint_n):
    for i in range(joint_n):
        th0[i] = rand.randrange(-deg, deg+1)  
        thf[i] = rand.randrange(-deg, deg+1)

def choose_T_from_acc_limit(th0, thf, a_lim, T_min=2.0):
    dtheta = np.array([thf[i]-th0[i] for i in range(6)], dtype=float)
    T_need = np.sqrt(6.0*np.abs(dtheta) / np.maximum(a_lim, 1e-6))
    T = float(np.max(np.nan_to_num(T_need, nan=0.0))) 
    return max(T, T_min)  



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

    print("\n========= JOINT-SPACE: 3 cases (cubic v0=vf=0) =========")

    for ci in range(1, 4):
        # ------------ สุ่ม start/stop joints ------------
        th0, thf, th = {}, {}, {}
        start2stop(th0, thf, 60, 6)

        # ------------ คำนวณ tf จากลิมิตเร่ง joint ------------
        tf = choose_T_from_acc_limit(th0, thf, a_lim, T_min=4.0)
        dtheta = np.array([thf[i] - th0[i] for i in range(6)], dtype=float)
        a_max_est = 6.0*np.abs(dtheta)/(tf**2)  # deg/s^2

        print(f"\n---------- Case {ci}/3 ----------")
        print("Start :", th0)
        print("Stop  :", thf)
        print("Times (auto from a_max):", round(tf,3), "s")
        print("a_max (theoretical) per joint:", a_max_est.round(2).tolist())
        print("a_lim:", a_lim.tolist())
        v0_check = [cubic_velocity(th0.get(i), thf.get(i), 0, 0, tf, 0.0) for i in range(6)]
        vf_check = [cubic_velocity(th0.get(i), thf.get(i), 0, 0, tf, tf) for i in range(6)]
        print("max |v(0)| =", max(abs(v) for v in v0_check), "deg/s")
        print("max |v(tf)|=", max(abs(v) for v in vf_check), "deg/s")
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], th0.get(i) * d2r)
        sim.switchThread()

        print("Start moving...")
        t = 0.0
        t1 = time.time()
        while t < tf:
            for i in range(6):
                th[i] = cubic_position(th0.get(i), thf.get(i), 0, 0, tf, t)
                sim.setJointTargetPosition(hdl_j[i], th.get(i) * d2r)

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

            # time & yield
            t = time.time() - t1
            sim.switchThread()

        print(f"Stop (Case {ci})")

        # ------------ หน่วงก่อนเคสถัดไป ------------
        wait_time = 3.0
        print(f"Waiting {wait_time}s before next case...")
        t_wait0 = time.time()
        while (time.time() - t_wait0) < wait_time:
            sim.switchThread()

    print("\nAll 3 cases done.")
    pass
