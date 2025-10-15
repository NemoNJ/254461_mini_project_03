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
class LSPB:
    """
    Linear with Parabolic Blend (start/end v = 0).
    a_max คือลิมิตความเร่งเชิงขนาด (deg/s^2), เครื่องหมายทิศทางจัดการด้วย sign(dq)
    """
    def __init__(self, q0, qf, a_max, T):
        self.q0 = float(q0)
        self.qf = float(qf)
        self.dq = self.qf - self.q0
        self.s  = 1.0 if self.dq >= 0.0 else -1.0
        self.a  = float(max(abs(a_max), 1e-6))
        self.T  = float(max(T, 1e-6))
        # ตรวจสอบเงื่อนไขความเป็นไปได้ของโปรไฟล์: T >= 2*sqrt(|dq|/a)
        Tmin = 2.0*m.sqrt(abs(self.dq)/self.a) if self.a > 0 else self.T
        if self.T < Tmin:
            # ปรับ T ให้ขั้นต่ำที่เป็นไปได้ (กันพัง)
            self.T = Tmin
        # คำนวณ tb และ v_max ตามสมการ LSPB
        # tb = (a*T - sqrt(a^2*T^2 - 4a*|dq|)) / (2a) = 0.5*(T - sqrt(T^2 - 4|dq|/a))
        disc = max(self.T*self.T - 4.0*abs(self.dq)/self.a, 0.0)
        self.tb = 0.5*(self.T - m.sqrt(disc))
        self.vmax = self.a * self.tb  # เป็นบวกเสมอ, เครื่องหมายทิศทางคูณด้วย s ตอนใช้งานจริง

        # ระยะที่วิ่งในเฟสเร่ง (และเฟสหน่วง)
        self.d_blend = 0.5*self.a*self.tb*self.tb
        # เฟสกลางความเร็วคงที่ยาวเท่าไร
        self.Tc = self.T - 2.0*self.tb

    def position(self, t):
        t = float(min(max(t, 0.0), self.T))
        if t < self.tb:
            # accel: q = q0 + 0.5*a*t^2
            return self.q0 + self.s*(0.5*self.a*t*t)
        elif t <= (self.T - self.tb):
            # cruise: q = q0 + d_blend + vmax*(t - tb)
            return self.q0 + self.s*(self.d_blend + self.vmax*(t - self.tb))
        else:
            # decel (mirror): q = qf - 0.5*a*(T - t)^2
            td = self.T - t
            return self.qf - self.s*(0.5*self.a*td*td)

    def velocity(self, t):
        t = float(min(max(t, 0.0), self.T))
        if t < self.tb:
            return self.s*(self.a*t)
        elif t <= (self.T - self.tb):
            return self.s*(self.vmax)
        else:
            td = self.T - t
            return self.s*(self.a*td)

    def acceleration(self, t):
        t = float(min(max(t, 0.0), self.T))
        if t < self.tb:          return self.s*self.a
        elif t <= (self.T-self.tb): return 0.0
        else:                    return -self.s*self.a

# ======================= Utils =======================
# ลิมิตความเร่งสูงสุดของแต่ละข้อ (deg/s^2)
a_lim = np.array([80, 80, 100, 150, 150, 200], dtype=float)

def start2stop(th0, thf, deg, joint_n):
    for i in range(joint_n):
        th0[i] = rand.randrange(-deg, deg+1)
        thf[i] = rand.randrange(-deg, deg+1)

def choose_T_from_acc_limit_LSPB(th0, thf, a_lim, T_min=2.0):
    """
    เงื่อนไข LSPB (v0=vf=0) ต้องมี T >= 2*sqrt(|dq|/a)
    เลือก T ให้ใหญ่พอสำหรับทุก joint
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
        # ------------ สุ่ม start/stop joints ------------
        th0, thf, th = {}, {}, {}
        start2stop(th0, thf, 60, 6)

        # ------------ คำนวณ tf จากลิมิตเร่ง (LSPB) ------------
        tf = choose_T_from_acc_limit_LSPB(th0, thf, a_lim, T_min=4.0)

        # เตรียมอ็อบเจ็กต์ LSPB ต่อ joint
        lspb = {}
        tb_list, vmax_list = [], []
        for i in range(6):
            lspb[i] = LSPB(th0[i], thf[i], a_lim[i], tf)
            tb_list.append(round(lspb[i].tb, 4))
            vmax_list.append(round(lspb[i].vmax, 3))  # เป็นขนาด (บวก)

        print(f"\n---------- Case {ci}/3 ----------")
        print("Start :", th0)
        print("Stop  :", thf)
        print("Time tf:", round(tf,3), "s")
        print("tb per joint:", tb_list, "s")
        print("v_max magnitude per joint:", vmax_list, "deg/s")
        print("a_lim:", a_lim.tolist())

        # ------------ ตั้งค่าเริ่มต้น ------------
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], th0.get(i) * d2r)
        sim.switchThread()

        # ------------ เดินทางด้วย LSPB ------------
        print("Start moving (LSPB)...")
        t = 0.0
        t1 = time.time()
        while t < tf:
            for i in range(6):
                th[i] = lspb[i].position(t)
                sim.setJointTargetPosition(hdl_j[i], th.get(i) * d2r)

            # อัปเดต DH_table เพื่อ debug/ตรวจ FK ได้ถ้าต้องการ
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
