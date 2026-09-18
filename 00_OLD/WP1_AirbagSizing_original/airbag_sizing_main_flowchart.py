import matplotlib.pyplot as plt
import numpy as np
from airbag_dynamics import system_dynamics_step
from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step
import matplotlib.ticker as ticker


# Shape / geometry parameters
D0 = 1.00        # [m] initial airbag height (diameter) 
L0 = 1.50        # [m] airbag axial length, fixed
d_or = 0.10      # [m] orifice diameter
A_or = (np.pi / 4.0) * d_or**2
d_fabric = 0.002 # [m] fabric thickness

# Gas parameters (air)
P0 = 101325      # [Pa] initial airbag pressure
T0 = 293.15      # [K] initial airbag temperature
P_open = 130000  # [Pa] threshold pressure for vent opening
R_gas = 286.9    # [J/(kg K)]
gamma = 1.4

# Payload parameters
M_payload = 500
u0 = 7.62

# Environment
g = 9.81
P_amb = 101325

# Time settings
dt = 1e-3
t_max = 0.5

# Initial conditions (state at t = 0)
t = 0.0
h = 0.0
u = u0  # [m/s] impact velocity

# Initialise geometry at t=0 (needed to get A0, V0)
geom0 = airbag_geometry_from_h(h, D0, L0)
A_prev = geom0["A_t"]
V_prev = geom0["V_t"]

# Initialise gas state at t=0
m_gas = P0 * V_prev / (R_gas * T0)
T_gas = T0
P_bag = P0  # carry pressure explicitly for dynamics
rho_prev = m_gas / V_prev

# History storage
t_hist, V_hist, P_hist, m_hist, a_hist, u_hist, h_hist = ([] for _ in range(7))

# Store initial point
t_hist.append(t)
V_hist.append(V_prev)
P_hist.append(P_bag)
m_hist.append(m_gas)
a_hist.append(np.nan)  # unknown at t=0 until first dynamics update
u_hist.append(u)
h_hist.append(h)



while t < t_max:

    # =========================
    # 1) System dynamics analysis
    #    uses (P_bag_prev, A_prev) to compute a, then updates u,h
    # =========================
    a, u_new, h_new = system_dynamics_step(
        h_prev=h,
        u_prev=u,
        P_bag=P_bag,     # P0 in the first step
        A_t=A_prev,      # 0 in the first step (calculated)
        M=M_payload,
        g=g,
        P_atm=P_amb,
        dt=dt
    )

    # enforce unilateral contact: no negative compression
    if h_new > D0:
        h_new = D0
    elif h_new <= 0:
        h_new = 0

    # =========================
    # 2) Airbag deformation assumption (geometry)
    #    compute new footprint area and new volume from updated displacement
    # =========================
    geom = airbag_geometry_from_h(h_new, D0, L0)
    A_new = geom["A_t"]     # will be fed to loop in the next step to recalculate acceleration, as A_prev
    V_new = geom["V_t"]

    # =========================
    # 3) Gas thermodynamic analysis (+ orifice flow inside thermo_step)
    #    compute new rho, T, P, m based on new volume
    # =========================
    thermo = thermo_step(
        m_prev=m_gas,
        V_t=V_new,

        gamma=gamma,
        P_amb=P_amb,
        P_open=P_open,

        dt=dt,
        A_or=(np.pi/4) * d_or**2,

        P_prev=P_bag,
        rho_prev=rho_prev,
        T_prev=T_gas,

        R_gas=R_gas
    )

    rho_t = thermo["rho_t"]
    P_new = max(thermo["P_bag"], P_amb)
    m_new = thermo["m_t"]
    dm = thermo["dm"]
    T_new = thermo["T_t"]

    # =========================
    # 4) Termination check 
    # =========================
    print(f"t={t:.4f}, h={h:.3f}, V={V_new:.4e}, P={P_new:.1f}, A={A_new:.3f}, a={a:.2f}, m={m_new:.3f}")
    t_next = t + dt

    # Save histories at the end of the step
    t_hist.append(t_next)
    V_hist.append(V_new)
    P_hist.append(P_new)
    m_hist.append(m_new)
    a_hist.append(a)
    u_hist.append(u_new)
    h_hist.append(h_new)

    if m_new <= 0:
        break

    # =========================
    # 5) Advance "previous" states for next loop
    # =========================
    t = t_next
    h = h_new           # calculated from dynamic deformation at the beginning of the loop step
    u = u_new           # calculated from dynamic deformation at the beginning of the loop step

    A_prev = A_new      # consequence of dynamic step
    V_prev = V_new      # consequence of dynamic step

    P_bag = P_new       # uptades gas-state variables found outside loop, which are then fed to the loop in the dynamic and thermo steps
    rho_prev = rho_t
    V_prev = V_new
    T_gas = T_new
    m_gas = m_new


# Convert lists to arrays
t_hist = np.array(t_hist)
V_hist = np.array(V_hist)
P_hist = np.array(P_hist)
m_hist = np.array(m_hist)
a_hist = np.array(a_hist)
u_hist = np.array(u_hist)
h_hist = np.array(h_hist)

# Plots 
plt.figure(); plt.plot(t_hist, V_hist); plt.xlabel("Time [s]"); plt.ylabel("V [m^3]"); plt.title("Airbag Volume vs Time"); plt.grid(True)


fig, ax = plt.subplots()
ax.plot(t_hist, P_hist)
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$P_{\mathrm{bag}}$ [Pa]")
ax.set_title("Internal Pressure vs Time")
ax.grid(True)
# Force scientific notation with 10^5 scaling
ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
ax.yaxis.get_offset_text().set_fontsize(12)


plt.figure(); plt.plot(t_hist, m_hist); plt.xlabel("Time [s]"); plt.ylabel("m [kg]"); plt.title("Gas Mass vs Time"); plt.grid(True)
plt.figure(); plt.plot(t_hist, a_hist); plt.xlabel("Time [s]"); plt.ylabel("a [m/s^2]"); plt.title("Payload Acceleration vs Time"); plt.grid(True)
plt.figure(); plt.plot(t_hist, u_hist); plt.xlabel("Time [s]"); plt.ylabel("u [m/s]"); plt.title("Payload Speed vs Time"); plt.grid(True)
plt.figure(); plt.plot(t_hist, h_hist); plt.xlabel("Time [s]"); plt.ylabel("h [m]"); plt.title("Payload Displacement vs Time"); plt.grid(True)
plt.show()
