import matplotlib.pyplot as plt
import numpy as np
from airbag_dynamics import system_dynamics_step
from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step



# Shape / geometry parameters
D0 = 1.00        # [m]   Initial airbag height
L0 = 1.50        # [m]   Airbag axial length (horizontal cylindrical airbag)
d_or = 0.10      # [m]   Orifice diameter
d_fabric = 0.002  # [m]  Fabric thickness

# Gas parameters (air)
rho_gas_amb = 1.205     # [kg/m^3] Gas density (ambient air)
P0 = 101325             # [Pa] Initial inflation pressure
T0 = 293.15             # [K] Initial temperature
P_open = 130000         # [Pa] Threshold (vent opening) pressure
R_gas = 286.9           # [J/(kg·K)] Specific gas constant
gamma = 1.4             # [-] Ratio of specific heats

# Payload parameters
M_payload = 0.500         # [kg] Payload mass
u0 = 63               # [m/s] Initial impact velocity

# Environment parameters
g = 9.81                # [m/s^2] Gravitational acceleration
P_amb = 101325          # [Pa] Ambient pressure

# Time settings (time increment and simulation duration)
dt = 0.5 * 1e-2     # [s]
t_max = 0.5   # [s]

# Initial conditions
h = 0.0                                     # h = h_t
u = u0
t = 0.0

# --- Initialisation ---
geom0 = airbag_geometry_from_h(h, D0, L0) 
V_prev = geom0["V_t"]                       # initial airbag volume from geom at h=0 
m_gas = P0 * V_prev / (R_gas * T0)          # initial gas mass (ideal gas)
T_gas = T0                                  # initial gas temperature
rho_prev = m_gas / V_prev
P_bag = P0

# --- History storage ---
t_hist = []
V_hist = []
P_hist = []
m_hist = []
a_hist = []
u_hist = []
h_hist = []


while t < t_max:

    # 1) Geometry from previous loop displacement (uses h at the end of the loop)
    geom = airbag_geometry_from_h(h, D0, L0)
    A_t = geom["A_t"]       # needed for dynamic step (3)
    V_t = geom["V_t"]       # needed for thermo step (2)


    # 2) Thermodynamics
    # parameters from previous step output are written in x_prev, while x_gas contains the new value calculated after the compression
    #thermo = thermo_step(
    #    m_prev=m_gas,          # [kg] stored from previous (thermo) step
    #    T_prev=T_gas,          # [K] stored from previous (thermo) step
    #    V_t=V_t,               # [m^3] new airbag volume calculated in geometry block
#
    #    R_gas=R_gas,           # constant (input parameter)
    #    gamma=gamma,           # constant (input parameter)
    #    P_amb=P_amb,           # constant (input parameter)
    #    P_open=P_open,         # constant (input parameter)
    #    dt=dt,                 # constant (input parameter)
#
    #    A_or=(np.pi/4) * d_or**2,     # orifice area from d_or
    #    T_mode="adiabatic",           # "adiabatic" or "isothermal"
    #    V_prev=V_prev      # store V_prev in loop if using adiabatic, see initialisation
    #)

    thermo = thermo_step(
        m_prev=m_gas,
        V_t=V_t,

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
    P_bag = thermo["P_bag"]
    T_gas = thermo["T_t"]
    m_gas = thermo["m_t"]
    dm = thermo["dm"]


    # 3) Dynamics
    a, u, h = system_dynamics_step(
        h_prev=h,               # old value, from previous loop dynam step (3)
        u_prev=u,               # old value, from previous loop dynam step (3)
        P_bag=P_bag,            # new value, from thermo step (2) (post-compression)
        A_t=A_t,                # new value, from geom step (1) (pre-compression)
        M=M_payload,            # constant (input parameter)
        g=g,                    # constant (input parameter)
        P_atm=P_amb,            # constant (input parameter)
        dt=dt                   # constant (input parameter)
    )

    # --- Save histories for plotting ---
    t_hist.append(t)
    V_hist.append(V_t)
    P_hist.append(P_bag)
    m_hist.append(m_gas)
    a_hist.append(a)
    u_hist.append(u)
    h_hist.append(h)

    # Update previous volume for next step
    V_prev = V_t                     # needed for thermo block, where V_t is overwritten

    t += dt


# --- Convert lists to arrays ---
t_hist = np.array(t_hist)
V_hist = np.array(V_hist)
P_hist = np.array(P_hist)
m_hist = np.array(m_hist)
a_hist = np.array(a_hist)
u_hist = np.array(u_hist)
h_hist = np.array(h_hist)

# --- Plot 1: Volume vs time ---
plt.figure()
plt.plot(t_hist, V_hist)
plt.xlabel("Time [s]")
plt.ylabel("Airbag volume V [m^3]")
plt.title("Airbag Volume vs Time")
plt.grid(True)

# --- Plot 2: Internal pressure vs time ---
plt.figure()
plt.plot(t_hist, P_hist)
plt.xlabel("Time [s]")
plt.ylabel("Internal pressure P_bag [Pa]")
plt.title("Internal Pressure vs Time")
plt.grid(True)

# --- Plot 3: Gas mass vs time ---
plt.figure()
plt.plot(t_hist, m_hist)
plt.xlabel("Time [s]")
plt.ylabel("Gas mass m [kg]")
plt.title("Gas Mass vs Time")
plt.grid(True)

# --- Plot 4: Payload acceleration vs time ---
plt.figure()
plt.plot(t_hist, a_hist)
plt.xlabel("Time [s]")
plt.ylabel("Payload acceleration a [m/s^2]")
plt.title("Payload Acceleration vs Time")
plt.grid(True)

# --- Plot 5: Payload speed vs time ---
plt.figure()
plt.plot(t_hist, u_hist)
plt.xlabel("Time [s]")
plt.ylabel("Payload speed u [m/s]")
plt.title("Payload Speed vs Time")
plt.grid(True)

# --- Plot 6: Payload displacement vs time ---
plt.figure()
plt.plot(t_hist, h_hist)
plt.xlabel("Time [s]")
plt.ylabel("Payload displacement h [m]")
plt.title("Payload Displacement vs Time")
plt.grid(True)

plt.show()
