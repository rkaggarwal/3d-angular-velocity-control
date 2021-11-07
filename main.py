# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:53:46 2021

@author: raggarwal
"""

"""
Trajectory optimization to nullify all body angular velocities with only the 
center-mounted thrust-gimbaled engine

For now we just assume this body is floating in space, and we don't mind if
it flips upside down or anything (no notion of orientation).  Just want to get
the omega vector = 0.

Key question: can we control roll velocity (ω_x) without an explicit roll thrust torque (M_x)?
Omega coupling in Euler equations says yes (provided Iyy =/= Izz) -- let's simulate it
"""


from casadi import *
import numpy as np
import matplotlib.pyplot as plt



# %% Initialization

plt.close("all");

# Vehicle inertias (principal axes)
Ixx = 1.0;  # kg-m^2, longitudinal axis
Iyy = 10.0; # kg-m^2
Izz = 11.0; # kg-m^2

# Initial body angular velocities
wx_0 = 1.0; # rad/s
wy_0 = 0.0; # rad/s
wz_0 = 0.0; # rad/s

alpha_lim = np.deg2rad(30); # rad, one-sided gimbal axis 1 limit (pitch)
beta_lim  = np.deg2rad(30); # rad, one sided gimbal axis 2 limit (yaw)

F_thrust_max = 10.0; # N, maximum engine thrust

d = 1.0; # m, distance between engine mount and center of mass

F_thrust_dot_max = 5; # N/s, engine slew rate limit
alpha_dot_max    = np.deg2rad(10); # rad/s, pitch gimbal max slew rate
beta_dot_max     = np.deg2rad(10); # rad/s, yaw gimbal max slew rate




# %% Dynamics equations

def dq_dt(q, u, Ixx, Iyy, Izz, d):
    """
    Euler's equation for rigid body dynamics
    I dw/dt + w x (Iw) = M
    (Transport Theorem applied to angular momentum vector)
    """
    
    # Unpack state
    omega_x = q[0];
    omega_y = q[1];
    omega_z = q[2];
    
    # Unpack raw controls
    Ft    = u[0];
    alpha = u[1];
    beta  = u[2];
    
    # Convert to body forces
    Fx =  Ft*cos(alpha)*cos(beta);
    Fy =  Ft*cos(alpha)*sin(beta);
    Fz = -Ft*sin(alpha);

    # Compute torques with M = r x F
    Mx =  0;
    My =  Fz*d;
    Mz = -Fy*d;
    
    omega_x_dot = (Mx - (Izz - Iyy)*omega_y*omega_z)/Ixx;
    omega_y_dot = (My - (Ixx - Izz)*omega_z*omega_x)/Iyy;
    omega_z_dot = (Mz - (Iyy - Ixx)*omega_x*omega_y)/Izz;
    
    dq_dt = vertcat(omega_x_dot, omega_y_dot, omega_z_dot);
    
    return dq_dt





# %% Nonlinear trajectory optimization

"""
Vehicle state:    q = [wx, wy, wz]
Vehicle control:  u = [F_thrust, alpha, beta]
"""


## Optimization setup ##
N   = 100; # number of horizon steps
dt = .10;  # sec

opti = Opti();
Q = opti.variable(3, N+1);
U = opti.variable(3, N+1);


## Cost function ##
cost = 0;
for i in range(0, N+1):
    # lQR-like cost
    cost += Q[0, i]**2 + Q[1, i]**2 + Q[2, i]**2;
    cost += .10*(U[0, i]**2 + U[1, i]**2 + U[2, i]**2);
    
opti.minimize(cost);


## Constraints ##
opti.subject_to(opti.bounded(0,          U[0, :], F_thrust_max));
opti.subject_to(opti.bounded(-alpha_lim, U[1, :], alpha_lim));
opti.subject_to(opti.bounded(-beta_lim,  U[2, :], beta_lim));

for k in range(N):
    # Trapezoidal collocation
    # Assumes u is piecewise linear and z is piecewise quadratic
    f_k   = dq_dt(Q[:, k],   U[:, k], Ixx, Iyy, Izz, d);
    f_kp1 = dq_dt(Q[:, k+1], U[:, k+1], Ixx, Iyy, Izz, d);
    opti.subject_to(Q[:, k+1] - Q[:, k] == 1/2*dt*(f_k + f_kp1));
    
    # Slew rate constraints on control
    opti.subject_to(opti.bounded(-F_thrust_dot_max*dt, U[0, k+1] - U[0, k], F_thrust_dot_max*dt));
    opti.subject_to(opti.bounded(-alpha_dot_max*dt,    U[1, k+1] - U[1, k], alpha_dot_max*dt));
    opti.subject_to(opti.bounded(-beta_dot_max*dt,     U[2, k+1] - U[2, k], beta_dot_max*dt));


## Boundary conditions ##

# Vehicle must start with initial angular velocity vector
opti.subject_to(Q[0, 0] == wx_0);
opti.subject_to(Q[1, 0] == wy_0);
opti.subject_to(Q[2, 0] == wz_0);

# No angular velocity at end of horizon
opti.subject_to(Q[0, N] == 0);
opti.subject_to(Q[1, N] == 0);
opti.subject_to(Q[2, N] == 0);

# Start and end with no thrust, and no gimbal angles
opti.subject_to(U[0, 0] == 0);
opti.subject_to(U[1, 0] == 0);
opti.subject_to(U[2, 0] == 0);

opti.subject_to(U[0, N] == 0);
opti.subject_to(U[1, N] == 0);
opti.subject_to(U[2, N] == 0);


## Initial guess ##
opti.set_initial(Q[0, :], np.linspace(wx_0, 0, N+1));
opti.set_initial(Q[1, :], np.linspace(wy_0, 0, N+1));
opti.set_initial(Q[2, :], np.linspace(wz_0, 0, N+1));

opti.set_initial(U[0, :], np.linspace(0, F_thrust_max, N+1));
opti.set_initial(U[1, :], np.linspace(0, .1, N+1));
opti.set_initial(U[2, :], np.linspace(0, .1, N+1));


## Solve! ##
p_opts = {"expand": True}
s_opts = {"print_level" : 5, "max_iter" : 1000};
opti.solver("ipopt", p_opts, s_opts)
sol = opti.solve()
    

## Process results ##
T_sol = np.linspace(0, dt*N, N+1);        
Q_sol = sol.value(Q)
U_sol = sol.value(U)




# %% Plot results

fig1, ax1 = plt.subplots(2, 1, constrained_layout = True);
fig1.suptitle("Vehicle Trajectory");

ax1[0].set_title("States");
ax1[0].set_ylabel("ω [rad/s]")
ax1[0].plot(T_sol, Q_sol[0, :], label = "ω_x");
ax1[0].plot(T_sol, Q_sol[1, :], label = "ω_y");
ax1[0].plot(T_sol, Q_sol[2, :], label = "ω_z");

ax1[1].set_title("Controls");
ax1[1].set_ylabel("Value")
ax1[1].plot(T_sol, U_sol[0, :], label = "F_t [N]");
ax1[1].plot(T_sol, np.rad2deg(U_sol[1, :]), label = "alpha [deg]");
ax1[1].plot(T_sol, np.rad2deg(U_sol[2, :]), label = "beta [deg]");

ax1[0].legend(loc = "best");
ax1[1].legend(loc = "best");
