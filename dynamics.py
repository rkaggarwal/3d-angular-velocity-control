# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 08:11:24 2021

@author: raggarwal
"""

from casadi import *

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


def dq_dt_moments(q, m, Ixx, Iyy, Izz):
    """
    Euler's equation, but inputs are body-aligned moments
    """

    # Unpack state
    omega_x = q[0];
    omega_y = q[1];
    omega_z = q[2];
    
    # Unpack controls
    Mx = m[0];
    My = m[1];
    Mz = m[2];

    omega_x_dot = (Mx - (Izz - Iyy)*omega_y*omega_z)/Ixx;
    omega_y_dot = (My - (Ixx - Izz)*omega_z*omega_x)/Iyy;
    omega_z_dot = (Mz - (Iyy - Ixx)*omega_x*omega_y)/Izz;
    
    dq_dt = vertcat(omega_x_dot, omega_y_dot, omega_z_dot);
    
    return dq_dt