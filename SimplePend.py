# Code for solving the EoM of a simple pendulm.
from numpy import array, cross, matmul, zeros, dot
from numpy import eye, linalg, pi, linspace, savetxt
from numpy.linalg import norm
from math import pi
from scipy.integrate import solve_ivp
from Basics import make_tilde, get_TransMat

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

global RvecAN

nbdys = 2
massA = 10
inertiaA33 = 0.02
InertiaA = zeros((3,3))
InertiaA[2,2] = inertiaA33
lengthA = 5
localAN = array([0,0.5*lengthA,0])
RvecAN = array([0,0,0])
ForceG_A = massA*array([0,-9.81,0])
Torque = 40 # N-m
Torvec = array([0,0,Torque]) 
thetaA_init = 10*(pi/180)
dthetaA_init = 50*pi/180
time0 = 0
timemax = 10
times = linspace(time0,timemax,100)

def simplepend_ode(t,y):
    global RvecAN
    thetaA = y[0]
    dthetaA = y[1]

    # Direct equation:
    # Coeff = inertiaA33 + (massA*lengthA*lengthA*0.25)
    # RHS = Torque - (0.5*massA*9.81*lengthA*sin(thetaA))
    # d2thetaA = RHS/Coeff

    # Position Analysis:
    Pos = position_analysis(thetaA)
    
    # Jacobian / Partial Velocities
    Vel, omega  = velocity_analysis(1.0)
    JvecA = omega
    JvecAO = Vel[3]
    
    # Velocity analysis and Acclerations (omega Component)
    Vel, omega  = velocity_analysis(dthetaA)
    temp_d2thetaA = 0
    Acc, alp = accelera_analysis(temp_d2thetaA, omega)
    AccAO_omg = Acc[3]

    # # Newton-Euler Equations:
    # Amat = zeros((3,3))
    # bvec = zeros((3,1))
    # Amat[0:2,0:2] = eye(2) 
    # Amat[0:2,2] = -massA*JvecAO[0:2]
    # tildeRvecAN = make_tilde(RvecAN)
    # Amat[2] = array([tildeRvecAN[2,0],tildeRvecAN[2,1],-inertiaA33]) 
    # bvec[0] = -ForceG_A[0] + massA*AccAO_omg[0]
    # bvec[1] = -ForceG_A[1] + massA*AccAO_omg[1]
    # bvec[2] = -Torque
    # sol = linalg.solve(Amat,bvec)
    # d2thetaA = sol[2]

    # # Kane's Equation:
    # # Compute K
    Kvar = dot(ForceG_A,JvecAO) + dot(Torvec,JvecA)
    RHS = -Kvar
    KstarK = -massA*dot(AccAO_omg,JvecAO)
    RHS = RHS - KstarK
    # Compute Coeff of KstarUK
    Coeff = dot(-massA*JvecAO,JvecAO) + dot(matmul(-InertiaA,JvecA),JvecA)
    d2thetaA = RHS/Coeff

    # 
    dy = array([0.0,0.0])
    dy[0] = dthetaA
    dy[1] = d2thetaA

    return dy

def position_analysis(thetaA):
    global RvecAN
    N_A = get_TransMat(thetaA)
    RvecAN = matmul(N_A,localAN)
    
    PosNO = array([0,0,0])
    PosNA = PosNO
    PosAN = PosNA
    PosAO = PosAN - RvecAN

    Pos = [PosNO, PosNA, PosAN, PosAO]
    return Pos

def velocity_analysis(dthetaA):
    global RvecAN
    omegaA = array([0,0,dthetaA]) 
    
    VelNO = array([0,0,0])
    VelNA = VelNO
    VelAN = VelNA
    VelAO = VelAN + cross(omegaA,-RvecAN)

    Vel = [VelNO, VelNA, VelAN, VelAO]
    omega = omegaA
    return Vel, omega

def accelera_analysis(d2thetaA, omega):
    global RvecAN
    omegaA = omega
    alphaA = array([0,0,d2thetaA]) 
    AccNO = array([0,0,0])
    AccNA = AccNO
    AccAN = AccNA
    AccAO = AccAN + cross(alphaA,-RvecAN) + cross(omegaA,cross(omegaA,-RvecAN))

    Acc = [AccNO, AccNA, AccAN, AccAO]
    alpha = [alphaA]
    return Acc, alpha

# Solve the ODE
sol = solve_ivp(simplepend_ode,array([time0, timemax]),array([thetaA_init, dthetaA_init]),t_eval=times)
print(type(sol))
print(sol.message)
print(sol.t)
print(sol.y)

time = sol.t
thetaA = sol.y[0]
dthetaA = sol.y[1]

datalen = len(time)
d2thetaA = zeros(datalen)
PosAOsave = zeros((3,datalen))

for ii in range(len(time)):

    yvec = array([thetaA[ii],dthetaA[ii]])
    dyvec = simplepend_ode(time[ii],yvec)
    d2thetaA[ii] = dyvec[1]
    Pos = position_analysis(thetaA[ii])
    PosAOsave[:,ii] = Pos[3]
    

fig, ax = plt.subplots(3)   
ax[0].plot(time,thetaA)
ax[1].plot(time,dthetaA)
ax[2].plot(time,d2thetaA)
fig.suptitle('angular pos., vel., and acc. wrt time')
ax[0].grid()
ax[0].set_xlabel('time')
ax[0].set_ylabel('$\\theta_A$')
ax[1].grid()
ax[1].set_xlabel('time')
ax[1].set_ylabel('$\dot{\\theta_A}$')
ax[2].grid()
ax[2].set_xlabel('time')
ax[2].set_ylabel('$\ddot{\\theta_A}$')

plt.show()

# Animate simple pendulum
fig, ax = plt.subplots()
xdata, ydata = [], []
ln1, = ax.plot([], [], 'k')
ln2, = ax.plot([], [], 'ko',mfc='w')

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    return ln1, ln2

def update(frame):
    PosAN = array([0,0,0])
    PosAB = 2*PosAOsave[:,frame]
    plotx1 = array([PosAN[0],PosAB[0]])
    ploty1 = array([PosAN[1],PosAB[1]])
    plotx2 = array([PosAN[0]])
    ploty2 = array([PosAN[1]])
    ln1.set_data(plotx1, ploty1)
    ln2.set_data(plotx2, ploty2)
    return ln1, ln2

ani = FuncAnimation(fig, update, frames=range(len(time)),
                    init_func=init, blit=True, repeat=False)

plt.grid()
plt.show()
#


