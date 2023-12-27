# Code for solving the EoM of double pendulm.
from numpy import array, cross, matmul, zeros, dot
from numpy import eye, linalg, pi, linspace, savetxt
from numpy.linalg import norm
from math import pi
from scipy.integrate import solve_ivp
from Basics import make_tilde, get_TransMat

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

global RvecAN, RvecAB, RvecBA
nbdys = 3
npnts = 7
nvars = 2
massA = 10
inertiaA33 = 0.02
InertiaA = zeros((3,3))
InertiaA[2,2] = inertiaA33
lengthA = 5
localAN = array([0,0.5*lengthA,0])
RvecAN = zeros((1,3))
localAB = array([0,-0.5*lengthA,0])
RvecAB = zeros((1,3))
ForceG_A = massA*array([0,-9.81,0])
TorqueA = 40 # N-m
TorvecA = array([0,0,TorqueA]) 
thetaA_init = 10*(pi/180)
dthetaA_init = 50*pi/180

massB = 8
inertiaB33 = 0.01
InertiaB = zeros((3,3))
InertiaB[2,2] = inertiaB33
lengthB = 3
localBA = array([0,0.5*lengthB,0])
RvecBA = zeros((1,3))
ForceG_B = massB*array([0,-9.81,0])
TorqueB = 20 # N-m
TorvecB = array([0,0,TorqueB]) 
thetaB_init = 30*(pi/180)
dthetaB_init = 10*pi/180

time0 = 0
timemax = 10
times = linspace(time0,timemax,200)

def doublepend_ode(t,y):
    global RvecAN, RvecAB, RvecBA
    thetaA = y[0]
    thetaB = y[1]
    dthetaA = y[2]
    dthetaB = y[3]

    # Position Analysis:
    theta = [thetaA,thetaB]
    Pos  = position_analysis(theta)

    # Jacobian / Partial Velocities
    JvecPnts = zeros((nvars,npnts,3))
    JvecBdys = zeros((nvars,nbdys-1,3))
    for ii in range(nvars):
        dthetas_for_Js = zeros((1,nvars))
        dthetas_for_Js[0,ii] = 1
        Vel, omega  = velocity_analysis(dthetas_for_Js)
        JvecPnts[ii,:,:] = Vel
        JvecBdys[ii,:,:] = omega
    
    # Velocity analysis and Acclerations (omega Component)
    dtheta = zeros((1,nvars))
    dtheta[0,0] = dthetaA
    dtheta[0,1] = dthetaB
    Vel, omega = velocity_analysis(dtheta)
    zero_d2thetas = zeros((1,nvars))
    Acc, alps = accelera_analysis(zero_d2thetas, omega)
    AccAO_omega = Acc[3]
    AccBO_omega = Acc[6]

    JvecAO_thetaA = JvecPnts[0,3,:]
    JvecAO_thetaB = JvecPnts[1,3,:]
    JvecBO_thetaA = JvecPnts[0,6,:]
    JvecBO_thetaB = JvecPnts[1,6,:]
    JvecA_thetaA = JvecBdys[0,0,:]
    JvecA_thetaB = JvecBdys[1,0,:]
    JvecB_thetaA = JvecBdys[0,1,:]
    JvecB_thetaB = JvecBdys[1,1,:]
    
    # # BEGIN: Sol. using Newton-Euler Equations
    # Amat = zeros((6,6))
    # bvec = zeros((6,1))

    # Amat[0:2,0:2] = eye(2)
    # Amat[0:2,2:4] = eye(2)
    # Amat[0:2,4] = -massA*JvecAO_thetaA[0:2]
    # Amat[0:2,5] = -massA*JvecAO_thetaB[0:2]
    # bvec[0:2,0] = -ForceG_A[0:2] + massA*AccAO_omega[0:2]

    # Amat[2:4,2:4] = -eye(2)
    # Amat[2:4,4] = -massB*JvecBO_thetaA[0:2]
    # Amat[2:4,5] = -massB*JvecBO_thetaB[0:2]
    # bvec[2:4,0] = -ForceG_B[0:2] + massB*AccBO_omega[0:2]

    # tildeRvecAN = make_tilde(RvecAN)
    # tildeRvecAB = make_tilde(RvecAB)
    # tildeRvecBA = make_tilde(RvecBA)
    
    # Amat[4,0:2] = tildeRvecAN[2,0:2]
    # Amat[4,2:4] = tildeRvecAB[2,0:2]
    # Amat[4,4] = -inertiaA33
    # bvec[4,0] = -TorqueA

    # Amat[5,2:4] = -tildeRvecBA[2,0:2]
    # Amat[5,5] = -inertiaB33
    # bvec[5,0] = -TorqueB

    # sol = linalg.solve(Amat,bvec)
    # d2thetaA = sol[4]
    # d2thetaB = sol[5]
    # # END: Sol. using Newton-Euler equations 

    # # BEGIN: Kane's Equation:
    Jmat = zeros((2,2))
    Kvec = zeros((2,1))

    for varno in range(nvars):

            if(varno==0):
                JvecAO_theta = JvecAO_thetaA
                JvecA_theta = JvecA_thetaA
                JvecBO_theta = JvecBO_thetaA
                JvecB_theta = JvecB_thetaA
            elif(varno==1):
                JvecAO_theta = JvecAO_thetaB
                JvecBO_theta = JvecBO_thetaB
                JvecA_theta = JvecA_thetaB
                JvecB_theta = JvecB_thetaB
            
            Kvec[varno,0] = Kvec[varno,0] + dot((ForceG_A + (-massA*AccAO_omega)),JvecAO_theta) +\
                    dot(TorvecA, JvecA_theta)
            Kvec[varno,0] = Kvec[varno,0] + dot((ForceG_B + (-massB*AccBO_omega)),JvecBO_theta) +\
                    dot(TorvecB, JvecB_theta) 
        
            Jmat[varno,0] = Jmat[varno,0] + dot((massA*JvecAO_thetaA),JvecAO_theta) +\
                        dot(matmul(InertiaA,JvecA_thetaA),JvecA_theta)
            Jmat[varno,0] = Jmat[varno,0] + dot((massB*JvecBO_thetaA),JvecBO_theta) +\
                        dot(matmul(InertiaB,JvecB_thetaA),JvecB_theta)
            
            Jmat[varno,1] = Jmat[varno,1]  + dot((massA*JvecAO_thetaB),JvecAO_theta) +\
                        dot(matmul(InertiaA,JvecA_thetaB),JvecA_theta)
            Jmat[varno,1] = Jmat[varno,1]  + dot((massB*JvecBO_thetaB),JvecBO_theta) +\
                        dot(matmul(InertiaB,JvecB_thetaB),JvecB_theta)

    
    sol = linalg.solve(Jmat,Kvec)
    d2thetaA = sol[0]
    d2thetaB = sol[1]
    # END: Kane's Equations

    dy = array([0.0,0.0,0.0,0.0])
    dy[0] = dthetaA
    dy[1] = dthetaB
    dy[2] = d2thetaA
    dy[3] = d2thetaB

    return dy

def position_analysis(theta):
    global RvecAN, RvecAB, RvecBA

    thetaA = theta[0]
    thetaB = theta[1]
    
    N_A = get_TransMat(thetaA)
    N_B = get_TransMat(thetaB)

    RvecAN = matmul(N_A,localAN)
    RvecAB = matmul(N_A,localAB)
    RvecBA = matmul(N_B,localBA)
    
    PosNO = array([0,0,0])
    PosNA = PosNO
    PosAN = PosNA
    PosAO = PosAN - RvecAN
    PosAB = PosAO + RvecAB
    PosBA = PosAB
    PosBO = PosBA - RvecBA

    Pos = [PosNO, PosNA, PosAN, PosAO, PosAB, PosBA, PosBO]
    return Pos

def velocity_analysis(dtheta):
    global RvecAN, RvecAB, RvecBA

    dthetaA = dtheta[0,0]
    dthetaB = dtheta[0,1]
    
    omegaA = array([0,0,dthetaA])
    omegaB = array([0,0,dthetaB])    

    VelNO = array([0,0,0])
    VelNA = VelNO
    VelAN = VelNA
    VelAO = VelAN + cross(omegaA,-RvecAN)
    VelAB = VelAO + cross(omegaA,+RvecAB)
    VelBA = VelAB 
    VelBO = VelBA + cross(omegaB,-RvecBA) 
    
    Vel = [VelNO, VelNA, VelAN, VelAO, VelAB, VelBA, VelBO]
    omega = [omegaA,omegaB]
    return Vel, omega

def accelera_analysis(d2theta, omega):
    global RvecAN, RvecAB, RvecBA
  
    omegaA = omega[0]
    omegaB = omega[1]
    
    d2thetaA = d2theta[0,0]
    d2thetaB = d2theta[0,1]
    alphaA = array([0,0,d2thetaA])
    alphaB = array([0,0,d2thetaB])  

    AccNO = array([0,0,0])
    AccNA = AccNO
    AccAN = AccNA
    AccAO = AccAN + cross(alphaA,-RvecAN) + cross(omegaA,cross(omegaA,-RvecAN))
    AccAB = AccAO + cross(alphaA,+RvecAB) + cross(omegaA,cross(omegaA,+RvecAB))
    AccBA = AccAB
    AccBO = AccBA + cross(alphaB,-RvecBA) + cross(omegaB,cross(omegaB,-RvecBA))

    Acc = [AccNO, AccNA, AccAN, AccAO, AccAB, AccBA, AccBO]
    alpha = [alphaA,alphaB]
    return Acc, alpha

# Solve the ODE
initvalues = array([thetaA_init, thetaB_init, dthetaA_init, dthetaB_init])

sol = solve_ivp(doublepend_ode,array([time0, timemax]),initvalues,t_eval=times)
print(type(sol))
print(sol.message)
print(sol.t)
print(sol.y)


# Compute all data
time = sol.t
thetaA = sol.y[0]
thetaB = sol.y[1]
dthetaA = sol.y[2]
dthetaB = sol.y[3]

datalen = len(time)
d2thetaA = zeros(datalen)
d2thetaB = zeros(datalen)

PosABsave = zeros((3,datalen))
PosBOsave = zeros((3,datalen))



for ii in range(len(time)):

    yvec = array([thetaA[ii],thetaB[ii],dthetaA[ii],dthetaB[ii]])
    dyvec = doublepend_ode(time[ii],yvec)
    d2thetaA[ii] = dyvec[2]
    d2thetaB[ii] = dyvec[3]

    theta = [thetaA[ii],thetaB[ii]]
    Pos = position_analysis(theta)
    PosABsave[:,ii] = Pos[4]
    PosBOsave[:,ii] = Pos[6]
    

fig, ax = plt.subplots(3)   
ax[0].plot(time,thetaA,label="$\\theta_A$")
ax[0].plot(time,thetaB,label="$\\theta_B$")
ax[0].legend()
ax[1].plot(time,dthetaA,label="$\dot{\\theta_A}$")
ax[1].plot(time,dthetaB,label="$\dot{\\theta_B}$")
ax[1].legend()
ax[2].plot(time,d2thetaA,label="$\ddot{\\theta_A}$")
ax[2].plot(time,d2thetaB,label="$\ddot{\\theta_B}$")
ax[2].legend()
fig.suptitle('angular pos., vel., and acc. wrt time')
ax[0].grid()
ax[0].set_xlabel('time')
ax[0].set_ylabel('rad')
ax[1].grid()
ax[1].set_xlabel('time')
ax[1].set_ylabel('rad/sec')
ax[2].grid()
ax[2].set_xlabel('time')
ax[2].set_ylabel('rad/sec^2')
plt.show()



##################
## Animate double pendulum
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
    PosAB = PosABsave[:,frame]
    PosBC = PosAB + 2*(PosBOsave[:,frame]-PosAB)
    plotx1 = array([PosAN[0],PosAB[0],PosBC[0]])
    ploty1 = array([PosAN[1],PosAB[1],PosBC[1]])
    plotx2 = array([PosAN[0],PosAB[0]])
    ploty2 = array([PosAN[1],PosAB[1]])
    ln1.set_data(plotx1, ploty1)
    ln2.set_data(plotx2, ploty2)
    return ln1, ln2

ani = FuncAnimation(fig, update, frames=range(len(time)),
                    init_func=init, blit=True, repeat=False)

plt.grid()
plt.show()
#

