# Code for solving the EoM of fourbar crank-rocker mechanism.
from numpy import array, cross, matmul, zeros, dot
from numpy import eye, linalg, pi, linspace, savetxt
from numpy.linalg import norm
from math import pi, sqrt
from scipy.integrate import solve_ivp
from Basics import make_tilde, get_TransMat, circlecircle, find_ang

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

global RvecAN, RvecAB, RvecBA, RvecBC, RvecCB, RvecCN

nbdys = 4
npnts = 12
nvars = 1

lengthA = 4; lengthB = 8; lengthC = 10; lengthN = 12

massA = 10
inertiaA33 = 0.02
InertiaA = zeros((3,3))
InertiaA[2,2] = inertiaA33
localAN = array([-0.5*lengthA,0,0])
RvecAN = zeros((1,3))
localAB = array([0.5*lengthA,0,0])
RvecAB = zeros((1,3))
ForceG_A = massA*array([0,-9.81,0])
TorqueA = 140 # N-m
TorvecA = array([0,0,TorqueA]) 
thetaA_init = 50*(pi/180)
dthetaA_init = 30*pi/180

massB = 8
inertiaB33 = 0.01
InertiaB = zeros((3,3))
InertiaB[2,2] = inertiaB33
localBA = array([-0.5*lengthB,0,0])
localBC = array([0.5*lengthB,0,0])
RvecBA = zeros((1,3))
RvecBC = zeros((1,3))
ForceG_B = massB*array([0,-9.81,0])

massC = 8
inertiaC33 = 0.01
InertiaC = zeros((3,3))
InertiaC[2,2] = inertiaC33
localCB = array([-0.5*lengthC,0,0])
localCN = array([0.5*lengthC,0,0])
RvecCB = zeros((1,3))
RvecCN = zeros((1,3))
ForceG_C = massC*array([0,-9.81,0])

time0 = 0
timemax = 5
times = linspace(time0,timemax,200)

def position_analysis(theta):

    global RvecAN, RvecAB, RvecBA, RvecBC, RvecCB, RvecCN

    thetaA = theta

    # Find dependent angles
    N_A = get_TransMat(thetaA)
    PosAB = array([0,0,0])
    PosCN = array([0,0,0])
    PosAB = matmul(N_A,-localAN+localAB)
    PosCN = array([lengthN,0,0])
    [xc,yc,flag] = circlecircle(PosAB[0], PosAB[1], lengthB, PosCN[0], PosCN[1], lengthC)
    pnt1 = array([xc[0],yc[0],0])
    pnt2 = array([xc[1],yc[1],0])
    
    
    file_in = open('where.txt', 'r')
    where = file_in.read()
    file_in.close()

    if(where=="atstart"):
        if(yc[0]>yc[1]):
            pnt = pnt1
        else:
            pnt = pnt2

        file = open("where.txt", "w")
        file.write("notatstart")
        file.close()    
    else:
        x = []
        file_in = open('pnt.txt', 'r')
        for line in file_in.readlines():
            x.append(float(line))
        file_in.close()

        dist1 = norm(pnt1-x)
        dist2 = norm(pnt2-x)
        if(dist1<=dist2):
            pnt = pnt1
        elif(dist2<=dist1):
            pnt = pnt2

    savetxt('pnt.txt', pnt)
    thetaB = find_ang(pnt-PosAB)
    thetaC = find_ang(PosCN-pnt)

    N_A = get_TransMat(thetaA)
    RvecAN = matmul(N_A,localAN)
    RvecAB = matmul(N_A,localAB)

    N_B = get_TransMat(thetaB)
    RvecBA = matmul(N_B,localBA)
    RvecBC = matmul(N_B,localBC)

    N_C = get_TransMat(thetaC)
    RvecCB = matmul(N_C,localCB)
    RvecCN = matmul(N_C,localCN)

    PosNO = array([0,0,0])
    PosNA = PosNO
    PosAN = PosNA
    PosAO = PosAN - RvecAN
    PosAB = PosAO + RvecAB
    PosBA = PosAB
    PosBO = PosBA - RvecBA
    PosBC = PosBO + RvecBC
    PosCB = PosBC
    PosCO = PosCB - RvecCB
    PosCN = PosCO + RvecCN
    PosNC = PosCN
    
    Pos = [PosNO, PosNA, PosAN, PosAO, PosAB, PosBA, PosBO, PosBC, PosCB, PosCO, PosCN, PosNC]
    return Pos

def velocity_analysis(dtheta):
    global RvecAN, RvecAB, RvecBA, RvecBC, RvecCB, RvecCN
    dthetaA = dtheta
    omegaA = array([0,0,dthetaA])

    VelAB = zeros((3,1))
    VelAB = cross(omegaA,-RvecAN+RvecAB).reshape(3,1)
    VelCN = zeros((3,1))

    VecB = -RvecBA + RvecBC
    VecC = -RvecCN + RvecCB
    bvec = zeros((2,1))
    Amat = zeros((2,2))
    tildevecB = make_tilde(VecB)
    tildevecC = make_tilde(VecC)
    Amat[0:2,0] = tildevecB[0:2,2]
    Amat[0:2,1] = -tildevecC[0:2,2]

    bvec[0:2] = VelAB[0:2] - VelCN[0:2] 
    sol = linalg.solve(Amat,bvec)
    dthetaB = sol[0]
    dthetaC = sol[1]

    omegaB = array([0,0,dthetaB[0]])    
    omegaC = array([0,0,dthetaC[0]])    

    VelNO = array([0,0,0])
    VelNA = VelNO

    VelAN = VelNA
    VelAO = VelAN + cross(omegaA,-RvecAN)
    VelAB = VelAO + cross(omegaA,+RvecAB)
    
    VelBA = VelAB 
    VelBO = VelBA + cross(omegaB,-RvecBA) 
    VelBC = VelBO + cross(omegaB,RvecBC)

    VelCB = VelBC 
    VelCO = VelCB + cross(omegaC,-RvecCB) 
    VelCN = VelCO + cross(omegaC,RvecCN)
    VelNC = VelCN

    Vel = [VelNO, VelNA, VelAN, VelAO, VelAB, VelBA, VelBO, VelBC, VelCB, VelCO, VelCN, VelNC]
    omega = [omegaA, omegaB, omegaC]
    
    return Vel, omega

def accelera_analysis(d2theta, omega):
    global RvecAN, RvecAB, RvecBA, RvecBC, RvecCB, RvecCN

    PosAB = -RvecAN + RvecAB

    omegaA = omega[0]
    omegaB = omega[1]
    omegaC = omega[2]
    d2thetaA = d2theta
    alphaA = array([0,0,d2thetaA])

    AccAB = cross(alphaA,PosAB) + cross(omegaA,cross(omegaA,PosAB))
    AccAB = AccAB.reshape(3,1)
    AccCN = zeros((3,1))

    VecB = -RvecBA + RvecBC
    VecC = -RvecCN + RvecCB
    bvec = zeros((2,1))
    Amat = zeros((2,2))
    tildevecB = make_tilde(VecB)
    tildevecC = make_tilde(VecC)
    Amat[0:2,0] = tildevecB[0:2,2]
    Amat[0:2,1] = -tildevecC[0:2,2]

    AccomgB = cross(omegaB,cross(omegaB,VecB)).reshape(3,1)
    AccomgC = cross(omegaC,cross(omegaC,VecC)).reshape(3,1)
    
    bvec[0:2] = AccAB[0:2] - AccCN[0:2] 
    bvec[0:2] = bvec[0:2] + AccomgB[0:2] - AccomgC[0:2]
    
    sol = linalg.solve(Amat,bvec)
    d2thetaB = sol[0]
    d2thetaC = sol[1]
    alphaB = array([0,0,d2thetaB[0]])
    alphaC = array([0,0,d2thetaC[0]])

    AccNO = array([0,0,0])
    AccNA = AccNO
    AccAN = AccNA
    AccAO = AccAN + cross(alphaA,-RvecAN) + cross(omegaA,cross(omegaA,-RvecAN))
    AccAB = AccAO + cross(alphaA,+RvecAB) + cross(omegaA,cross(omegaA,+RvecAB))
    AccBA = AccAB
    AccBO = AccBA + cross(alphaB,-RvecBA) + cross(omegaB,cross(omegaB,-RvecBA))
    AccBC = AccBO + cross(alphaB,+RvecBC) + cross(omegaB,cross(omegaB,+RvecBC))
    AccCB = AccBC
    AccCO = AccCB + cross(alphaC,-RvecCB) + cross(omegaC,cross(omegaC,-RvecCB))
    AccCN = AccCO + cross(alphaC,+RvecCN) + cross(omegaC,cross(omegaC,+RvecCN))
    AccNC = AccCN
    
    Acc = [AccNO, AccNA, AccAN, AccAO, AccAB, AccBA, AccBO, AccBC, AccCB, AccCO, AccCN, AccNC]
    alpha = [alphaA,alphaB,alphaC]

    return Acc, alpha

def fourbar_ode(t,y):
    global RvecAN, RvecAB, RvecBA, RvecBC, RvecCB, RvecCN

    thetaA = y[0]
    dthetaA = y[1]

    theta = thetaA
    Pos  = position_analysis(theta)
    # Jacobian / Partial Velocities
    JvecPnts = zeros((nvars,npnts,3))
    JvecBdys = zeros((nvars,nbdys-1,3))

    dthetas_for_Js = 1
    JVel, Jomega  = velocity_analysis(dthetas_for_Js)
    JvecPnts[0,:,:] = JVel
    JvecBdys[0,:,:] = Jomega

    # Velocity analysis and Acclerations (omega Component)
    Vel, omega = velocity_analysis(dthetaA)
    Acc, alp = accelera_analysis(0, omega)
    
    AccAO_omega = Acc[3]
    AccBO_omega = Acc[6]
    AccCO_omega = Acc[9]

    alpA_omega = alp[0]
    alpB_omega = alp[1]
    alpC_omega = alp[2]

    JvecAO = JvecPnts[0,3,:]
    JvecBO = JvecPnts[0,6,:]
    JvecCO = JvecPnts[0,9,:]

    JvecA = JvecBdys[0,0,:]
    JvecB = JvecBdys[0,1,:]
    JvecC = JvecBdys[0,2,:]
    
    ##### BEGIN: Sol. using Newton-Euler Equations
    # Amat = zeros((9,9))
    # bvec = zeros((9,1))

    # Amat[0:2,0:2] = eye(2)
    # Amat[0:2,2:4] = eye(2)
    # Amat[0:2,8] = -massA*JvecAO[0:2]
    # bvec[0:2,0] = -ForceG_A[0:2] + massA*AccAO_omega[0:2]

    # Amat[2:4,2:4] = -eye(2)
    # Amat[2:4,4:6] = eye(2)
    # Amat[2:4,8] = -massB*JvecBO[0:2]
    # bvec[2:4,0] = -ForceG_B[0:2] + massB*AccBO_omega[0:2]

    # Amat[4:6,4:6] = -eye(2)
    # Amat[4:6,6:8] = eye(2)
    # Amat[4:6,8] = -massC*JvecCO[0:2]
    # bvec[4:6,0] = -ForceG_C[0:2] + massC*AccCO_omega[0:2]

    # tildeRvecAN = make_tilde(RvecAN)
    # tildeRvecAB = make_tilde(RvecAB)
    # tildeRvecBA = make_tilde(RvecBA)
    # tildeRvecBC = make_tilde(RvecBC)
    # tildeRvecCB = make_tilde(RvecCB)
    # tildeRvecCN = make_tilde(RvecCN)


    # Amat[6,0:2] = tildeRvecAN[2,0:2]
    # Amat[6,2:4] = tildeRvecAB[2,0:2]
    # Amat[6,8] = -InertiaA[2,2]*JvecA[2]
    # bvec[6,0] = -TorqueA + InertiaA[2,2]*alpA_omega[2]

    # Amat[7,2:4] = -tildeRvecBA[2,0:2]
    # Amat[7,4:6] = tildeRvecBC[2,0:2]
    # Amat[7,8] = -InertiaB[2,2]*JvecB[2]
    # bvec[7,0] = InertiaB[2,2]*alpB_omega[2]

    # Amat[8,4:6] = -tildeRvecCB[2,0:2]
    # Amat[8,6:8] = tildeRvecCN[2,0:2]
    # Amat[8,8] = -InertiaC[2,2]*JvecC[2]
    # bvec[8,0] = InertiaC[2,2]*alpC_omega[2]

    # sol = linalg.solve(Amat,bvec)
    # d2thetaA = sol[8]
    ###### END: Sol. using Newton-Euler equations 

    ### BEGIN: Kane's Equation:
    bvec = 0
    bvec = bvec + dot((ForceG_A + (-massA*AccAO_omega)),JvecAO) + (-InertiaA[2,2]*alpA_omega[2]*JvecA[2])
    bvec = bvec + dot((ForceG_B + (-massB*AccBO_omega)),JvecBO) + (-InertiaB[2,2]*alpB_omega[2]*JvecB[2])
    bvec = bvec + dot((ForceG_C + (-massC*AccCO_omega)),JvecCO) + (-InertiaC[2,2]*alpC_omega[2]*JvecC[2])
    bvec = bvec + (TorqueA*JvecA[2])
    
    Amat = 0
    Amat = Amat + dot((massA*JvecAO),JvecAO) + ((InertiaA[2,2]*JvecA[2])*JvecA[2])
    Amat = Amat + dot((massB*JvecBO),JvecBO) + ((InertiaB[2,2]*JvecB[2])*JvecB[2])
    Amat = Amat + dot((massC*JvecCO),JvecCO) + ((InertiaC[2,2]*JvecC[2])*JvecC[2])

    d2thetaA = bvec/Amat
    ### END: Kane's Equation:

    dy = array([0.0,0.0])
    dy[0] = dthetaA
    dy[1] = d2thetaA
    return dy


# Solve the ODE
initvalues = array([thetaA_init, dthetaA_init])
file = open("where.txt", "w")
file.write("atstart")
file.close()

sol = solve_ivp(fourbar_ode,array([time0, timemax]),initvalues,t_eval=times)
print(sol.t)
print(sol.y)
print(type(sol))
print(sol.message)

# Compute all data
time = sol.t
thetaA = sol.y[0]
dthetaA = sol.y[1]

datalen = len(time)
d2thetaA = zeros(datalen)

PosABsave = zeros((3,datalen))
PosBCsave = zeros((3,datalen))
PosCNsave = zeros((3,datalen))

for ii in range(len(time)):

    if(ii==0):
        file = open("where.txt", "w")
        file.write("atstart")
        file.close()

    yvec = array([thetaA[ii],dthetaA[ii]])
    dyvec = fourbar_ode(time[ii],yvec)
    d2thetaA[ii] = dyvec[1]
  
    theta = thetaA[ii]
    Pos = position_analysis(theta)
    PosABsave[:,ii] = Pos[4]
    PosBCsave[:,ii] = Pos[7]
    PosCNsave[:,ii] = Pos[10]  
    
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

## Animate 
fig, ax = plt.subplots()
xdata, ydata = [], []
ln1, = ax.plot([], [], 'k')
ln2, = ax.plot([], [], 'ko',mfc='w')

def init():
    ax.set_xlim(-5, 20)
    ax.set_ylim(-10, 15)
    return ln1, ln2

def update(frame):
    PosAN = array([0,0,0])
    PosAB = PosABsave[:,frame]
    PosBC = PosBCsave[:,frame]
    PosCN = PosCNsave[:,frame]
    plotx = array([PosAN[0],PosAB[0],PosBC[0],PosCN[0]])
    ploty = array([PosAN[1],PosAB[1],PosBC[1],PosCN[1]])
    ln1.set_data(plotx, ploty)
    ln2.set_data(plotx, ploty)
    return ln1, ln2

ani = FuncAnimation(fig, update, frames=range(len(time)),
                    init_func=init, blit=True, repeat=False)
plt.grid()
plt.show()
#

