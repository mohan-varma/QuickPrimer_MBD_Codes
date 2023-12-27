from numpy.linalg import norm, solve
from numpy import add, array, divide
from numpy import cross, zeros
from numpy import subtract
from math import pi, cos, sin
from math import acos, sqrt

def get_TransMat(theta):
    TransMat = array([[cos(theta), -sin(theta), 0],[sin(theta), cos(theta), 0],[0, 0, 1]])
    return TransMat

def make_tilde(Rvec):
    tildeRvec = zeros((3,3))
    tildeRvec[0,1] = -Rvec[2]
    tildeRvec[0,2] =  Rvec[1]
    tildeRvec[1,0] =  Rvec[2]
    tildeRvec[1,2] = -Rvec[0]
    tildeRvec[2,0] = -Rvec[1]
    tildeRvec[2,1] =  Rvec[0]
    return tildeRvec

def find_ang(v):
# Author: D. S. Mohan Varma
# subroutine to find angle made by the given vector with positive x-axis
# Input Column vector, # Output angle in radians
	v[2] = 0
	absval = norm(v,None)
	v = divide(v,absval)
	
	if(v[0]>=0):
		if(v[1]>=0):
			b = acos(v[0])
		 
		else:
			b = 2*pi - acos(v[0])

	else:
		if(v[1]>=0):
			b = pi - acos(abs(v[0]))
		 
		else:
			b = pi + acos(abs(v[0]))
	return b
def circlecircle(x1, y1, r1, x2, y2, r2):
# Author: D. S. Mohan Varma
# subroutine to find intersection of two circles
# Input: centre coordinates and radii # Output: Points of intersection
	flag = 0
	c1 = array([x1, y1, 0])
	c2 = array([x2, y2, 0])
	uvec = subtract(c2,c1)
	#print(uvec)
	d = norm(uvec,None)
	
	if(d==0):
		flag = 1
		if(r1==r2):
			print('Concentric circles of same radius')
		else:
			print('Concentric circles')
		xc = []
		yc = []
			
	elif(d!=0):
		u1 = divide(uvec,d)
		kcap = array([0,0,1])
		u2 = cross(kcap,u1)
		if(d>(r1+r2)):
			flag = 1
			print('Circles do not intersect d>(r1+r2)')
			xc = []
			yc = []

		elif(d==(r1+r2)):
			I1 = c1 + r1*u1
			xc = [I1[0], I1[0]]
			yc = [I1[1], I1[1]]
		
		elif(d<(r1+r2)):
			if(d<abs(r1-r2)):
				print('Circles do not intersect d<abs(r1-r2)')
				#print([d, r1, r2, abs(r1-r2)])
				xc = []
				yc = []
			elif(d==abs(r1-r2)):
				if(r1>r2):
					I1 = c1 + r1*u1
				elif(r2>r1):
					I1 = c2 - r2*u1
				xc = [I1[0], I1[0]]
				yc = [I1[1], I1[1]]
			elif(d>abs(r1-r2)):
				if(r1<=d and r2>d):
					x = (-r1**2 + r2**2 - d**2)/(2*d)
					h = sqrt(r1**2 - x**2)
					I1 = c1 - x*u1 - h*u2
					I2 = c1 - x*u1 + h*u2
				elif(r1>d and r2<=d):
					x = (r1**2 - r2**2 - d**2)/(2*d)
					h = sqrt(r1**2 - (d+x)**2)
					I1 = c1 + (d+x)*u1 - h*u2
					I2 = c1 + (d+x)*u1 + h*u2
				elif(r1<=d and r2<=d):
					x = (r1**2 - r2**2 + d**2)/(2*d)
					h = sqrt(r1**2 - x**2)
					I1 = c1 + x*u1 - h*u2
					I2 = c1 + x*u1 + h*u2
				elif(r1>d and r2>d):
					x = (r1**2 - r2**2 + d**2)/(2*d)	
					h = sqrt(r1**2 - x**2)
					I1 = c1 + x*u1 - h*u2
					I2 = c1 + x*u1 + h*u2

				xc = [I1[0], I2[0]]
				yc = [I1[1], I2[1]]	
	return [xc,yc,flag]

def circleline(PosLx, PosLy, dirL, PosCx, PosCy, radC):

	Pline = array([PosLx,PosLy,0])
	Ccirc = array([PosCx,PosCy,0])
	kcap = array([0,0,1])
	hcap = cross(dirL,kcap)
	bvec = zeros((2,1))
	Amat = zeros((2,2))
	Amat[0:2,0] = dirL[0:2]
	Amat[0:2,1] = hcap[0:2]
	bvec = Ccirc[0:2] - Pline[0:2]
	sol = solve(Amat,bvec)
	mvar = sol[0]
	hvar = sol[1]
	
	if(hvar>radC):
		print('The line and the circle do not intersect')
		xc = []
		yc = []
	
	elif(hvar==radC):
		xc = Pline[0]+ mvar*dirL[0]
		yc = Pline[1]+ mvar*dirL[1]
	
	elif(hvar<radC):
		avar = sqrt(radC**2 - hvar**2)
		PInter = Pline + mvar*dirL
		P1 = PInter - avar*dirL
		P2 = PInter + avar*dirL
		xc = [P1[0],P2[0]]
		yc = [P1[1],P2[1]]
		
	return xc, yc

	

	



