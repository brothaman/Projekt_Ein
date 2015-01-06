#!/usr/bin/env python

import sys
sys.path.append(r"/Users/robertbrothers/Desktop/Fall 2014/Fundamentals_of_Robotics/robo_git/python/")

import robotics_functions as rf, numpy as np, scipy as sp, sympy as sy
from numpy import cos,sin,sqrt, pi
from scipy import integrate
from scipy.sparse import csc_matrix, linalg as sla

import warnings
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

l0,l1,l2,m1,m2,I1,I2 = [0,0,.5,2.0,2.0,.01,.01]
T1,T2 = 0,0
g = 9.81



init_con = [0, 0, 0, 0]
E = []
def update( ti, x):
	theta,thetadot,q2,qdot2 = x
	G1 = 1.0*g*(l1*m1*sin(theta) + m2*(q2*cos(theta) + (-l1 - l2 + q2)*sin(theta)))
	G2 = -1.0*sqrt(2)*g*m2*cos(theta + pi/4)
	C1 = 0
	C2 = 1.0*m2*thetadot*(l1*thetadot + l2*thetadot - q2*thetadot - qdot2)
	M11 = 1.0*I1 + 1.0*I2 + 1.0*l1**2*m1 + 1.0*l1**2*m2 + 2.0*l1*l2*m2 - 2.0*l1*m2*q2 + 1.0*l2**2*m2 - 2.0*l2*m2*q2 + 1.0*m2*q2**2
	M12 = 0
	M21 = 0
	M22 = 1.0*m2
	G = np.array([
		[G1],
		[G2],
		])
	C = np.array([
		[C1],
		[C2],
		])
	M = csc_matrix([
		[M11,M12],
		[M21,M22],
		])
	lu = sla.splu(M)
	x2,y2 = lu.solve()
#	K = 0.5*qdot2*(qdot2*(1.0*m2*sin(theta)**2 + 1.0*m2*cos(theta)**2 + 3.74939945665464e-33*m2) + thetadot*(-1.0*m2*(l1 + l2 - q2)*sin(theta)**2 - 1.0*m2*(l1 + l2 - q2)*cos(theta)**2)) + 0.5*thetadot*(qdot2*(-1.0*m2*(l1 + l2 - q2)*sin(theta)**2 - 1.0*m2*(l1 + l2 - q2)*cos(theta)**2) + thetadot*(I1 + I2 + 1.0*l1**2*m1*sin(theta)**2 + 1.0*l1**2*m1*cos(theta)**2 + 1.0*m2*(l1 + l2 - q2)**2*sin(theta)**2 + 1.0*m2*(l1 + l2 - q2)**2*cos(theta)**2))
#	P = 1.0*g*l1*m1*cos(theta) - g*m2*(1.0*q2*sin(theta) + 1.0*(l1 + l2 - q2)*cos(theta))
#	E.append(K-P)
	x2 = -1.0*m2*(-1.0*g*l2 + 2.0*g*q2 + 1.0*qdot2**2)/I2
	y2 = (1.0*I2*g + 1.0*g*l2**2*m2 - 3.0*g*l2*m2*q2 + 2.0*g*m2*q2**2 - 1.0*l2*m2*qdot2**2 + 1.0*m2*q2*qdot2**2)/I2
	x1,y1 = thetadot,qdot2
	return [x1,x2,y1,y2]

t0, tf, dt = [0, 10.0, .1]
wow = True
if wow:
	solver = sp.integrate.ode( update).set_integrator('dopri5', nsteps=500000, atol=.1, rtol=.1)
	solver.set_initial_value(init_con, t0)

	x0 = []
	t = []
	warnings.filterwarnings("ignore", category=UserWarning)
	while solver.successful() and solver.t < tf:
		solver.integrate(solver.t+dt)
		x0.append(solver.y)
		t.append(solver.t)
	x0 = np.array(x0)
else:
	t = np.arange(t0,tf,dt)
	x0 = sp.integrate.odeint( update, init_con,t)

print x0
#************* plot energy of ein ****************
plt.plot(E)
plt.savefig("/Users/robertbrothers/Desktop/Energy_of_EIN_vs_Time.png", format='png')
plt.show()
#*************************************************

x1 = l1*sin(x0[:,0])
y1 = -l1*cos(x0[:,0])

x2 = x0[:,2]*cos(x0[:,0]) + x1
y2 = -x0[:,2]*sin(x0[:,0]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.01, 0.9, '', transform=ax.transAxes)


def init():
	line.set_data([], [])
	time_text.set_text('')
	return line, time_text

def animate(i):
	thisx = [0, x1[i], x2[i]]
	thisy = [0, y1[i], y2[i]]

	line.set_data(thisx, thisy)
	time_text.set_text(time_template%(i*dt))
	return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x0)),
		    interval=25, blit=False, init_func=init)

#ani.save('ein_animation.mp4', fps=15)
plt.show()
