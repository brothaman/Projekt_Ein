#!/usr/bin/env python

import sys
sys.path.append(r"/Users/robertbrothers/Desktop/Fall 2014/Fundamentals_of_Robotics/robo_git/python/")

import robotics_functions as rf, numpy as np, scipy as sp, sympy as sy
from numpy import cos,sin
from scipy import integrate

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

l1,l2,m1,m2,I1,I2 = [.5,.5,2.0,2.0,.01,.01]
g = 9.81

thetaddot = lambda x1,x2,y1,y2: (4.0*g*l1*m1*cos(x1) - 8.0*g*l2*m2*sin(x1) + 4.0*g*m2*y1*sin(x1) + 4.0*l2*m2*x2*y2 - 2.0*m2*x2*y1*y2)/(4.0*I1 + 4.0*I2 + l1**2*m1 + 4.0*l2**2*m2 - 4.0*l2*m2*y1 + m2*y1**2) 

lddot = lambda x1,x2,y1,y2: -g*cos(x1) + l2*x2**2 - 0.5*x2**2*y1

init_con = [0, 0, 0, 0]

def update( x, ti):
	xx1 = x[1]
	yy1 = x[3]
	xx2 = thetaddot(x[0],x[1],x[2],x[3])
	yy2 = lddot(x[0],x[1],x[2],x[3])
	return xx1,xx2,yy1,yy2


t = np.arange(0,5.0,.01)
x0 = sp.integrate.odeint( update, init_con,t)
for val in x0:
	print val


onelinkthetaddot = lambda x1, x2: -g/l1*sin(x1)

def olupdate(x,ti):
	xx1 = x[1]
	xx2 = onelinkthetaddot(x[0],x[1])
	return xx1,xx2

#x0 = sp.integrate.odeint( olupdate, init_con[:2],t)
for val in x0:
	print val
dt = .01

x1 = l1*sin(x0[:,0])
y1 = -l1*cos(x0[:,0])

x2 = -x0[:,2]*sin(x0[:,0]) + x1
y2 = x0[:,2]*cos(x0[:,0]) + y1

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
