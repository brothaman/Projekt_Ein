#!/usr/bin/env python



import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('Trajectory_Ein.pdf')

################################################################################
#################### Put dimensions and time of swing here #####################
l = .512
h = .2*l
lt = 2*l*np.sin(15*np.pi/180)/.5
t1 = -15*np.pi/180
t2 = -t1
# number of iterations
N = 100
time = lambda t0,tf,N: np.arange(t0,tf,(tf-t0)/(N+1))
################################################################################
################################################################################


################################################################################
########################### Begin Retraction ###################################
qi = np.array([ 0,h,h,h,h,0, 0,0,0,0,0,0 ])
ti = np.array([ 0,.2,.2,lt-.2,lt-.2,lt,   0,.2,.2,lt-.2,lt-.2,lt ])

q01 = lambda t: np.array([1,t,t**2,t**3,  0,0,0,0, 0,0,0,0])
qd01 = lambda t: np.array([0,1,2*t,3*t**2, 0,0,0,0, 0,0,0,0])

q12 = lambda t: np.array([0,0,0,0, 1,t,t**2,t**3, 0,0,0,0])
qd12 = lambda t: np.array([ 0,0,0,0, 0,1,2*t,3*t**2, 0,0,0,0])

q23 = lambda t: np.array([  0,0,0,0, 0,0,0,0, 1,t,t**2,t**3,])
qd23 = lambda t: np.array([ 0,0,0,0, 0,0,0,0, 0,1,2*t,3*t**2,])

# build the matrix
Q = np.array([
	q01(ti[0]),
	q01(ti[1]),
	q12(ti[2]),
	q12(ti[3]),
	q23(ti[4]),
	q23(ti[5]),
	qd01(ti[0]),
	qd01(ti[1]),
	qd12(ti[2]),
	qd12(ti[3]),
	qd23(ti[4]),
	qd23(ti[5]),
	])

a = np.linalg.solve(Q, qi.T)
[a10,a11,a12,a13,   a20,a21,a22,a23,    a30,a31,a32,a33] = a
# functions to calculate position, velocity and acceleration
qi 		= lambda a,t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
qdi		= lambda a,t: a[1] + 2*a[2]*t**2 + 3*a[3]*t**2
qddi	= lambda a,t: 2*a[2] + 6*a[3]*t

# set up time vectors
time1 = time(0.0, .2, N)
time2 = time( .2,lt-.2,N)
time3 = time(lt-.2,lt,N)

# pack a-coefficient arrays
a1 = a[0:4]
a2 = a[4:8]
a3 = a[8:12]

# compute and pack position arrays
q1 = qi(a1,time1[:])
q2 = qi(a2,time2[:])
q3 = qi(a3,time3[:])

# compute and pack velocity arrays
qd1 = qdi(a1,time1[:])
qd2 = qdi(a2,time2[:])
qd3 = qdi(a3,time3[:])

# compute and pack acceleration arrays
qdd1 = qddi(a1,time1[:])
qdd2 = qddi(a2,time2[:])
qdd3 = qddi(a3,time3[:])
############################## End Retraction ##################################
################################################################################


################################################################################
############################## Begin Swing #####################################
q01 = lambda t: np.array([1,t,t**2,t**3])
qd01 = lambda t: np.array([0,1,2*t,3*t**2])

qi = np.array([ t1,t2,  0,0])
Q = np.array([
	q01(ti[0]),
	q01(ti[-1]),
	qd01(ti[0]),
	qd01(ti[-1]),
	])

a = np.linalg.solve(Q, qi.T)
# functions to calculate position, velocity and acceleration
qi 		= lambda a,t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
qdi		= lambda a,t: a[1] + 2*a[2]*t**2 + 3*a[3]*t**2
qddi	= lambda a,t: 2*a[2] + 6*a[3]*t

time_swing = time(0.0, lt, N*3.0)
# compute and pack position arrays
q = qi(a, time_swing[:])
# compute and pack velcoty arrays
qd = qdi(a,time_swing[:])
# compute and pack aceeleration arrays
qdd = qddi(a,time_swing[:])
################################ End Swing #####################################
################################################################################


################################################################################
####################### Begin Torque Calculations ##############################
Iavg = 1.416e-2
dl = .1
mass = .47757
fout = mass*np.hstack((qdd1,qdd2,qdd3))
y = 2*dl - np.hstack((q1,q2,q3)) 
n = len(y)+1

alpha = qdd
theta = lambda y: np.pi-np.arcsin((2*dl-y)/(2*dl))
t = -np.pi + theta(y[:]) + 50*np.pi/180
#t = np.arange(20.0*np.pi/180, 20.0*np.pi/180 + (theta(max(y))-theta(min(y))) , (theta(max(y))-theta(min(y)))/n)
#t = np.arange(0,90)
#t = t[:]*np.pi/180
ma = [np.cos(np.pi/2.0-2*theta)*np.cos(theta) for theta in t]

Torque_driver = lambda fout, ma: fout/(ma)*dl
Torque_swingr = lambda a: Iavg*a

Td = Torque_driver(fout[:],ma[:])
Ts = Torque_swingr(alpha)
################################################################################
################################ Begin Plot ###################################
print max(Td),min(Td)
print max(qdd1),min(qdd1)
print max(qd1),min(qd1)
print max(Ts),min(Ts)
print max(qdd),min(qdd)
print max(qd),min(qd)

plt.plot(
		time1,q1,'r',
		time2,q2,'r',
		time3,q3,'r',
		)
plt.title("Ein's Foot Trajectory")
plt.xlabel("Time (seconds)")
plt.ylabel("y Position in (m)")
pp.savefig()
plt.show()

plt.plot(
		time1,qd1,'r',
		time2,qd2,'r',
		time3,qd3,'r',
		)
plt.title("Ein's Foot Trajectory")
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity in the y-direction")
pp.savefig()
plt.show()


plt.plot(
		time1,qdd1,'b',
		time2,qdd2,'b',
		time3,qdd3,'b',
		)
plt.title("Ein's Foot Trajectory")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration in the y-direction")
pp.savefig()
plt.show()


time = np.hstack((time1,time2,time3))
plt.plot(
		time, Td,'r',
		time_swing, Ts,'b',
		)
plt.title("Ein's Foot Trajectory")
plt.xlabel("Time (seconds)")
plt.ylabel("Torque of Slider Crank Motor in (N m)")
pp.savefig()
plt.show()
pp.close()
