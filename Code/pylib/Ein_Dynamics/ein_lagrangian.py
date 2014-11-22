#!/usr/bin/env python
import sys
sys.path.append(r"/Users/robertbrothers/Desktop/Fall 2014/Fundamentals_of_Robotics/robo_git/python/")
import robotics_functions as rf, numpy as np, scipy as sp, sympy as sy
from scipy import integrate

[l1, l2, l3, t1, t2, t3, a1, a2, a3, d1, d2, d3, I1, I2] = sy.symbols("l1 l2 l3 t1 t2 t3 a1 a2 a3 d1 d2 d3 I1 I2")
[q1, q2, qdot1, qdot2, qddot1, qddot2, m1, m2, r1, r2] = sy.symbols("q1 q2 qdot1 qdot2 qddot1 qddot2 m1 m2 r1 r2")
theta,thetadot,thetaddot = sy.symbols("theta,thetadot,thetaddot")
link_list_cm = [[
		[0, np.pi/2, 0, theta],
		[0, 0, l2, 0]
		],
		[ sy.Matrix([[-l1/2],[0],[0],[1]]),
			sy.Matrix([[0],[0],[-q2/2],[1]])
			]
		]

m = np.array([m1, m2])
l = np.array([l1, l2])


M = [sy.Matrix([
	[m[i],0,0],
	[0,m[i],0],
	[0,0,m[i]]
	]) for i in range(len(m))]
I = [sy.Matrix([
	[I1,0,0],
	[0,I1, 0],
	[0, 0,I1]
	]), 
	sy.Matrix([
		[I2,0,0],
		[0,I2,0],
		[0,0,I2]
		])
	]
q = sy.Matrix([
	[theta],
	[q2]
	])
qdot = sy.Matrix([
	[thetadot],
	[qdot2]
	])
tdv_vec = [
		(thetadot,thetaddot),
		(qdot2,qddot2),
		(theta, thetadot),
		(q2, qdot2),
		]


# get jacobian
J = rf.sym_pt_jacobian(link_list_cm)
# get lagrangian
L = rf.sym_lagrangian(link_list_cm, M, I, qdot)[0]
# get equations of motion
[rotational, radial] = rf.equations_of_motion( L, q, qdot, tdv_vec)
# get state space equations
thetaddot = sy.solve( rotational, thetaddot)[0]
rddot = sy.solve( radial, qddot2)[0]

# state space stuff
[x1, x2, xdot1, xdot2, y1, y2, ydot1, ydot2] = sy.symbols("x1, x2, xdot1, xdot2, y1, y2, ydot1, ydot2")
sublist = [
		(theta, x1),
		(thetadot, x2),
		(thetaddot, xdot2),
		(q2, y1),
		(qdot2, y2),
		(qddot2, ydot2)
		]

def cleanup( func):
	for (i,j) in sublist:
		func = func.subs(i,j)
	return func

thetaddot = cleanup(thetaddot)
rddot = cleanup(rddot)

theta_ss = sy.Matrix([
	[x2],
	[thetaddot]
	])
rad_ss = sy.Matrix([
	[y2],
	[rddot],
	])


'''
############################## something tricky ################################
[r1,m1,m2,l1,l2,g] = [.074, 1.03/2, 1.03/2, .512/2, .512/2,9.81]
f1 = lambda (x1,x2,y1,y2): ((-15000000000000.0*g*l1*m1*np.cos(x1) + 15000000000000.0*g*m2*y1*np.sin(x1) - 7500000000000.0*m2*x2*y1*y2)/
(3750000000000.0*l1**2*m1 + 5000000000000.0*l2**2*m2 + 15707963267949.0*m1*r1**2 + 3750000000000.0*m2*y1**2))
f2 = lambda (x1,x2,y1,y2): -g*np.cos(x1) - 0.5*x2**2*y1
# initial conditions
st = [-15*np.pi/180, 0 , l1, 0]
def nexus(ti, st0, st1, st2, st3):
	st = [st0, st1, st2, st3]
	rst = st
	[rst[0],rst[2]] = [st1,st3]
	[rst[1],rst[3]] = [f1(st),f2(st)]
	return rst

time = np.arange(0,1.0, 1.0/1e3)
y0,t0 = st, 0]
stuff = sp.integrate.ode(nexus).set_integrator('zvode', method='bdf', with_jacobian=False)
r.set_initial_value(st, t0)

t1 = 6
dt = .001
while r.successful() and r.t < t1:
	r.integrate(r.t+dt)
	print r.t, r.y

'''
if __name__ == "__main__":
	'''
	print "mass matrix"
	print sy.printing.latex(M)
	print "Inertia matrix"
	print sy.printing.latex(I)
	print "jacobian"
	for j in J:
		print sy.simplify(sy.trigsimp(j))
	print "lagrangian"
	print sy.simplify(sy.trigsimp(L))
	print "equations of motion"
	print " rotational: "
	print sy.simplify(sy.trigsimp(rotational))
	print " radial: "
	print sy.simplify(sy.trigsimp(radial))
	print " TOTAL: "
	print sy.simplify(sy.trigsimp(rotational + radial))
	print "State Space: "
	'''
	print "	rotational: "
	print (sy.simplify(sy.trigsimp(theta_ss[1])))
	print "	radial: "
	print (sy.simplify(sy.trigsimp(rad_ss[1])))

