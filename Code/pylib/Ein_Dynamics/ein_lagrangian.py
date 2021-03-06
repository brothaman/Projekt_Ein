#!/usr/bin/env python
import sys
import robotics_functions as rf, numpy as np, scipy as sp, sympy as sy

[l1, l2, l3, t1, t2, t3, a1, a2, a3, d1, d2, d3, I1, I2] = sy.symbols("l1 l2 l3 t1 t2 t3 a1 a2 a3 d1 d2 d3 I1 I2")
[q1, q2, qdot1, qdot2, qddot1, qddot2, m1, m2, r1, r2] = sy.symbols("q1 q2 qdot1 qdot2 qddot1 qddot2 m1 m2 r1 r2")
theta,thetadot,thetaddot = sy.symbols("theta,thetadot,thetaddot")
link_list_cm = [[
		[0, np.pi/2, 0, theta],
		[0, 0, q2, 0]
		],
		[sy.Matrix([[-l1],[0],[0],[1]]),
			sy.Matrix([[-q2+l1+l2],[0],[0],[1]])
			]
		]
gravity = sy.symbols("g")
g = sy.Matrix([
	[-gravity],
	[0],
	[0]
	])
m = np.array([m1, m2])


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
qmat = sy.Matrix([
	[theta, thetadot, thetaddot],
	[q2, qdot2, qddot2],
	])
q = sy.Matrix([
	[theta],
	[q2]
	])
qdot = sy.Matrix([
	[thetadot],
	[qdot2]
	])
qddot = sy.Matrix([
	[thetaddot],
	[qddot2]
	])
tdv_vec = [
		(thetadot,thetaddot),
		(qdot2,qddot2),
		(theta, thetadot),
		(q2, qdot2),
		]

#A0i = rf.sym_get_A0n(link_list_cm[0])
#for a in A0i:
#	sy.pprint(sy.simplify(sy.trigsimp(a)))

J = rf.sym_pt_jacobian(link_list_cm)
#for j in J:
#	sy.pprint(sy.simplify(sy.trigsimp(j)))
# get lagrangian
K,P = rf.sym_lagrangian(link_list_cm, g, M, I, qdot)
L = K+P
# get equations of motion
[eom1, eom2] = rf.equations_of_motion( L, qmat)

print "\nEquations of motion from THETA"
print sy.pprint(sy.simplify(sy.trigsimp(eom1)))
print "\nEquations of motion from TRANSLATING"
print sy.pprint(sy.simplify(sy.trigsimp(eom2)))
G1 = sy.simplify(sy.trigsimp(eom1.subs(zip([qdot2,thetadot,qddot2,thetaddot],[0,0,0,0]))))
G2 = sy.simplify(sy.trigsimp(eom2.subs(zip([qdot2,thetadot,qddot2,thetaddot],[0,0,0,0]))))

C1 = sy.simplify(sy.trigsimp(eom1.subs(zip([qddot2,thetaddot],[0,0]))-G1))
C2 = sy.simplify(sy.trigsimp(eom2.subs(zip([qddot2,thetaddot],[0,0]))-G2))

M1 = sy.simplify(sy.trigsimp(eom1-C1-G1))
M2 = sy.simplify(sy.trigsimp((eom2-C2-G2)))


M11 = sy.simplify(sy.trigsimp(M1.subs(zip([thetaddot,qddot2],[1,0]))))
M12 = sy.simplify(sy.trigsimp(M1.subs(zip([thetaddot,qddot2],[0,1]))))

M21 = sy.simplify(sy.trigsimp(M2.subs(zip([thetaddot,qddot2],[1,0]))))
M22 = sy.simplify(sy.trigsimp(M2.subs(zip([thetaddot,qddot2],[0,1]))))

G = sy.Matrix([
	[G1],
	[G2],
	])
C = sy.Matrix([
	[C1],
	[C2],
	])
M = sy.Matrix([
	[M11,M12],
	[M21,M22],
	])


print("G1 = " + str(G1))
print("G2 = " + str(G2))
print("C1 = " + str(C1))
print("C2 = " + str(C2))
print("M11 = " + str(M11))
print("M12 = " + str(M12))
print("M21 = " + str(M21))
print("M22 = " + str(M22))
print("K = " + str(K))
print("P = " + str(P))

dots = sy.simplify(sy.trigsimp(M.LUsolve(-G-C)))
print "x2 = "+ str(sy.simplify(dots[0]))
print "y2 = "+ str(sy.simplify(dots[1]))

sy.pprint(dots)
'''
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
	print "	rotational: "
	print (sy.simplify(sy.trigsimp(theta_ss[1])))
	print "	radial: "
	print (sy.simplify(sy.trigsimp(rad_ss[1])))

	'''
