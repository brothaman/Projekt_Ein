#!/usr/bin/env python

import numpy as np, Kinematic_Characteristics as kc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from textwrap import wrap

gravity = 9.81
# calculate the instantaneous radial characteristics
def rtorque( theta, theta_t, r_cg, mass, r_tt):
	return (
			(mass*(r_tt-theta_t**2/r_cg)-mass*gravity*np.cos(theta))/
			(np.cos(np.pi-2*theta)*np.cos(theta))
			)
def rtt( torque, length_of_driver, mass, theta_t, r, theta_i):
	return (
		theta_t**2/r +
		gravity*np.cos( theta) +
		torque/length_of_driver/mass*(np.cos(np.pi-2*theta)*np.cos(theta))
		)
def rt(r_tt, current_time, previous_time):
	return r_tt*(current_time-previous_time)
def r(r_tt, r_t, r_i, current_time, previous_time):
	return (
		r_tt*(current_time-previous_time)**2/2 +
		r_t *(current_time - previous_time) +
		r_i
		)
def theta_driver( length_of_driver, r_cg):
	return np.arcsin(r_cg/(2*length_of_driver))


# function to calculate instantaneous rotational characteristic
def atorquei(alpha, length_to_cg, mass, theta):
	return mass*length_to_cg*(alpha-gravity*np.sin(theta))

# Function to calculate the Instantaneous rotational acceleration
def thetatt(torque, length_to_cg, mass, theta):
	return torque/(length_to_cg*mass)+gravity*np.sin(theta)

# function to calculate the instantaneous rotational velocity
def thetat(theta_tt, current_time, previous_time, theta_t):
	return theta_tt*(current_time-previous_time) + theta_t

# function to calculate the instantaneous rotational posisiton
def theta(theta_tt, theta_t, theta_i, current_time, previous_time):
	return (
		theta_tt*(current_time-previous_time)**2/2 +
		theta_t *(current_time - previous_time) +
		theta_i
		)


# calulate dynamic characteristics with constant acceleration calculating motor 
# - torque
def update_rotational( 
		alpha, mass, 
		iterations, initial_time, 
		final_time, initial_position,
		length_to_cg
		):


	# set up initial conditions
	time_limit = final_time-initial_time
	direction = 1
	theta_i, theta_t, theta_tt, torque = initial_position, 0, -alpha, 0
	step_size = time_limit/float(iterations)
	time = np.arange(0, (time_limit + step_size), step_size)
	position = np.array([])
	velocity = np.array([])
	motor_torque = np.array([])
	posi = np.array([])
	switch = False
	[rdd, r_t, r_i] = [.512*.5/.04, 0, 0]
	r_tt = rdd
	lshin = .512/2
	mass = 1.03/2


	# cycle through for each time increment
	for i in range(len(time)):
		current_time = time[i]
		previous_time = time[i-1]
		if i == 0:
			previous_time = current_time
		theta_i = theta(theta_tt, theta_t, theta_i, current_time, previous_time)
		theta_t = thetat(theta_tt, current_time, previous_time, theta_t)
		torque  = atorquei(theta_tt, length_to_cg, mass, theta_i)
		dtor = rtorque( theta_i, theta_t, r_i + lshin/2,  mass, r_tt)
		r_t = rt(r_tt, current_time, previous_time) + r_t
		print r_tt
		r_i = r(r_tt, r_t, r_i, current_time, previous_time)
		j = i-1
		if i > 0:
			if (
					((position[j] < 0 and position[j-1] > 0) or 
						(position[j] > 0 and position[j-1] < 0)) and
					switch == False):
				switch = True
				theta_tt = -1*thetatt(torque, length_to_cg, mass, theta_i)
			else:
				switch = False
				theta_tt = thetatt(torque, length_to_cg, mass, theta_i)
		else:
			theta_tt = thetatt(torque, length_to_cg, mass, theta_i)
		# for time til 200ms calculate accelration is non zero
		if time[i] < .2:
			r_tt = rdd
		elif time[i] > .2 and time[i] < final_time-.2:
			r_tt = 0
			r_t = 0
		elif time[i] > final_time-.2:
			r_tt = -rdd

#		print theta_tt
		# append values to arrays
		position = np.hstack((position, np.array([theta_i])))
		velocity = np.hstack((velocity, np.array([theta_t])))
		motor_torque = np.hstack((motor_torque, np.array([torque])))
	return time, position, velocity, motor_torque



# get the calculate the necessary values
def get_requirement():
	# calc values
	length_of_leg = .5
	dx = 2*length_of_leg*np.sin(15.0*np.pi/180)
	vx = .5
	time_of_swing = dx/vx
	dt0 = (15.0-0.0)*np.pi/180.0
	dt1 = (0.0-15.0)*np.pi/180.0
	alpha0 = 2*(dt0-dt1)/( time_of_swing)**2
	alpha1 = 2*dt1/( time_of_swing/2)**2
	iterations = 1000
	# mass of the leg 1.0292kg
	mass_of_leg = 1.0292
	# declare position vectors
	#determine necessary rotational acceleration
	rotational_chars = update_rotational( alpha0, mass_of_leg, iterations, 
			0.0, 2*time_of_swing, dt0, .258  )
	print alpha0
	print max(rotational_chars[3]), min(rotational_chars[3])

	print "alpha is: " + str(alpha0)
	print "time of swing: " +str(time_of_swing)
	print("range of rotation is from: "+ str(dt0)+ 
			" to: " + str(dt1))
	print "the mass of the leg is: " + str(mass_of_leg)
	print ("the max and min positions: " + str(max(rotational_chars[1])) + ", " + 
			str(min(rotational_chars[1])) )
	print ("the max and min velocity: " + str(max(rotational_chars[2])) + ", " + 
			str(min(rotational_chars[2])) )
	print ("the max and min acceleration: " + str(max(rotational_chars[3])) + ", " + 
			str(min(rotational_chars[3])) )

	#plot some fancies
	posi = plt.plot(
			rotational_chars[0], rotational_chars[1], color='r', 
			linewidth=3, label=r'Position of cg'# position
			)
	velo = plt.plot(
			rotational_chars[0], rotational_chars[2], color='b',
			linewidth=3, label=r'Velocity of cg'# velocity
			)
	acce = plt.plot(
			rotational_chars[0], rotational_chars[3], color='m',
			label=r'Torque for Constant $\alpha$'# acceleration
			)
	plt.subplots_adjust(top=0.875)
	ax = plt.gca()
	plt.legend(loc=4)
	plt.xlabel('Time (s)')
	ax.set_xlim([0,1])
	ylab = r'Kinematic Characteristic Values'+'\n'+r'Torque in $(n*m)$, Velocity in $(\frac{rad}{s})$, Position in $(rad)$'
	plt.ylabel(ylab)
	ax.set_ylim([-1.8, 1.8])
	plt.show()
	return 0

if __name__ == "__main__":
	get_requirement()
