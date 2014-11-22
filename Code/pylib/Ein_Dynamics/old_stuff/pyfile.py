#!/usr/bin/env python

import numpy as np, Kinematic_Characteristics as kc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gravity = 9.81

################## calculate instantaneous rotational characteristic #########################
def atorquei(alpha, length_to_cg, mass, theta):
	return (alpha-gravity*np.sin(theta)*length_to_cg)*mass/length_to_cg


def thetatt(torque, length_to_cg, mass, theta):
	return torque*length_to_cg/mass+gravity*np.sin(theta)


def thetat(theta_tt, current_time, previous_time, theta_t):
	return theta_tt*(current_time-previous_time) + theta_t


def theta(theta_tt, theta_t, theta_i, current_time, previous_time):
	return (
		theta_tt*(current_time-previous_time)**2/2 +
		theta_t *(current_time - previous_time) +
		theta_i
		)
#############################################################################################

def r_cg_leg( length_of_leg, r_cp, m_thigh=1, m_slider=1):
	cg_thigh = length_of_leg/4
	cg_slider = (r_cp+length_of_leg/4)
	return ( cg_thigh*m_thigh + cg_sliger*m_slider)/( m_slider + m_thigh)


def r_cg_sli( length_of_leg, r_cp):
	return r_cp + length_of_leg/4

############# calculate the instantaneous radial characteristics ############################
def r_cp( r_cg, length_of_slider):
	return r_cg - length_of_slider/2


def mech_ad( length_of_driver, length_of_slider, r_cg):
	rcp = r_cp( r_cg, length_of_slider)
	driver_angle = np.arcsin( rcp/(2*length_of_driver))
	return np.cos( np.pi/2 - driver_angle)*np.cos( driver_angle)

def dtorquei( driver_angle, theta_t, r_cg, r_tt):
	return (
			(mass*(r_tt-theta_t**2/r_cg)-mass*gravity*np.cos(driver_angle))/
			(np.cos(np.pi-2*driver_angle)*np.cos(driver_angle))
			)

def rtt(torque, length_of_driver, mass, mech_adv, theta_t, r, theta_i):
	return (
		theta_t**2/r +
		gravity*np.cos(theta) +
		torque*length_of_driver*mech_adv/mass
		)

def rt(r_tt, current_time, previous_time):
	return r_tt*(current_time-previous_time)

def r(r_tt, r_t, r_i, current_time, previous_time):
	return (
		r_tt*(current_time-previou_time)**2/2 +
		r_t *(current_time - previous_time) +
		r_i
		)

def theta_driver( length_of_driver, r_cp):
	return np.arcsin(r_cp/(2*length_of_driver))
#############################################################################################

################### develope array of radial and rotational information #####################
def main(
		mass_leg, mass_slider,
		rot_rom, tra_rom,
		rot_accel, tra_accel,
		initial_time, final_time,
		length_of_thigh, length_of_slider,
		iterations
		):
	# set up initial condition
	length_of_leg = length_of_slider + length_of_thigh

# calulate dynamic characteristics with constant acceleration calculating motor 
# - torque
def update_rotational( alpha, mass, iterations, time_limit, initial_position,
		length_to_cg):
	# set up initial conditions
	print time_limit
	direction = 1
	theta_i, theta_t, theta_tt = initial_position, 0, 0
	step_size = time_limit/float(iterations)
	time = np.arange(0, (time_limit + step_size), step_size)
	position = np.array([])
	velocity = np.array([])
	motor_torque = np.array([])
	posi = np.array([])
	for i in range(len(time)):
		if theta_i > 0:
			direction = -1
		else:
			direction = 1
		current_time = time[i]
		previous_time = time[i-1]
		if i == 0:
			previous_time = current_time
		theta_i = theta(alpha*direction, theta_t*direction, theta_i, current_time, previous_time)
		theta_t = thetat(alpha*direction, current_time, previous_time, theta_t)
		torque  = atorquei(alpha*direction, length_to_cg, mass, theta_i)
		# append values to arrays
		position = np.hstack((position, np.array([theta_i])))
		velocity = np.hstack((velocity, np.array([theta_t])))
		motor_torque = np.hstack((motor_torque, np.array([torque])))
	return time, position, velocity, motor_torque

# calculate the radial dynamic chararteristics
def update_radial():
	time = np.arange(0, (time_limit + step_size), step_size)
	positition = np.array([])
	velocity = np.array([])
	motor_torque = np.array([])
	for i in range(len(time)):
		current_time = time[i]
		previous_time = time[i-1]
		if i == 0:
				previous_time = current_time
		r_i  = r(r_tt, r_t, r_i, current_time, previous_time)
		r_t  = rt(r_tt, current_time, previous_time)
		motor_torque = rtorque(acceleration, length_of_driver, mass, mech_adv, theta_t, r, theta_i)
		position = np.concatenate((position, np.array([r_i])))
		velocity = np.concatenate((velocity, np.array([r_t])))
		motor_torque = np.concatenate((motor_torque, np.array([torque])))

	return position, velocity, motor_torque

# get the calculate the necessary values
def get_requirement():
	iterations = 200
	length_of_leg = .5
	# time of slider is .2s
	time_of_slider = .2
	# time of leg is 2*l*sin(theta)/velocity
	time_of_swing = 2*length_of_leg*np.sin(15.0*np.pi/180)/.5
	# theta range is -15 to 15
	theta_swing_max_min = np.array([-15.0*np.pi/180.0, 15.0*np.pi/180.0])
	# length of the driver is .125 m
	length_of_driver = .1
	# mass of the leg 1.0292kg
	mass_of_leg = 1.0292
	# theta of the driver range is from 20 - 75 degrees
	theta_driver_max_min = np.array([20.0,75.0])
	# declare position vectors
	pvec1 = np.array([np.amin(theta_swing_max_min), 0, 0])
	pvec2 = np.array([np.amax(theta_swing_max_min), 0, 0])
	#determine necessary rotational acceleration
	alpha = kc.acceleration_necessary( 0, pvec1, pvec2, time_of_swing)[0]
	rotational_chars = update_rotational( alpha, mass_of_leg, iterations, 
			time_of_swing, theta_swing_max_min[0], .258  )

	print rotational_chars[3]

	#plot some fancies
	'''
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
	plt.legend(loc=2)
	plt.xlabel('Time')
	plt.ylabel('Kinematic Characteristic Values')
	plt.title('Position, Velocity and Torque as a Function of Time')
	plt.show()
	'''
	return 0


if __name__ == "__main__":
	get_requirement()
