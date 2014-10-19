#!/usr/bin/env python

import numpy as np

# function to calculate instantaneous rotational characteristic
def torquei(alpha, length_to_cg, mass, theta):
	return alpha*mass/length_to_cg-gravity*np.sin(theta)
def thetatt(torque, length_to_cg, mass, theta):
	return torque*length_to_cg/mass+gravity*np.sin(theta)
def thetat(theta_tt, current_time, previous_time):
	return theta_tt*(current_time-previous_time)
def theta(theta_tt, theta_t, theta_i, current_time, previous_time):
	return (
		theta_tt*(current_time**2-previous_time**2)/2 +
		theta_t *(current_time   -previous_time) +
		theta_i
		)

# calculate the instantaneous radial characteristics
def rtt(torque, length_of_driver, mass, mech_adv, theta_t, r, theta_i):
	return (theta_t**2/r +
		gravity*np.cos(theta) +
		torque*length_of_driver*mech_adv/mass
		)
def rt(r_tt, current_time, previous_time):
	return r_tt*(current_time-previous_time)
def r(r_tt, r_t, r_i, current_time, previous_time):
	return (
		r_tt*(current_time**2-previou_time**2)/2 +
		r_t *(current_time - previous_time) +
		r_i
		)

# calulate dynamic characteristics with constant velocity calculating motor 
# - torque
def update_rotational( alpha, mass, step_size, time_limit, initial_position,):
	time = np.arange(0, (time_limit + step_size), step_size)
	position = np.array([])
	velocity = np.array([])
	motor_torque = np.array([])
	for i in range(len(time)):
		current_time = time[i]
		previous_time = time[i-1]
		if i == 0:
			previous_time = current_time
		theta_i = theta(alpha, theta_t, theta_i, current_time, previous_time)
		theta_t = thetat(alpha, current_time, previous_time)
		torque  = torguei(alpha, length_to_cg, mass, theta)
		# append values to arrays
		position = np.concatenate((position, np.array([theta_i])))
		velocity = np.concatenate((velocity, np.array([theta_t])))
		motor_torque = np.concatenate((motor_torque, np.array([torque])))
	return position, velocity, motor_torque
def update_radial():
	return bla
def get_requirement():
	return requrements

