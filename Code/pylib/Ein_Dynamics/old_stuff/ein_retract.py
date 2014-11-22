#!/usr/bin/env python

import numpy as np, Kinematic_Characteristics as kc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ein_swinging

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
		r_tt*(current_time-previou_time)**2/2 +
		r_t *(current_time - previous_time) +
		r_i
		)
def theta_driver( length_of_driver, r_cg):
	return np.arcsin(r_cg/(2*length_of_driver))

# calculate the radial dynamic chararteristics
def update_radial( acceleration, time, iterations, rom, ):
	[r_tt, r_t, r_i] = [acceleration, 0, max(rom)]
	time = np.arange(0, (time + time/iterations), time/iterations)
	positition = np.array([])
	velocity = np.array([])
	driver_angle = np.array([])
	motor_torque = np.array([])
	for i in range(len(time)):
		current_time = time[i]
		previous_time = time[i-1]
		if i == 0:
			previous_time = current_time
		dtheta = theta_driver
		r_i  = r(r_tt, r_t, r_i, current_time, previous_time)
		r_t  = rt(r_tt, current_time, previous_time)
		motor_torque = rtorque(r_tt, length_of_driver, mass, mech_adv, theta_t, r, theta_i)
		r_tt = rtt(motor_torque, length_of_driver, mass, theta_t, r_i, theta_i)
		position = np.hstack((position, np.array([r_i])))
		velocity = np.hstack((velocity, np.array([r_t])))
		motor_torque = np.hstack((motor_torque, np.array([torque])))

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
