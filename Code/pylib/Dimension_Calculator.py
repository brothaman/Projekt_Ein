#!/usr/bin/env python

import sys, scipy as sp

def tele_velocity_min(travel_angle, travel_velocity):
	'''
	This function will compute the minimum speed at which the leg needs to be 
	raised to provide clearance 
	'''
	travel_angle_theta = sp.pi/180.0*travel_angle
	return (travel_velocity*(1.0-sp.cos(travel_angle_theta))
			/(2*sp.sin(travel_angle_theta)))

def clearance_period(leg_length, travel_angle, travel_velocity):
	'''
	This function will compute the inverse of the amount of time necessary for
	leg to make clearance 
	'''
	travel_angle_theta = sp.pi/180.0*travel_angle
	return leg_length*(1-sp.cos(travel_angle_theta))/travel_velocity

def travel_period(leg_length, travel_angle, travel_velocity):
	'''
	This function will:
	Compute the inverse of the time necessary for the leg to travel between 
	contact points
	'''
	travel_angle_theta = sp.pi/180.0*travel_angle
	return 2*leg_length*(1-sp.cos(travel_angle_theta))/travel_velocity

def travel_length( leg_length, travel_angle):
	return (2*leg_length*sp.sin( sp.pi/180.0*travel_angle) )

def clearance_length( leg_length, travel_angle):
	return (leg_length*(1 - sp.cos(sp.pi/180.0*travel_angle)))


