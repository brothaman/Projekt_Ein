#!/usr/bin/env python

import numpy as np

def cross_product( vec1, vec2):
	return [ (vec1[1]*vec2[2] - vec2[1]*vec1[2]),
					-(vec1[0]*vec2[2] - vec2[0]*vec1[2]),
					 (vec1[0]*vec2[1] - vec2[0]*vec1[1])
					 ]

def dot_product( vec1, vec2):
	return [ vec1[i]*vec2[i] for i in range(len(vec1)) ]

def magnitude(vec):
	val = 0
	for i in vec:
		val = val + i**2
	return val**(1.0/2.0)

def unit( vec):
	mag = magnitude(vec)
	return [val/mag for val in vec]

def moment( weight, radius_to_cg):
	'''
	this function accepts 2 vectors of weight and radius of cg to determine 
	moment about the cg
	[ Wx, Wy, Wz] x [ rx, ry, rz]
	'''
	return cross_product( weight, radius_to_cg)

def Angular_Momentum_l2_about_hip():
	'''
	this function will use equations of motion in a cylindrical path to determine
	the angular momentum about of link 2 about the hip
	'''

def angular_momentum_l1_about_hp():
	'''
	this function will use equations of motion in a cylindrical path to determine
	the angular momentum about of link 1 about the hip
	'''