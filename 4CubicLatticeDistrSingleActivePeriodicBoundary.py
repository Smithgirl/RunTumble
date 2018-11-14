# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:25:42 2018

@author: JINGYUE
"""

# Python code for generating a square lattice distribution of walkers, all passive except one active, no boundary imposed
import matplotlib.pyplot as plt # a collection of functions that make some change to a figure  
import random
import math 
import numpy as np

hotIndex = 820   # the chosen index number of the hot particle os that it's near the centre of the plot
d = 86  # dimension of the square box
n = 1600  # number of particles 
a = 1 # radius of the particles
k = 0.00001 # the spring constant 
vo = 0.5 # the self propulsion velocity
D = 0.1 # the diffusion constant
_ = 0 # just a frame counter
xy = []  # an empty list xy
i = 1 + (d - 2*a*math.sqrt(n))/2   # the start of the lattice distribution

while i < d - (d - 2*a*math.sqrt(n))/2:     # to first distribute the particles in a square lattice
    j = 1 + (d - 2*a*math.sqrt(n))/2
    while j < d - (d - 2*a*math.sqrt(n))/2:
        cur = [0, 0]
        cur[0] = i
        cur[1] = j
        xy.append(cur[:])
        j += 2*a
    i += 2*a

def distance(A, B): # to calculate the distance between any two particles in the 2D space and the calculation follows the minimal image convention
    d0 = A[0] - B[0]    # A is the particle of interest
    if d0 > d*0.5: 
        d0 = d0 - d
    if (d0 < - d*0.5) or (d0 == -d*0.5):
        d0 = d0 + d
    d1 = A[1] - B[1]
    if d1 > d*0.5: 
        d1 = d1 - d
    if (d1 < - d*0.5) or (d1 == -d*0.5):
        d1 = d1 + d
    dist = math.sqrt(d0**2 + d1**2)
    return dist

def truncatedHarmonicPotential(ri, rj):  # U = 0.5*k*(2*a - distance(ri - rj))**2, F = /K*delDistance/
    magP = k*(2*a-distance(ri, rj))   # magnitude of the potential force
    forceX = (ri[0] - rj[0])/distance(ri, rj)   # direction of the potential force
    forceY = (ri[1] - rj[1])/distance(ri, rj)
    forceMag = [magP*forceX, magP*forceY]
    return forceMag

def afterOnePotentialStep(xy):
    xyDuplicate = []
    for i in range(len(xy)):
        xyElement = xy[i]
        for j in range(len(xy)):
            if (j != i) and (distance(xy[i], xy[j]) < 2*a):     # here we impose an equilibrium neighbour distance 2a and the distance follows minimal image convention
                xyElement[0] += truncatedHarmonicPotential(xy[i], xy[j])[0]
                xyElement[1] += truncatedHarmonicPotential(xy[i], xy[j])[1]
        xyDuplicate.append(xyElement)
    return xyDuplicate

def thermalNoiseMagnitude():    # the magnitude is 2D*standardnormal, and the direction is random.uniform, the noise is a list of 2 elements
    thetaT = random.uniform(-math.pi, math.pi)
    mag = math.sqrt(2*D)*np.random.normal()
    noiseX = math.cos(thetaT)
    noiseY = math.sin(thetaT)
    noiseMag = [mag*noiseX, mag*noiseY]    
    return noiseMag

def afterOneNoiseStep(xy):    # to generate the updated particle positions after one run step
    for i in range(len(xy)):
        particleIpos = xy[i]
        particleIpos[0] = particleIpos[0] + thermalNoiseMagnitude()[0] # to update the particle i's position after the one run step
        particleIpos[1] = particleIpos[1] + thermalNoiseMagnitude()[1]
        xy[i] = particleIpos    # to put the updated position back to the list xy
    return xy

def runStep(): # to take a step of size vo in a random angle theta per frame
    theta = random.uniform(-math.pi, math.pi)
    xo = math.cos(theta)
    yo = math.sin(theta)
    O = [vo*xo, vo*yo]
    return O

def afterOneRunStep(xy):
    particleIpos = xy[hotIndex]
    particleIpos[0] = particleIpos[0] + runStep()[0] # to update the particle i's position after the one run step
    particleIpos[1] = particleIpos[1] + runStep()[1]
    xy[hotIndex] = particleIpos    # to put the updated position back to the list xy
    return xy

def PBC(xy):  # PBC algorithm--continuity
    for i in range(len(xy)):
        xyElement = xy[i]
        for m in range(2):
            if (xyElement[m] > d) or (xyElement[m] == d):
                xyElement[m] = xyElement[m] % d
            if (xyElement[m] < 0):
                xyElement[m] = - xyElement[m] % d 
        xy[i] = xyElement
    return xy

def oneMotionFrame(xy):
    afterOnePotentialStep(xy)   # the effect is considered serial in the presented order
    afterOneNoiseStep(xy)   # the displacement due to noise is independent of the particle positions
    afterOneRunStep(xy)     # the running direction is assumed to be independent of the particle orientation in the previous frame here
    PBC(xy)
    return xy

while ( _ < 10):     # the execution of one motion frame combining all forces
    xy = oneMotionFrame(xy)
    #print(xy)    
    fig = plt.figure() # to create a figure object after each frame of motion
    axes = fig.gca()  # to call for the axes
    axes.set_xlim(0,d) # to set the range for the axes
    axes.set_ylim(0,d)
    x, y = zip(*xy) # to create a list of two lists, each has a series of numbers for x, y coordinates, respectively
    axes.scatter(x, y, c='r', alpha = 0.5, s=20)  # to plot a 3D graph with the input for x,y,z
    axes.scatter(x[hotIndex], y[hotIndex], c='g', alpha = 0.75, s=20)
    plt.show()
    _ += 1

plt.close('all')