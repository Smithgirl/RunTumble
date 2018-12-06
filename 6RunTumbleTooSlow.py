# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:12:29 2018

@author: JINGYUE 

This code has not been optimised
"""
import matplotlib.pyplot as plt  
import random
import math 
import numpy as np
import time
import scipy.constants as sc

# to start the timer:
start_time = time.time()  

# the constants:
hotIndex = 820 # the chosen index number of the hot particle os that it's near the centre of the plot
d = 86  # dimension of the square box, to ensure 0.68 area fraction
n = 1600  # number of particles 
a = 1 # radius of the particles 
D = 0.1 # the diffusion constant
T = 298 # the passive medium temperature
u = (sc.k)*T*10**13 # the mobility of the particle， from the formula that u = Kb*T/D
k = 100*D/(u*(a**2)) # the spring constant 
vo = 100 # the self propulsion velocity, 100
c = 4430 #### the number of timesteps iterated, by right, it's 2,000,000 = 10 time units
t = 0.5*(10**(-5)) #### **(-5) unit length of the timestep/time interval
tao = 0.00025 # tumbling coefficient, when it's 1, the active particle will produce an empty streak. When it's 0.01 or below, a void is expected            
rott = tao*(a**2)*D  # the time interval between tumblings, equal to 2.5e-5
m = rott/t  # number of frames between each tumling event
l = (d - 2*a*math.sqrt(n))/math.sqrt(n)  # the width of the fringe around each particle x2
it0 = 10   # it0, t0max, and tmax are related to the diffuse() function
t0max = int((c - it0)/it0)
tmax = c - it0
z = 1000  # the plot() function is called every z timesteps

# the variables:
angle = 0 # random.uniform(-math.pi, math.pi)
_ = 0 # just a frame counter
xy = np.empty([n, 2])  # an empty array xy that will store elements of the form [x, y]
i = (2*a + l) / 2   # the start of the lattice distribution
p = 0

# to first distribute the particles in a semi-random square lattice
while i < d:  
    j = (2*a + l) / 2
    while j < d:
        cur = [0, 0]
        cur[0] = i + random.uniform(-l/2, l/2)
        cur[1] = j + random.uniform(-l/2, l/2)
        xy[p] = cur[:]
        j += 2*((2*a + l) / 2)
        p += 1
    i += 2*((2*a + l) / 2)

def diffuse(switch, nsamp ):
    """this function calculates the velocity autocorrelation function and the mean-squared displacement"""
    global ntel, t0, vxy, time0, x0, y0, vx0, vy0, ntime, vacfX, vacfY, vacfXY, r2tX, r2tY, r2tXY
    if (switch == 0):   # to initialise
        ntel = 0
        t0 = 0
        #dtime = dt*nsamp   
        ntime = np.zeros(tmax)
        vacfX = np.zeros(tmax)
        vacfY = np.zeros(tmax)
        vacfXY = np.zeros(tmax)
        r2tX = np.zeros(tmax)
        r2tY = np.zeros(tmax)
        r2tXY = np.zeros(tmax)
        time0 = np.zeros(t0max, dtype=int)
        x0 = np.zeros((n, t0max))
        y0 = np.zeros((n, t0max))
        vx0 = np.zeros((n, t0max))
        vy0 = np.zeros((n, t0max))
    
    elif (switch == 1):    # called at each time step
        ntel += 1
        if (ntel % it0 == 0):   # if-statement will only be activated when ntel is greater or equal to it0
            t0 += 1
            tt0 = ((t0 - 1) % t0max) + 1
            time0[tt0 - 1] = ntel
            for q in range(n):
                x0[q][tt0 - 1] = xy[q][0]
                y0[q][tt0 - 1] = xy[q][1]
                vx0[q][tt0 - 1] = vxy[q][0]
                vy0[q][tt0 - 1] = vxy[q][1]
        for p in range(0, min(t0, t0max)):  
            delt = ntel - time0[p]
            if (delt < tmax + 1):
                ntime[delt] += 1
                for q in range(len(xy)):
                    vacfX[delt] += vxy [q][0]*vx0[q][p]
                    vacfY[delt] += vxy[q][1]*vy0[q][p]
                    r2tX[delt] += (xy[q][0] - x0[q][p])**2
                    r2tY[delt] += (xy[q][1] - y0[q][p])**2
                    
    elif (switch == 2):
        for p in range(tmax):
            #time = dtime*(p + 0.5)
            vacfX[p] /= (n*ntime[p])
            vacfY[p] /= (n*ntime[p])
            vacfXY[p] = vacfX[p] + vacfY[p]
            r2tX[p] /= (n*ntime[p])
            r2tY[p] /= (n*ntime[p])
            r2tXY[p] = r2tX[p] + r2tY[p]
        print("length of the arrays:", (len(vacfX), len(vacfY), len(r2tX), len(r2tY), len(vacfXY), len(r2tXY)))    
    return        

def distance(A, B):   # to calculate the distance between any two particles in the 2D space and the calculation follows the minimal image convention
    d0 = A[0] - B[0]    # A is the particle of interest, B is its neighbour
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
    dis = distance(ri, rj)
    magP = k*(2*a-dis)    # magnitude of the potential force
    forceX = (ri[0] - rj[0])/dis   # direction of the potential force
    forceY = (ri[1] - rj[1])/dis
    forceMag = [u*magP*forceX, u*magP*forceY]   # multiplied by the mobility term u
    return forceMag

def afterOnePotentialStep(f1):
    global vxy
    f1Duplicate = []
    for i in range(len(f1)):
        f1DuplicateElement = f1[i]
        for j in range(len(f1)):
            if (j != i) and (distance(f1[i], f1[j]) < 2*a):   # here we impose an equilibrium neighbour distance 2a and the distance follows minimal image convention
                vxy[i, 0] += truncatedHarmonicPotential(f1[i], f1[j])[0]
                vxy[i, 1] += truncatedHarmonicPotential(f1[i], f1[j])[1]
        f1DuplicateElement[0] += t*vxy[i, 0]
        f1DuplicateElement[1] += t*vxy[i, 1]        
        f1Duplicate.append(f1DuplicateElement)
    f1 = f1Duplicate
    return f1

def thermalNoiseMagnitude():   # the magnitude is 2D*standardnormal, and the direction is random.uniform, the noise is a list of 2 elements
    thetaT = random.uniform(-math.pi, math.pi)
    mag = math.sqrt(2*D)*np.random.normal()
    noiseX = math.cos(thetaT)
    noiseY = math.sin(thetaT)
    noiseMag = [mag*noiseX, mag*noiseY]    
    return noiseMag

def afterOneNoiseStep(f2):    # to generate the updated particle positions after one run step
    global vxy
    for i in range(len(f2)):
        particleIpos = f2[i]
        tn = thermalNoiseMagnitude()
        particleIpos[0] = particleIpos[0] + t*tn[0]    # to update the particle i's position after the one run step
        vxy[i, 0] += tn[0]
        particleIpos[1] = particleIpos[1] + t*tn[1]
        vxy[i, 1] += tn[1]
        f2[i] = particleIpos    # to put the updated position back to the list xy
    return f2

def runStep(theta):   # to take a step of size vo in the current theta angle per frame
    xo = math.cos(theta)
    yo = math.sin(theta)
    O = [vo*xo, vo*yo]
    return O

def afterOneRunStep(f3):
    global vxy
    particleIpos = f3[hotIndex]
    rs = runStep(angle)
    particleIpos[0] = particleIpos[0] + t*rs[0]  # to update the particle i's position after the one runStep
    particleIpos[1] = particleIpos[1] + t*rs[1]
    vxy[hotIndex, 0] = rs[0]
    vxy[hotIndex, 1] = rs[1]
    f3[hotIndex] = particleIpos    # to put the updated position back to the list xy
    return f3

def PBC(f4):   # PBC algorithm--continuity
    for i in range(len(f4)):
        f4Element = f4[i]
        for m in range(2):
            if (f4Element[m] > d) or (f4Element[m] == d):
                f4Element[m] = f4Element[m] % d
            if (f4Element[m] < 0):
                f4Element[m] = - f4Element[m] % d 
        f4[i] = f4Element
    return f4

def oneMotionFrameWithRun(f):     # after a time interval of t (the timestep)
    f = afterOnePotentialStep(f)   # the effect is considered serial in the presented order
    f = afterOneNoiseStep(f)   # the displacement due to noise is independent of the particle positions
    f = afterOneRunStep(f)     
    f = PBC(f)
    return f

def plotScatter(mn):
    fig = plt.figure() # to create a figure object after each frame of motion
    axes = fig.gca()  # to call for the axes
    axes.set_xlim(0,d) # to set the range for the axes
    axes.set_ylim(0,d) 
    x, y = zip(*mn) # to create a list of two lists, each has a series of numbers for x, y coordinates, respectively
    axes.scatter(x, y, c='r', alpha = 0.5, s=20)  # to plot a 3D graph with the input for x,y,z
    axes.scatter(x[hotIndex], y[hotIndex], c='g', alpha = 0.75, s=20)
    plt.show()
    
def plotLine(pac):
    for i in pac:
        plt.plot(i)
        plt.show()
    return

print("the simulation models the particle motion every", t, "unit time interval, for", c, "iterations, across a time span of", c*t, "unit(s)")
diffuse(0, 100)

# the execution and data-tallying
while ( _ < c + 1):     
    #print(xy)    
    if ( _ % z == 0):    
       plotScatter(xy) 
       print("position of the particles after", _, "iteration(s)")
    _ += 1
    if ( _ > c):   # to avoid calling oneMotionFrameWithRun in vain
        break
    if ( _ % m == 0):   # when the tumbling time is reached
        vxy = np.zeros((len(xy), 2))
        angle = random.uniform(-math.pi, math.pi)   # to tumble
        xy = oneMotionFrameWithRun(xy)
        diffuse(1, 100)
    else: 
        vxy = np.zeros((len(xy), 2))
        xy = oneMotionFrameWithRun(xy)  # to continue running
        diffuse(1, 100)
diffuse(2, 100)

# to present the data
plotLine([vacfX, vacfY, vacfXY, r2tX, r2tY, r2tXY])
print(_)
print(ntel)
print(vxy[6])
    
plt.close('all')
print("--- took %s seconds to run---" % (time.time() - start_time))