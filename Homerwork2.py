#! /usr/bin/env python
#
# Model with two basins making use of TensorFlow
# - more about TensorFlow on www.tensorflow.org
#   or more specifically about automatic differentiation 
#       https://www.tensorflow.org/tutorials/customization/autodiff
# - this model is part of a homework exercise for the course WI4475 at the TU Delft.
#
# model description
# Consider a free falling spherial object in air
# we model the air friction with a quadratic drag relation
# application of Newton's famous F=ma gives
#
# m du/dt = - m g - rho_a cd A |u|u
# with:
# m=rho_o 4/3 pi r^3
# A=4 pi r^2
# du/dt = -g - (3 rho_a cd)/(rho_o r) |u|u
# du/dt = -g - beta |u|u
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# read measurements from file
data = np.genfromtxt('data.csv', delimiter=' ')

# Here we make most variables of type tf.Variable, which is not necessary for many of these variables,
# but here we err on the safe side
dt   =0.1                 #time-step [s]
T    =10.0                #end-time of simulation
g    =tf.Variable(9.81)   #acceleration of gravity [m/s^2]
rho_o=tf.Variable(1000.0) #density of object [kg/m^3]
r    =tf.Variable(0.05)   #radius of spherical object
cd   =tf.Variable(0.45)   #drag coefficient for a sphere (0.47) 
                          #https://en.wikipedia.org/wiki/Drag_coefficient
rho_a=tf.Variable(1.25)   #density of air

beta = 3*rho_a*cd/(rho_o*r)
z=tf.Variable(0.0)        #initial position
u=tf.Variable(0.0)        #initial velocity
i=0
times=np.arange(dt,T,dt) 
all_u=np.zeros(len(times))
cost=tf.Variable(0.0)
for t in np.arange(dt,T,dt):
    z = z + dt*u
    u = u + dt*(-g-beta*abs(u)*u)
    all_u[i]=u
    #print("t=%f z=%f u=%f"%(t,z,u))
    z_obs=data[i,1]
    print("t=%f z=%f z_obs=%f"%(t,z,z_obs))
    cost=cost+(z_obs-z)**2
    i=i+1
cost_norm=cost/i #normalize by number of elements
print("cost_norm=%f"%(cost_norm))

plt.plot(times,all_u)         #computed velocity
#plt.plot(data[:,0],data[:,2]) #measured velocity
plt.xlabel("time [s]")
plt.ylabel("u [m/s]")
plt.title("Fall velocity")

# =============================================================================
# 2.d derivative of cost_norm with respect to cd and r
# =============================================================================
with tf.GradientTape(persistent=True) as tape:
    fcost = ((data[i-1,1] - dt*(dt*(-g-(3*rho_a*cd/(rho_o*r))*abs(u)*u)))**2)/i
    print("f_cost=%f"%(fcost))

dfcost_dcd = tape.gradient(fcost,cd)
print("dfcost/dcd=%f"%(dfcost_dcd))
dfcost_dr = tape.gradient(fcost,r)
print("dfcost/dr=%f"%(dfcost_dr))

# =============================================================================
# 2.e add small random perturbation of the input variables and perform the gradient
# =============================================================================
# cd=tf.Variable(2.0)
# r=tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
    fcost = ((data[i-1,1] - dt*(dt*(-g-(3*rho_a*cd/(rho_o*r))*abs(u)*u)))**2)/i

print("small random perturbation")
dfcost_dcd = tape.gradient(fcost,cd)
print("dfcost/dcd=%f"%(dfcost_dcd))
dfcost_dr = tape.gradient(fcost,r)
print("dfcost/dr=%f"%(dfcost_dr))
dfcost_dcdr = tape.gradient(fcost,[cd,r])
print("grad(fcost) wrt (cd,r) = ",np.array(dfcost_dcdr))