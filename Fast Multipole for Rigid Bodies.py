# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ##A simple problems from Physics 301 Fall 2010 Problem set 7 (3D-example)

# <markdowncell>

# <p>
#         Import Modules
#         </p>

# <codecell>

import ipdb

import numpy as np

import scipy as sc
from scipy import misc
from scipy import special

import matplotlib.pyplot as pyplt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
%matplotlib inline

# <codecell>

def my_sph(m,n,theta,phi):
    x = np.cos(theta)
    C = np.sqrt(sc.misc.factorial(n-np.abs(m))/sc.misc.factorial(n+np.abs(m)))
    Pmn = (-1)**np.abs(m)*(1-x**2)**(np.abs(m)/2)*sc.special.eval_legendre((n-np.abs(m)), x)
    Ymn = C*Pmn*sc.exp(1j*m*phi)
    return Ymn

# <codecell>

class ChargedBody:
    """
    This is the class that contains the charge properties of a body
        inputs: 
            q        - list of the value of charge for the body
            q_coords - list of x and y coords
            iD       - number of the body
    """
    # Initialize instance
    def __init__(self, q, q_coords, iD):
        self.q        = q
        self.iD       = iD
        self.num_q    = len(q)
        self.q_coords = q_coords # q_coords[i] = [rho_i alpha_i beta_i]
        self.M        = []
            
    def __repr__(self):
        """
        Defines the print method
        """
        return "Body - "                 + repr(self.iD)        + "\n" + \
               "N_charges = "            + repr(self.num_q)     + "\n" + \
               "Charge values = "        + repr(self.q)         + "\n" + \
               "Charge coords = " + "\n" + repr(self.q_coords)  + "\n" + \
               "M = "                    + repr(self.M)          
                
    def mul_exp(self,p):
        """
        This function computes the multipole expansions for the componentwise force computation
            inputs:
                m - degree of the expansion
                n - order of the expansion
        """
        self.p = p
        self.M = np.array([[np.sum([self.q[i] * self.q_coords[i][0]**(n) * \
                           my_sph(-m, n, self.q_coords[i][1], self.q_coords[i][2]) \
                           for i in range(self.num_q)]) for m in range(-n,n+1)] for n in range(p)])
    
    def rotate(self, theta, alpha, beta, gamma):
        """
        Performs the rigid body rotation of the inertial properties and the rotation of the multipole expansions
            inputs: 
                theta - angle for the rigid body rotation ** not implemented yet
        """
        d = [[[0 for m in range(-n,n+1)] for mp in range(-n,n+1)] for n in range(self.p)]
        D = d
        d[0][0][0] = 1
        C = np.zeros(3)
        
        for n in range(1,p):
            for mp in range(-n,n+1):
                for m in range(-n,n+1):
                    if mp < -(n-1):
                        C[0] = np.sin(beta/2)**2*np.sqrt((n+m)*(n+m-1)/((n-mp)*(n-mp-1)))
                        C[1] = 2*np.sin(beta/2)*np.cos(beta/2)*np.sqrt((n+m)*(n-m)/((n-mp)*(n-mp-1)))
                        C[2] = np.cos(beta/2)**2*np.sqrt((n-m)*(n-m-1)/((n-mp)*(n-mp-1)))
                        d[n][mp+n][m+n] = np.sum([C[i+(n-1)]*d[n-1][mp+1+(n-1)][i+(n-1)] 
                                                  for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    elif mp > (n-1):
                        C[0] = np.cos(beta/2)**2*np.sqrt((n+m)*(n+m-1)/((n+mp)*(n+mp-1)))
                        C[1] = -2*np.sin(beta/2)*np.cos(beta/2)*np.sqrt((n+m)*(n-m)/((n+mp)*(n+mp-1)))
                        C[2] = np.sin(beta/2)**2*np.sqrt((n-m)*(n-m-1)/((n+mp)*(n+mp-1)))
                        d[n][mp+n][m+n] = np.sum([C[i+(n-1)]*d[n-1][mp-1+(n-1)][i+(n-1)] 
                                                  for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    else:
                        C[0] = np.sin(beta/2)*np.cos(beta/2)*np.sqrt((n+m)*(n+m-1)/((n+mp)*(n-mp)))
                        C[1] = (np.cos(beta/2)**2-np.sin(beta/2))*np.sqrt((n-m)*(n+m)/((n-mp)))
                        C[2] = -np.sin(beta/2)*np.cos(beta/2)*np.sqrt((n-m)*(n-m+1)/((n-mp)*(n+mp)))
                        d[n][mp+n][m+n] = np.sum([C[i+(n-1)]*d[n-1][mp+(n-1)][i+(n-1)] 
                                                  for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    D[n][mp+n][m+n] = np.exp(1j*(m+n)*gamma)*d[n][mp+n][m+n]*np.exp(1j*(m+n)*alpha)
                self.M[n][mp+n] = np.dot(D[n][mp+n]*self.M[n][mp+n])
        return D,d
              
    def potential(self, loc):
        """
        This function computes the couloumb potential due to a charged body at a particluar point in space.
            inputs:
                loc - spherical coordinates of the point of interest
            outputs:
                Phi - potential
        """
        Phi = np.sum([np.sum([self.M[n][m+n]/loc[0]**(n+1)*my_sph(m, n, loc[1], loc[2]) \
                              for m in range(-n,n+1)]) for n in range(self.p)])
        return Phi

# <codecell>

# Describe the test systems

# Classes needed for arrow plotting
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# characteristic length        
a = 1

# system a
r_a = np.array([[0, 0, a],[0, 0, 0]])
q_a = np.array([3, -1])

# system b
r_b = np.array([[0, 0, 0],[0, 0, -a]])
q_b = np.array([3, -1])

# system C
r_c = np.array([[0, a, 0],[0, 0, 0]])
q_c = np.array([3, -1])


# Graphical Representation of test systems
fig = pyplt.figure(figsize=(18,6))

# System 1
# Plot unit vectors
ax = fig.add_subplot(1, 3, 1, projection='3d')
x_hat = Arrow3D([0,.5],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(x_hat)
y_hat = Arrow3D([0,0],[0,.5],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(y_hat)
z_hat = Arrow3D([0,0],[0,0],[0,.5], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(z_hat)

# Plot charge locations
ax.scatter(r_a[0,0], r_a[0,1], r_a[0,2], zdir='z', s=60, c='r')
ax.scatter(r_a[1,0], r_a[1,1], r_a[1,2], zdir='z', s=20, c='b')

# axis limits
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot title
ax.set_title('System A')

# System 2
# Plot unit vectors
ax = fig.add_subplot(1, 3, 2, projection='3d')
x_hat = Arrow3D([0,.5],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(x_hat)
y_hat = Arrow3D([0,0],[0,.5],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(y_hat)
z_hat = Arrow3D([0,0],[0,0],[0,.5], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(z_hat)

# plot charges
ax.scatter(r_b[0,0], r_b[0,1], r_b[0,2], zdir='z', s=60, c='r')
ax.scatter(r_b[1,0], r_b[1,1], r_b[1,2], zdir='z', s=20, c='b')

# axis limits
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot title
ax.set_title('System B')

# System 3
# Plot unit vectors
ax = fig.add_subplot(1, 3, 3, projection='3d')
x_hat = Arrow3D([0,.5],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(x_hat)
y_hat = Arrow3D([0,0],[0,.5],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(y_hat)
z_hat = Arrow3D([0,0],[0,0],[0,.5], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
ax.add_artist(z_hat)

# Plot charge locations
ax.scatter(r_c[0,0], r_c[0,1], r_c[0,2], zdir='z', s=60, c='r')
ax.scatter(r_c[1,0], r_c[1,1], r_c[1,2], zdir='z', s=20, c='b')

# axis limits
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot title
ax.set_title('System C')

pyplt.show()

# <codecell>

# Convert the difference systems to polar
def cart_to_sph(r_c):
    """
    Converts from cartesian coords to spherical
    inputs:  
        r_c   - matrix of cartesian particle coordinates, r_c[i] = [xi yi zi]
        r_sph - martis of spherical particle cooridnates, r_sph[i] = [rho_i alpha_i beta_i]
    """
    # Define this quantity since it is used multiple times
    r01 = r_c[:,0]**2 + r_c[:,1]**2
    
    # initialize the new vector 
    r_sph = np.empty(r_c.shape)
    
    # compute new vector quantities
    r_sph[:,0] = np.sqrt(r01 + r_c[:,2]**2)
    r_sph[:,1] = np.arctan2(np.sqrt(r01), r_c[:,2]) # for elevation angle defined from Z-axis down
    r_sph[:,2] = np.arctan2(r_c[:,1], r_c[:,0])
    
    # return new corrdinates
    return r_sph

q_coords_a = cart_to_sph(r_a)
q_coords_b = cart_to_sph(r_b)
q_coords_c = cart_to_sph(r_c)

q_coords_a_deg = q_coords_a
q_coords_b_deg = q_coords_b
q_coords_c_deg = q_coords_c

q_coords_a_deg[:,1:3] = q_coords_a[:,1:3]*180/np.pi
q_coords_b_deg[:,1:3] = q_coords_b[:,1:3]*180/np.pi
q_coords_c_deg[:,1:3] = q_coords_c[:,1:3]*180/np.pi

# print('q_coords_a = ')
# print(q_coords_a_deg)
# print('\n')
# print('q_coords_b = ')
# print(q_coords_b_deg)
# print('\n')
# print('q_coords_c = ')
# print(q_coords_c_deg)

# <markdowncell>

# ###Evaluate Potential from Multipole and Check Agaist Known Solution

# <codecell>

# Create a charged body for system A
bodyA = ChargedBody(q_a, q_coords_a, 1)
print(bodyA)

# <codecell>

# Form p = 3 Multipole Expansion
p = 3
bodyA.mul_exp(p)
# for i in range(p):
#     print('Body A M['+repr(i)+'] = ' + repr(bodyA.M[i]))

# <codecell>

# Evaluate Potential at a point [r theta phi] for body A
r = np.array([[-1, 2, 1]])
loc = cart_to_sph(r)
Phi = bodyA.potential(loc[0])
print(Phi)

# <codecell>

V = np.sum([q/np.linalg.norm(-r + p) for q,p in zip(q_a,r_a)])
print(V)

# <codecell>

alpha = 0
beta = np.pi/2
gamma = np.pi/2
bodyA.rotate(0, alpha, beta, gamma) 

# <codecell>


