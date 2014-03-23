# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# import modules
import numpy as np

import scipy as sc
from scipy import misc
from scipy import special

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc

# <codecell>

# Note: this transforms A to B
def DCM(theta):
    C = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    return C

# <codecell>

# function that converts cartesian coordinates to pherical
def cart_to_sph(r):
        """
        Converts from cartesian coords to spherical
        inputs:  
            r_c   - matrix of cartesian particle coordinates, r_c[i] = [xi yi zi]
            r_sph - martis of spherical particle cooridnates, r_sph[i] = [rho_i alpha_i beta_i]
        """
        # Define this quantity since it is used multiple times
        r01 = r[:,0]**2 + r[:,1]**2
        
        # initialize the new vector 
        r_sph = np.empty(r.shape)
        
        # compute new vector quantities
        r_sph[:,0] = np.sqrt(r01 + r[:,2]**2)
        r_sph[:,1] = np.arctan2(np.sqrt(r01), r[:,2]) # for elevation angle defined from Z-axis down
        r_sph[:,2] = np.arctan2(r[:,1], r[:,0])
        
        # return new spherical coords dictionary
        r_sph = [dict(zip(['rho','alpha','beta'], r)) for r in r_sph]
        return r_sph

# <codecell>

# compute spherical harmonics using semi-normalized formula
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
        self.q_coords = q_coords
            
    def __repr__(self):
        """
        Defines the print method
        """
        return "Body - "                 + repr(self.iD)        + "\n" + \
               "N_charges = "            + repr(self.num_q)     + "\n" + \
               "Charge values = "        + repr(self.q)         + "\n" + \
               "Charge coords = " + "\n" + repr(self.q_coords)  + "\n"
                
    def mul_exp(self,p):
        """
        This function computes the multipole expansions for the componentwise force computation
            inputs:
                m - degree of the expansion
                n - order of the expansion
        """
        self.p = p
        self.M = np.array([[np.sum([q * qc['rho'] **(n) * my_sph(-m, n, qc['alpha'], qc['beta']) 
                                    for q,qc in zip(self.q,self.q_coords)]) 
                                    for m in range(-n,n+1)] for n in range(self.p+1)])
    
    def rotate(self, theta, alpha, beta, gamma):
        """
        Performs the rigid body rotation of the inertial properties and the rotation of 
        the multipole expansions
            inputs: 
                theta - angle for the kinematic rotations 
                ** not implemented yet (not needed at this time)
        """
        # initialze arrays with zeros
        C = np.zeros(3)
        d = [[[0.0+1j*0 for m in range(-n,n+1)] for mp in range(-n,n+1)] for n in range(self.p+1)]
        Mp = [[0.0+1j*0 for m in range(-n,n+1)] for n in range(self.p+1)]
        
        # also set to zeros
        D = d
        
        # d[0][0][0] and D[0][0][0] are known 
        d[0][0][0] = 1
        D[0][0][0] = 1

        Mp[0][0] = self.M[0][0]

        # recursive computation of terms of d and D matricies
        for n in range(1,self.p+1):
            for mp in range(-n,n+1):
                for m in range(-n,n+1):
                    if mp < -(n-1):
                        C[0] = np.sin(beta/2)**2*np.sqrt((n+m)*(n+m-1)/((n-mp)*(n-mp-1)))
                        C[1] = 2*np.sin(beta/2)*np.cos(beta/2) \
                                               *np.sqrt((n+m)*(n-m)/((n-mp)*(n-mp-1)))
                        C[2] = np.cos(beta/2)**2*np.sqrt((n-m)*(n-m-1)/((n-mp)*(n-mp-1)))
                        d[n][mp+n][m+n] = np.sum([C[i-m+1]*d[n-1][mp+1+(n-1)][i+(n-1)] 
                            for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    elif mp > (n-1):
                        C[0] = np.cos(beta/2)**2*np.sqrt((n+m)*(n+m-1)/((n+mp)*(n+mp-1)))
                        C[1] = -2*np.sin(beta/2)*np.cos(beta/2) \
                                                *np.sqrt((n+m)*(n-m)/((n+mp)*(n+mp-1)))
                        C[2] = np.sin(beta/2)**2*np.sqrt((n-m)*(n-m-1)/((n+mp)*(n+mp-1)))
                        d[n][mp+n][m+n] = np.sum([C[i-m+1]*d[n-1][mp-1+(n-1)][i+(n-1)] 
                            for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    else:
                        C[0] = np.sin(beta/2)*np.cos(beta/2) \
                                             *np.sqrt((n+m)*(n+m-1)/((n+mp)*(n-mp)))
                        C[1] = (np.cos(beta/2)**2-np.sin(beta/2))*np.sqrt((n-m)*(n+m)/((n-mp)))
                        C[2] = -np.sin(beta/2)*np.cos(beta/2) \
                                              *np.sqrt((n-m)*(n-m+1)/((n-mp)*(n+mp)))
                        d[n][mp+n][m+n] = np.sum([C[i-m+1]*d[n-1][mp+(n-1)][i+(n-1)] 
                            for i in range(np.max([m-1,-(n-1)]), np.min([m+1,n-1])+1)])
                    D[n][mp+n][m+n] = np.exp(1j*m*gamma)*d[n][mp+n][m+n]*np.exp(1j*m*alpha)
                Mp[n][mp+n] = np.dot(D[n][mp+n],self.M[n])
        self.M = Mp
    
    def potential(self, rp):
        """
        This function computes the couloumb potential due to a charged body at a 
        particluar point in space.
            inputs:
                loc - spherical coordinates of the point of interest
            outputs:
                Phi - potential
        """
        rp = rp[0]
        Phi = np.sum([np.sum([self.M[n][m+n]/rp['rho']**(n+1)
                              *my_sph(m, n, rp['alpha'], rp['beta']) 
                              for m in range(-n,n+1)]) for n in range(self.p+1)])
#         [[print('M[',n,'][',m+n,']= ',"{0:.3f}".format(self.M[n][m+n]),
#                 'rp^n = ',"{0:.3f}".format(rp['rho']**(n+1)),
#                 'Y(theta,phi) = ',"{0:.3f}".format(my_sph(m, n, rp['alpha'], rp['beta']))) 
#                               for m in range(-n,n+1)] for n in range(self.p+1)]
#         [print('Phi[',n,'] = ', "{0:.3f}".format(np.sum([self.M[n][m+n]/rp['rho']**(n+1)*my_sph(m, n, rp['alpha'], rp['beta']) 
#                               for m in range(-n,n+1)]))) for n in range(self.p+1)]
        return Phi

# <codecell>

def example(zeta,d,p):
    # Describe system        
    # Characteristic length        
    a = 1

    # charge values [O H H]
    q = [-1, 1]

    # location of charges w.r.t origin
    roq =  np.array([[-a/2, 0, 0],
                     [ a/2, 0, 0]])
    
    # Define test point
    rcq_p = np.array([0, -d, 0])


    # Compute the center of charge and locate particles w.r.t. center of charge
    rocq = np.sum([abs(q)*r for q,r in zip(q,roq)],0)/np.sum(np.abs(q))
    rcq_q = np.array([rq - rocq for rq in roq])
    # print(rocq)
    # print()
    # print(rcq_q)

    # array of rotation increments
    angle = np.array([0, zeta])

    # dipole orientation 'A'
    rcq_qa = rcq_q

    # Create system 'B' by rotating system 'A'
    CAB = DCM(angle[1])
    rcq_qbT = np.dot(CAB,rcq_qa.T)
    rcq_qb = rcq_qbT.T

    # Transform coordinates of point of interest
    rcq_p_sph = cart_to_sph(np.array([rcq_p]))

    # Transform coordinates of charge locations
    rcq_qa_sph = cart_to_sph(rcq_qa)
    rcq_qb_sph = cart_to_sph(rcq_qb)

    # Create a charged body for system A
    bodyA = ChargedBody(q, rcq_qa_sph, 1)
    bodyB = ChargedBody(q, rcq_qb_sph, 2)

    # Form Multipole Expansions
    bodyA.mul_exp(p)
    bodyB.mul_exp(p)

    # Evaluate Potential at a point [r theta phi]
#     PhiA = bodyA.potential(rcq_p_sph)
    PhiB = bodyB.potential(rcq_p_sph)
    PhiB = PhiB.real
#     print("Potential (via M.E.) of System B at point 'p' = ",PhiB.real)

    # Compute the exact solution 
    VB = np.sum([qb/np.linalg.norm(-rcq_p + r) for qb,r in zip(q,rcq_qb)])
#     print("Potential (exact) of System A at point 'p' = ",VA)
#     print("Potential (exact) of System B at point 'p' = ",VB)

    # Perform a rotation on System A so that it is the same configuration as B 
    alpha = -zeta
    beta = 0
    gamma = 0
    bodyA.rotate(0, alpha, beta, gamma)

    # Evaluate potential of 'A' at 'B'
    PhiA_B  = bodyA.potential(rcq_p_sph)
    PHiA_B = PhiA_B.real
    
    # Compute the error bound
    rr = np.linalg.norm(rcq_p)
    #     [print('rho = ',r['rho']) for r in rcq_qb_sph]
    aa = np.max([rq['rho'] for rq in rcq_qb_sph])
    #     print('a = ',a)
    Q = np.sum(np.abs(q))
    #     print('Q = ',Q)
    #     print('r - a = ',r - a)
    #     print('a/r = ',a/r)
    #     bound = Q/(rr-aa)*(aa/rr)**(p+1)/np.abs(VB)*100
    bound = Q/(rr-aa)*(aa/rr)**(p+1)
    return VB, PhiB, PhiA_B, bound

# <markdowncell>

# ### Try theta = 5 to 355 degrees and R/L = 1 to 10

# <codecell>

p  = 3
# Terms in multipole exansion 

# Number of data points in range
n_points = 20

# Create variable arrays
theta = np.linspace(0,2*np.pi,n_points)
dist = np.linspace(.6,2.5,n_points)

# Evaluate Error in List Comprehension
results = np.array([[example(zeta,d,p) \
                     for d in dist] for zeta in theta])
#                      for zeta in theta] for d in dist]);

VB = results[:,:,0].real
PhiB = results[:,:,1].real
PhiA_B = results[:,:,2].real
bound = results[:,:,3].real
VB_rms = np.sqrt(1/VB.size*np.linalg.norm(VB)**2)
print(VB_rms)

# Compute error and print results
# error_exact = np.abs((VB - PhiA_B))
error_exact = np.abs((VB - PhiA_B))/VB_rms*100
error_rot = np.abs((PhiB - PhiA_B))/VB_rms*100
bound = bound/VB_rms*100

# print("Potential (via M.E.) of System A rot to B evaluated at point 'p' = ",PhiA_B.real)
# print("% Error compared to exact = ","{0:.3f}".format(error_exact.real),'%')
# print("% Error bound = ","{0:.3f}".format(bound),'%')
# print("% Error compared to potential using M.E. of 'B' = ","{0:.3f}".format(error_rot.real),'%')

# <codecell>

# Plot the results
%matplotlib inline

plt.close('all')

R = (dist - 0.5)/1.0

levels = np.linspace(0,20,10)

plt.close('all')
plt.figure(figsize=(10, 8))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot the error vs. exact
fig = plt.figure(figsize=(10, 8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
levels = np.linspace(0,20,10)
CS = plt.contourf(R, theta*180/np.pi, error_exact, levels)
CB = plt.colorbar(CS, extend='both')
plt.title('Error vs Exact')
plt.xlabel(r'$\frac{R}{L}$')
plt.ylabel(r'Rotation Angle $\left(\theta\right)$')
plt.savefig('ErrorExact.pdf')
# plt.show()

# Plot the error vs. multipole
fig = plt.figure(figsize=(10, 8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
levels = np.linspace(0,20,10)
CS = plt.contourf(R, theta*180/np.pi, error_rot, levels)
CB = plt.colorbar(CS, extend='both')
plt.title(r'Error vs Multipole')
plt.xlabel(r'$\frac{R}{L}$')
plt.ylabel(r'Rotation Angle $\left(\theta\right)$')
plt.savefig('ErrorMultipole.pdf')
# plt.show()


# plt.subplot(2, 1, 1)
# plt.contourf(R, theta*180/np.pi, error_exact, levels)
# plt.title(r'Percent Error v.s. Exact')
# plt.ylabel(r'Rotation Angle $\left(\theta\right)')

# plt.subplot(2, 1, 2)
# plt.contourf(R, theta*180/np.pi, error_rot, levels)
# plt.title(r'Percent Error v.s. Direct Spatial Multipole')
# plt.xlabel(r'$\frac{R}{L}$')
# plt.ylabel(r'Rotation Angle $\left(\theta\right)')

# plt.savefig('ErrorSubplot.pdf')
# plt.show()

# # Two subplots, the axes array is 1-d
# fig, axarr = plt.subplots(2, sharex=True)
# axarr[0].contourf(R, theta*180/np.pi, error_exact, levels)
# axarr[0].set_title(r'Percent Error v.s. Exact')
# axarr[1].contourf(R, theta*180/np.pi, error_rot, levels)
# axarr[1].set_title(r'Percent Error v.s. Direct')

# # Plot the error bound
# plt.figure()
# levels = np.linspace(0,20,10)
# CS = plt.contourf(R, theta*180/np.pi, bound, levels)
# # plt.clabel(CS, inline=1, fontsize=10)
# # make a colorbar for the contour lines
# CB = plt.colorbar(CS, extend='both')
# plt.clabel(CS, fontsize=8, inline=1)
# plt.title('Error Bound')
# plt.savefig('ErrorBound', dpi=900)
# plt.show()

# <codecell>

 p = 2
# dist = 2
# theta = 0*np.pi/180
# error = example(theta,dist,p)
# RL = (dist - 0.5)/0.5
# print('R/L = ',RL,'theta = ',theta*180/np.pi)
# print('% Error Compared to Exact = ',error[0])
# print('% Error Compared to Direct Multipole = ',error[1])
# print('% Error Bound= ',error[2])

# <codecell>


