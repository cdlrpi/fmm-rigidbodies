# import modules
import numpy as np

import scipy as sc
from scipy import misc
from scipy import special

# Note: this transforms A to B
def DCM(theta):
    C = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    return C

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

# compute spherical harmonics using semi-normalized formula
def my_sph(m,n,theta,phi):
    x = np.cos(theta)
    C = np.sqrt(sc.misc.factorial(n-np.abs(m))/sc.misc.factorial(n+np.abs(m)))
    Pmn = (-1)**np.abs(m)*(1-x**2)**(np.abs(m)/2)*sc.special.eval_legendre((n-np.abs(m)), x)
    Ymn = C*Pmn*sc.exp(1j*m*phi)
    return Ymn

# class ChargedBody:
cdef class ChargedBody:
    """
    This is the class that contains the charge properties of a body
        inputs: 
            q        - list of the value of charge for the body
            q_coords - list of x and y coords
            iD       - number of the body
    """
    # Initialize instance
    # def __init__(self, q, q_coords, iD):
    def __cinit__(self, q, q_coords, iD):
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
        return Phi

def example(zeta,d,p):
    # Describe system        
    # Characteristic length        
    a = 1

    # charge values [O H H]
    q = [1, -1]

    # location of charges w.r.t origin
    roq =  np.array([[-a/2, 0, 0],
                     [ a/2, 0, 0]])

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

    # Define test point
    rcq_p = np.array([0, -d, 0])

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
    PhiA = bodyA.potential(rcq_p_sph)
    PhiB = bodyB.potential(rcq_p_sph)
    # print("Potential (via M.E.) of System B at point 'p' = ",PhiB.real)

    # Compute the exact solution 
    VB = np.sum([qb/np.linalg.norm(-rcq_p + r) for qb,r in zip(q,rcq_qb)])
    # print("Potential (exact) of System A at point 'p' = ",VA)
    # print("Potential (exact) of System B at point 'p' = ",VB)

    # Perform a rotation on System A so that it is the same configuration as B 
    alpha = -zeta
    beta = 0
    gamma = 0
    bodyA.rotate(0, alpha, beta, gamma)

    # Evaluate potential of 'A' at 'B'
    PhiA_B  = bodyA.potential(rcq_p_sph)

    # Compute error and print results
    error_exact = np.abs((PhiA_B-VB)/VB)*100
    error_rot = np.abs((PhiA_B-PhiB)/PhiB)*100

    rr = np.linalg.norm(rcq_p)
    aa = np.max([r['rho'] for r in rcq_qb_sph])

    # Compute the error bound
    bound = (np.sum(np.abs(q))/(rr-aa)*(aa/rr)**(p+1))/VB*100

    # print("Potential (via M.E.) of System A rot to B evaluated at point 'p' = ",PhiA_B.real)
    # print("% Error compared to exact = ","{0:.3f}".format(error_exact.real),'%')
    # print("% Error bound = ","{0:.3f}".format(bound),'%')
    # print("% Error compared to potential using M.E. of 'B' = ","{0:.3f}".format(error_rot.real),'%')
    return error_exact.real
