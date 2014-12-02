
# coding: utf-8

# In[1]:

# import modules
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from scipy import misc
from scipy import special
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# In[2]:

get_ipython().magic('matplotlib inline')
# %pdb


# ###Functions and Classes Needed for Charged Rigid Bodies

# In[3]:

# define roation matrix ccw positive (planar about 3rd axis)
# Note: this transforms A to B
def DCM(theta):
    C = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    return C


# In[4]:

# varialbes
# amount of rotation 
zeta = 45*np.pi/180

# number of terms in the multipole expansion
p = 3

# distance
d = 2


# ###Describe the problem

# ##### Assign charge values and position vectors (Oxygen at origin)

# In[5]:

# characteristic length        
a = 1

# describe anglular location of hydrogens 
# theta = 109.5/2*np.pi/180
# psi = -(np.pi/2 - theta)
# phi = 3/2*np.pi - theta

# charge values [O H H]
q = [1, -1]

# location of charges w.r.t origin
roq =  np.array([[-a/2, 0, 0],
                 [ a/2, 0, 0],
                 [   0, 0, 0]])


# ##### Compute the center of charge and locate particles w.r.t. center of charge

# In[6]:

rocq = np.sum([abs(q)*r for q,r in zip(q,roq)],0)/np.sum(np.abs(q))
rcq_q = np.array([rq - rocq for rq in roq])
# print(rocq)
# print()
# print(rcq_q)


# In[14]:

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

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

# array of rotation increments
angle = np.array([0, zeta, 2*zeta, 3*zeta])

# water orientation 'A'
rcq_qa = rcq_q

# system 'B' by rotating system 'A'
CAB = DCM(zeta)
rcq_qbT = np.dot(CAB,rcq_qa.T)
rcq_qb = rcq_qbT.T

# system 'C' by rotating system 'B'
CAB = DCM(zeta)
rcq_qcT = np.dot(CAB,rcq_qb.T)
rcq_qc = rcq_qcT.T

# system 'B' by rotating system 'A'
CAB = DCM(zeta)
rcq_qdT = np.dot(CAB,rcq_qc.T)
rcq_qd = rcq_qdT.T

R = [rcq_qa, rcq_qb, rcq_qc, rcq_qd]

# Test point
rcq_p = np.array([0, d, 0])

fig_size = (3,3)

# Graphical Representation of test systems
for i,r in enumerate(R,start=1):
    
    # Close all plots
    plt.close('all')
    # Use Latex Fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Create Plot
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1,1,1, projection='3d')

    # Plot Origin
#     ax.scatter(0, 0, 0, zdir='z', s=5, c='0.75')
    
    # plot scale
    l = d
    
    # Plot axis unit vectors
#     x_hat = Arrow3D([0,l/2],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
#     ax.add_artist(x_hat)
#     y_hat = Arrow3D([0,0],[0,l/2],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
#     ax.add_artist(y_hat)
#     z_hat = Arrow3D([0,0],[0,0],[0,l/2], mutation_scale=20, lw=1, arrowstyle="-|>", color="0.75")
#     ax.add_artist(z_hat)
    
    # plot points
    ax.scatter(r[0][0], r[0][1], r[0][2], zdir='z', s=50*np.log(l)*np.abs(q[0]), c='b')
    ax.scatter(r[1][0], r[1][1], r[1][2], zdir='z', s=50*np.log(l)*np.abs(q[1]), c='r')
    
    # Plot evaluation point
    ax.scatter(rcq_p[0], rcq_p[1], rcq_p[2], zdir='z', s=10*l, c='green')
    
    # axis limits
#     ax.set_xlim(-l, l)
#     ax.set_ylim(-l, l)
#     ax.set_zlim(-l, l)

    ax.set_xticks(np.arange(-l,l+1,2))
    ax.set_yticks(np.arange(-l,l+1,2))
    ax.set_zticks(np.arange(-l,l+1,2))
    
#     majorLocator   = MultipleLocator(1)
#     majorFormatter = FormatStrFormatter('%d')
#     minorLocator   = MultipleLocator(1)
    
#     ax.xaxis.set_major_locator(majorLocator)
#     ax.xaxis.set_major_formatter(majorFormatter)
#     ax.yaxis.set_major_locator(majorLocator)
#     ax.yaxis.set_major_formatter(majorFormatter)
#     ax.zaxis.set_major_locator(majorLocator)
#     ax.zaxis.set_major_formatter(majorFormatter)
    

#     #for the minor ticks, use no labels; default NullFormatter
#     ax.xaxis.set_minor_locator(minorLocator)
#     ax.yaxis.set_minor_locator(minorLocator)
#     ax.zaxis.set_minor_locator(minorLocator)
    
    # axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # plot title
#     ax.set_title('Rotation '+repr(angle[i-1]*180/np.pi))
    
    ax.view_init(elev=30., azim=60.0)
    plt.savefig('dipole_'+"{0:.0f}".format(angle[i-1]*180/np.pi)+'.pdf') 
    print('dipole_'+"{0:.0f}".format(angle[i-1]*180/np.pi)+'.pdf')
    
    plt.show()


# In[7]:



