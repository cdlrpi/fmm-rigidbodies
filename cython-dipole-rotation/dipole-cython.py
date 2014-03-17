# import modules
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Create cython compilation
import pyximport
pyximport.install()
import DipoleRotation

# run example over zeta = 2*pi and d = 1-10 
theta = np.linspace(10.0*np.pi/180,350.0*np.pi/180,100)
R = np.linspace(1.0,3.0,100)
error = np.array([[DipoleRotation.example(zeta,d,3) \
                   for zeta in theta] for d in R])

# Plot the results
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

CS = plt.contourf(R/0.5, theta*180/np.pi, error)
plt.clabel(CS, inline=1, fontsize=10)

# make a colorbar for the contour lines
cbar = plt.colorbar(CS, shrink=0.8, extend='both')
cbar.ax.set_ylabel('Percent Error Compared to Exact Solution')

plt.xlabel(r'$\displaystyle\frac{R}{L}$')
plt.ylabel(r'$\displaystyle\theta$ (degrees)',fontsize=16)
plt.title(r'Percent Error Compared to Exact')

plt.show()
