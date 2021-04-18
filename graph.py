import math
import scipy
from scipy import linalg
import numpy as np
import re

from pylab import meshgrid
from matplotlib import cm
import matplotlib.pyplot as plt

def NetToReal(value):
    teta = (value-0)/1
    return -1+teta*2

filename = "graphs\\n10_10r80"
n = 161

f = open(filename, "r")
line = f.readline()
num = line[line.find(" ")+1:]
num = int(num)

X = list()
Y = list()
Z = list()

for i in range(num):
    line = f.readline()
    match = re.search(r'{((?:[\d\,\.\-]+\s?)+)} => {((?:[\d\,\.\-]+\s?)+)}', line)
    X.append(float(match.group(1).split(' ')[0]))
    Y.append(float(match.group(1).split(' ')[1]))
    Z.append(float(match.group(2).split(' ')[0]))

X = np.array(X).reshape((n,n))
Y = np.array(Y).reshape((n,n))
Z = np.array(Z).reshape((n,n))
Z = NetToReal(Z)

X = X[1:161:4, 1:161:4]
Y = Y[1:161:4, 1:161:4]
Z = Z[1:161:4, 1:161:4]

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')

color = lambda Z: cm.jet((Z-np.amin(Z))/(np.amax(Z)-np.amin(Z)))

p1 = ax1.plot_surface(X, Y, Z,
                 rstride = 1,
                 cstride = 1,
                 cmap=cm.jet)
fig1.colorbar(p1)

plt.xlabel('X')
plt.ylabel('Y')

plt.show()