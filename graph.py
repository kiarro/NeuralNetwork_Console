import math
import scipy
from scipy import linalg
import numpy as np
import re

from pylab import meshgrid
from matplotlib import cm
import matplotlib.pyplot as plt

# filename = "H:\\Programs\\Neural Networks\\NeuralNetwork_Console\\graph\\rn100"
filename = "H:\\Programs\\Neural Networks\\NeuralNetwork_Console\\case\\case26"


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
    # key = list(map(float, match.group(1).split(' ')))
    X.append(float(match.group(1).split(' ')[0]))
    Y.append(float(match.group(1).split(' ')[1]))
    # value = list(map(float, match.group(2).split(' ')))
    Z.append(float(match.group(2).split(' ')[0]))

X = np.array(X).reshape((26,26))
Y = np.array(Y).reshape((26,26))
Z = np.array(Z).reshape((26,26))

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')

color = lambda Z: cm.jet((Z-np.amin(Z))/(np.amax(Z)-np.amin(Z)))

ax1.plot_surface(X, Y, Z,
                 rstride = 1,
                 cstride = 1)
plt.show()