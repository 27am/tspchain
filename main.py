# https://github.com/jvkersch/pyconcorde
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
import binascii
import numpy as np

BIT_OF_GRID = 32


def scalePoints(coordList):
    """ changes the coordinate of the given points so that they fit the huge space determined by the hash function"""

    middlePointX = (max(coordList) - min(coordList)) / 2
    #print(f"Middle point of the list is {middlePointX}")
    middleGrid = np.power(2, BIT_OF_GRID) / 2
    #print(f"Middle point of the grid is {middleGrid}")

    magnification = middleGrid / middlePointX
    print(f"Magnification is {magnification}")

    result = [np.round(coordinate * magnification) for coordinate in coordList]

    return result


def extractCoordinates(inputString):
    digest = hashlib.sha256(inputString.encode("UTF-8")).digest()
    Xcoo = int.from_bytes(digest[:4], byteorder="little")
    Ycoo = int.from_bytes(digest[4:8], byteorder="little")
    print(Xcoo, Ycoo)

    return Xcoo, Ycoo




fname = get_dataset_path("berlin52")
#print(fname)

X = []
Y = []
numbers = "0123456789"

X1, Y1 = extractCoordinates("papa come stai")

with open(fname) as file:
    for line in file.readlines():
        if line[0] in numbers:
            # print(line.split())
            X.append(float(line.split()[1]))
            Y.append(float(line.split()[2]))

Xscaled = scalePoints(X)
print(Xscaled)

Yscaled = scalePoints(Y)
print(Yscaled)

#print(f"X coordinates of cities lie between {min(X)} and {max(X)}")
#print(f"Y coordinates of cities lie between {min(Y)} and {max(Y)}")

plt.scatter(x=Xscaled, y=Yscaled, cmap="red")
plt.scatter(x=X1, y=Y1)
plt.show()

# solver = TSPSolver.from_tspfile(fname)
solver = TSPSolver.from_data(xs=Xscaled, ys=Yscaled, norm="EUC_2D")
solution = solver.solve()
