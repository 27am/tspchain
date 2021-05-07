# COMMENTS: Concorde crashes [Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)] with coordinates that are higher than 1 billion

# https://github.com/jvkersch/pyconcorde
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
import matplotlib.pyplot as plt
import hashlib
import numpy as np
from tqdm import tqdm

BIT_OF_GRID = 32
fname = get_dataset_path("berlin52")

def scalePoints(coordList):
    """ changes the coordinate of the given points so that they fit the huge space determined by the hash function"""

    middlePointX = (max(coordList) - min(coordList)) / 2
    #print(f"Middle point of the list is {middlePointX}")
    middleGrid = np.power(2, BIT_OF_GRID) / 2
    #print(f"Middle point of the grid is {middleGrid}")

    #magnification = middleGrid / middlePointX
    magnification = 100000
    print(f"Magnification is {magnification}")

    result = np.array([(coordinate * magnification) for coordinate in coordList])

    return result

def scaleToBillion(listX, listY):
    maxX = max(listX)

    maxY = max(listY)
    magnifyX = np.round(100000000 / maxX)
    magnifyY = np.round(100000000 / maxY)

    genMax = max(maxX, maxY)

    magnify = np.round(100000000 / genMax)
    print(f"I am scaling all points by a factor of {magnify}")

    scaledX = [(xcoo * magnifyX) for xcoo in listX]
    print(f"X points scaled are: {scaledX} and the max now is {max(scaledX)}")

    scaledY = [(ycoo * magnifyY) for ycoo in listY]
    print(f"Y points scaled are: {scaledY} and the max now is {max(scaledY)}")

    return scaledX, scaledY


def extractCoordinates(inputString):
    digest = hashlib.sha256(inputString.encode("UTF-8")).digest()
    Xcoo = int.from_bytes(digest[:4], byteorder="little")
    Ycoo = int.from_bytes(digest[4:8], byteorder="little")
    print(Xcoo, Ycoo)

    return Xcoo, Ycoo





#print(fname)

X = []
Y = []
numbers = "0123456789"



with open(fname) as file:
    for line in file.readlines():
        if line[0] in numbers:
            # print(line.split())
            X.append(float(line.split()[1]))
            Y.append(float(line.split()[2]))

#Xscaled = scalePoints(X)
#print(f"X points scaled are: {Xscaled}")
#print(f"max X is {max(Xscaled)}")

#Yscaled = scalePoints(Y)
#print(f"Y points scaled are: {Yscaled}")
#print(f"max Y is {max(Yscaled)}")
Xscaled, Yscaled = scaleToBillion(X, Y)


#print(f"X coordinates of cities lie between {min(X)} and {max(X)}")
#print(f"Y coordinates of cities lie between {min(Y)} and {max(Y)}")

#X1, Y1 = extractCoordinates("famigliaaaaa")
extractedPoint = np.random.randint(100000000, size=2)
X1 = extractedPoint[0]
Y1 = extractedPoint[1]


fig = plt.figure(1, figsize=(18, 12))
plt.scatter(x=Xscaled, y=Yscaled, cmap="red")
plt.scatter(x=X1, y=Y1, marker="X")
plt.xlim(0, 100001000) # 100001000 to have some little margin
plt.ylim(0, 100001000)
plt.xlabel('Coordinate X', fontsize=24)
plt.ylabel('Coordinate Y', fontsize=24)
plt.title("Traveling Salesman Problem Map",fontsize=32)
plt.show()


solver = TSPSolver.from_tspfile(fname)
solver = TSPSolver.from_data(xs=Xscaled, ys=Yscaled, norm="EUC_2D", name="ttx")
solution = solver.solve()
print(solution.optimal_value)
optVal = solution.optimal_value


calculatedOptimalPaths = []
N_MAX_STEPS = 10000
EPSILON = 1e-7
DIFFICULTY = np.round(1/EPSILON)
N_SAMPLES_REQ = 2

nSamplesObtained = 0
sampleList = []
while nSamplesObtained < N_SAMPLES_REQ:
    for i in tqdm(range(N_MAX_STEPS)):
        extractedPoint = np.random.randint(100000000, size=2)
        X1 = extractedPoint[0]
        Y1 = extractedPoint[1]

        XscaledPlusOne = Xscaled.copy()
        XscaledPlusOne.append(X1)

        YscaledPlusOne = Yscaled.copy()
        YscaledPlusOne.append(Y1)

        solver = TSPSolver.from_data(xs=XscaledPlusOne, ys=YscaledPlusOne, norm="EUC_2D")
        solution = solver.solve()
        #print(solution.optimal_value)
        foundOpt = solution.optimal_value
        calculatedOptimalPaths.append(foundOpt)

        if (foundOpt/optVal) < 1 + EPSILON:
            print(f"Found PIN close enough! after {i} tries!")
            sampleList.append(i)
            nSamplesObtained += 1
            break

print(f"samples obtained {sampleList}")

print(optVal)
print(calculatedOptimalPaths)

overhead = [(val/optVal) for val in calculatedOptimalPaths]
print(overhead)
print(f"min overhead after {N_MAX_STEPS} is {min(overhead)}")