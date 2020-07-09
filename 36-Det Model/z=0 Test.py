import matplotlib
import numpy as np
import scipy.optimize as op
import pandas as pd
from itertools import chain
import sys
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(threshold=sys.maxsize)

Out = []
xl = pd.ExcelFile('../Data/pointSource.xls')

Sheets = ['z =0', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']
def fillOut(Out, sheet):
    df = xl.parse(sheet)
    Out.append(df.iloc[:, 0])

fillOut(Out, Sheets[0])
Out = np.array(Out[0]).tolist()

R = []
xl = pd.ExcelFile('../Data/respMatr.xls')

Sheets = ['z 0', 'z 1', 'z 2', 'z 3', 'z 4', 'z 5']
def fillResponse(Rx, sheet):
    df = xl.parse(sheet)
    for x in range(1, 36):
        Rx.append(df.iloc[x, :])
fillResponse(R, Sheets[0])
R = np.array(R).tolist()

# # M * N is the number of pixels
N = 12
no_labels = 5  # how many labels to see on axis x and y.
                # M, N is divisible by no_labels.
W = 12 # 10 cm

# Info of the pipe
innerradius = 10.2 / 2  # in cm
outerradius = 11.4 / 2  # in cm

zero_pixels = []
index = 0
for y in range(0, N):
    for x in range(0, 12):
        x_p = -5.5 + (x) * 1
        y_p = -5.5 + (y) * 1
        if (x_p) ** 2 + (y_p) ** 2 > innerradius ** 2:
            zero_pixels.append((x, y))
            for k in range(0,35):
                del R[k][index]
        else:
            index += 1

fig,ax = plt.subplots(6,3,figsize=(10,10))

def SolveAndGraph(R, Out, x, y):
    # Calculating In with a Bounded-Variable Least-Squares algorithm
    In = op.lsq_linear(R, Out, (0, np.inf),
                           method='bvls',
                           #method='trf',
                           tol=1e-30,
                           max_iter=400,
                           verbose=0)['x']

    activity = np.sum(In)
    print("The activity using BLVS is {} Bq".format(activity))

    indexIn = 0
    Z = np.zeros((N, N))
    for b in range(0, N):
        for a in range(0, N):
            if (b, a) in zero_pixels:
                Z[b][a] = 0
            else:
                Z[b][a] = In[indexIn]
                indexIn += 1

    print(In)

    matfig = ax[x][y].imshow(Z, extent=[-6,6,-6,6], origin='lower')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax[x][y])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matfig, cax=cax)

    # plot the pipe
    center = (0, 0)  # in px
    circle1 = plt.Circle(center, innerradius, color='r', fill=False)
    circle2 = plt.Circle(center, outerradius, color='r', fill=False)
    ax[x][y].add_artist(circle1)
    ax[x][y].add_artist(circle2)

    # labels
    ax[x][y].set_xlabel('x (cm)')
    ax[x][y].set_ylabel('y (cm)')
    ax[x][y].set_title('z = {} cm'.format(0))

    fig.tight_layout()


counterClockwise = []
for x in range(1, 18):
    counterClockwise.append(x)

clockwise = []
for x in range(18, 35):
    clockwise.append(x)

CurrentResponses = [R[0]]
CurrentOut = [Out[0]]

print(counterClockwise)

k = 0
for i in range(6):
    for j in range(3):
        if i==0 and j==0:
            SolveAndGraph(CurrentResponses, CurrentOut, i, j)
            continue
        if i==6 and j==3:
            continue
        CurrentResponses.append(R[counterClockwise[k]])
        CurrentResponses.append(R[clockwise[k]])

        CurrentOut.append(Out[counterClockwise[k]])
        CurrentOut.append(Out[clockwise[k]])

        SolveAndGraph(CurrentResponses, CurrentOut, i, j)
        print(k)
        k += 1

plt.show()
