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
xl = pd.ExcelFile('../Data/read (difulvio@illinois.edu).xls')

Sheets = ['z =0', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']


def fillOut(Out, sheet):
    df = xl.parse(sheet)
    Out.append(df.iloc[:, 0])


fillOut(Out, Sheets[0])
fillOut(Out, Sheets[1])
fillOut(Out, Sheets[2])
fillOut(Out, Sheets[3])
fillOut(Out, Sheets[4])
fillOut(Out, Sheets[5])

Out = list(chain.from_iterable(Out))
Out = np.asarray(Out)

R = []
xl = pd.ExcelFile('../Data/respMatr.xls')

Sheets = ['z 0', 'z 1', 'z 2', 'z 3', 'z 4', 'z 5']


def fillResponse(sheet):
    Rx = []
    df = xl.parse(sheet)
    for x in range(1, 36):
        Rx.append(df.iloc[x, :])
    return np.array(Rx).tolist()


R0 = fillResponse(Sheets[0])
R1 = fillResponse(Sheets[1])
R2 = fillResponse(Sheets[2])
R3 = fillResponse(Sheets[3])
R4 = fillResponse(Sheets[4])
R5 = fillResponse(Sheets[5])


# # M * N is the number of pixels
N = 12
no_labels = 5  # how many labels to see on axis x and y.
# M, N is divisible by no_labels.
W = 12  # 10 cm

# Info of the pipe
innerradius = 10.2 / 2  # in cm
outerradius = 11.4 / 2  # in cm

def deleteOutsiders(R):
    zero_pixels = []
    index = 0
    for y in range(0, N):
        for x in range(0, 12):
            x_p = -5.5 + (x) * 1
            y_p = -5.5 + (y) * 1
            if (x_p) ** 2 + (y_p) ** 2 > innerradius ** 2:
                zero_pixels.append((x, y))
                for k in range(0, 35):
                    del R[k][index]
            else:
                index += 1
    return R
def changeOutsiders(R):
    zero_pixels = []
    G = np.zeros((12, 12))
    for y in range(0, 12):
        y_p = y - 5.5
        for x in range(0, 12):
            x_p = x - 5.5
            if (x_p) ** 2 + (y_p) ** 2 <= innerradius ** 2:
                G[x, y] = 1
            else:
                # G[x,y] = 0
                zero_pixels.append((x, y))
    G = np.ravel(G)
    return np.multiply(R,G)

WantToGraph = False
if not WantToGraph:
    R0 = np.array(deleteOutsiders(R0))
    R1 = np.array(deleteOutsiders(R1))
    R2 = np.array(deleteOutsiders(R2))
    R3 = np.array(deleteOutsiders(R3))
    R4 = np.array(deleteOutsiders(R4))
    R5 = np.array(deleteOutsiders(R5))
    Zero = np.zeros((35, 80))

if WantToGraph:
    R0 = np.array(changeOutsiders(R0))
    R1 = np.array(changeOutsiders(R1))
    R2 = np.array(changeOutsiders(R2))
    R3 = np.array(changeOutsiders(R3))
    R4 = np.array(changeOutsiders(R4))
    R5 = np.array(changeOutsiders(R5))
    Zero = np.zeros((35, 144))

R = np.vstack((np.hstack((R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero, Zero, Zero, Zero, Zero)),
               np.hstack((Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero, Zero, Zero, Zero)),
               np.hstack((Zero, Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero, Zero, Zero)),
               np.hstack((Zero, Zero, Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero, Zero)),
               np.hstack((Zero, Zero, Zero, Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero)),
               np.hstack((Zero, Zero, Zero, Zero, Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5))))

if WantToGraph:
    fig, ax = plt.subplots(1,35)

    dets = np.array([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34])

    for j, detj in enumerate(dets):
        matfig = ax[j].imshow((R0[detj]).reshape(12,12), extent=[-6,6,-6,6], origin='lower')
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax[i][j])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(matfig, cax=cax)

        # plot the pipe
        innerradius = innerradius / 12 * 12  # in px
        outerradius = outerradius / 12 * 12 # in px
        center = (0, 0)  # in px
        circle1 = plt.Circle(center, innerradius, color='r', fill=False)
        circle2 = plt.Circle(center, outerradius, color='r', fill=False)
        ax[j].add_artist(circle1)
        ax[j].add_artist(circle2)

        # labels
        ax[j].set_xlabel('x (cm)')
        ax[j].set_title('Det {}'.format(detj))
    ax[0].set_ylabel('y (cm)')
    plt.show()
    # plt.savefig("matfig")

print(R.shape)
print(Out.shape)
# Calculating In with a Bounded-Variable Least-Squares algorithm
In = op.lsq_linear(R, Out, (0, np.inf),
                   method='bvls',
                   # method='trf',
                   tol=1e-30,
                   max_iter=400,
                   verbose=0)['x']

activity = np.sum(In)
print("The activity using BLVS is {} Bq".format(activity))

if WantToGraph:
    In = np.array_split(In, 16)
    In = np.asarray(In)

    # plt.imshow(In[4].reshape(12,12))
    fig,ax = plt.subplots(4,4,figsize=(10,10))
    for i in range(4):
        for j in range(4):
            n = 3*i + j
            matfig = ax[i][j].imshow(In[n].reshape(12,12), extent=[-6,6,-6,6], origin='lower')
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax[i][j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(matfig, cax=cax)

            # plot the pipe
            center = (0, 0)  # in px
            circle1 = plt.Circle(center, innerradius, color='r', fill=False)
            circle2 = plt.Circle(center, outerradius, color='r', fill=False)
            ax[i][j].add_artist(circle1)
            ax[i][j].add_artist(circle2)

            # labels
            ax[i][j].set_xlabel('x (cm)')
            ax[i][j].set_ylabel('y (cm)')
            ax[i][j].set_title('z = {} cm'.format(n-5))

            fig.tight_layout()

    plt.show()