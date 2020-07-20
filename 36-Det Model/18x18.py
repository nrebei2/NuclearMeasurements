import numpy as np
import scipy.optimize as op
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

np.set_printoptions(threshold=sys.maxsize)

Out = []
xl = pd.ExcelFile('../Data/PipeSource.xlsx')

def fillOut(Out, sheet):
    df = xl.parse(sheet)
    Out.extend(list(np.asarray(df.iloc[:, 0])))
    # remove last element (det 36)
    # del Out[-1]

#Sheets = ["17.5", "18.5", "19.5", "newpos1", "newpos2", "corner1", "newpos3", "newpos4", "25o5", "25o6", "25o7", "25o8", "newpos5", "newpos6", "corner2", "newpos7", "newpos8", "31.0"]
Sheets = ["19.5", "newpos1", "newpos2", "corner1", "newpos3", "newpos4", "25o5"]

for i in range(len(Sheets)):
    fillOut(Out, Sheets[i])


Out = np.asarray(Out)


# why does dividing this introduce noise? Fixed, had to start minimization with zero
max = max(Out)
Out = Out/np.max(Out)


R = []
xl = pd.ExcelFile('../Data/NewRespMatr.xlsx')

RSheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6', 'Sheet7', 'Sheet8', 'Sheet9', 'Sheet10']

def fillResponse(sheet):
    Rx = []
    df = xl.parse(sheet)
    # range(36) for all detectors
    for x in range(36):
        Rx.append(df.iloc[x, :])
    return np.array(Rx).tolist()

R0 = fillResponse(RSheets[0])
R1 = fillResponse(RSheets[1])
R2 = fillResponse(RSheets[2])
R3 = fillResponse(RSheets[3])
R4 = fillResponse(RSheets[4])
R5 = fillResponse(RSheets[5])
R6 = fillResponse(RSheets[6])
R7 = fillResponse(RSheets[7])
R8 = fillResponse(RSheets[8])
R9 = fillResponse(RSheets[9])

plt.show()

# # M * N is the number of pixels
N = 12
no_labels = 5  # how many labels to see on axis x and y.
# M, N is divisible by no_labels.
W = 12  # 10 cm

innerradius = 4.75
outerradius = 5.25

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
    return np.multiply(R, G)

R0 = np.array(changeOutsiders(R0))
R1 = np.array(changeOutsiders(R1))
R2 = np.array(changeOutsiders(R2))
R3 = np.array(changeOutsiders(R3))
R4 = np.array(changeOutsiders(R4))
R5 = np.array(changeOutsiders(R5))
R6 = np.array(changeOutsiders(R6))
R7 = np.array(changeOutsiders(R7))
R8 = np.array(changeOutsiders(R8))
R9 = np.array(changeOutsiders(R9))
Zero = np.zeros((R0.shape[0], 144))


R = np.vstack((np.hstack((R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #17.5
                   np.hstack((R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #18.5
                   np.hstack((R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero)), #19.5
                   np.hstack((R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero)), #newpos1
                   np.hstack((R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero)), #newpos2
                   np.hstack((R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero)), #corner1
                   np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero)), #newpos3
                   np.hstack((R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero)), #newpos4
                   np.hstack((R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9)),  # 25o5
                   np.hstack((R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8)),  # 25o6
                   np.hstack((Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7)),  # 25o7
                   np.hstack((Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6)),  # 25o8
                   np.hstack((Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5)),  # newpos5
                   np.hstack((Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4)),  # newpos6
                   np.hstack((Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3)),  # corner2
                   np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2)),  # newpos7
                   np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1)),  # newpos8
                   np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0))))  # 31.0

# R = np.vstack((np.hstack((R0, R1, R2, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #17.5
#                np.hstack((R1, R0, R1, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #18.5
#                np.hstack((R2, R1, R0, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #19.5
#                np.hstack((Zero, Zero, R1, R0, R1, R2, R3, R4, R5, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #newpos1
#                np.hstack((Zero, Zero, R2, R1, R0, R1, R2, R3, R4, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #newpos2
#                np.hstack((Zero, Zero, R3, R2, R1, R0, R1, R2, R3, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #corner1
#                np.hstack((Zero, Zero, R4, R3, R2, R1, R0, R1, R2, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #newpos3
#                np.hstack((Zero, Zero, R5, R4, R3, R2, R1, R0, R1, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #newpos4
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R0, R1, R2, R3, Zero, Zero, Zero, Zero, Zero, Zero)),  # 25o5
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R1, R0, R1, R2, Zero, Zero, Zero, Zero, Zero, Zero)),  # 25o6
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R2, R1, R0, R1, Zero, Zero, Zero, Zero, Zero, Zero)),  # 25o7
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R3, R2, R1, R0, Zero, Zero, Zero, Zero, Zero, Zero)),  # 25o8
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R1, R0, R1, R2, R3, R4, R5)),  # newpos5
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R2, R1, R0, R1, R2, R3, R4)),  # newpos6
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R3, R2, R1, R0, R1, R2, R3)),  # corner2
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R4, R3, R2, R1, R0, R1, R2)),  # newpos7
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R5, R4, R3, R2, R1, R0, R1)),  # newpos8
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R0))))  # 31.0

R = np.vstack((np.hstack((R0, R1, R2, R3, Zero, Zero, Zero)), #19.5
                np.hstack((R1, R0, R2, R3, Zero, Zero, Zero)), #
                np.hstack((R2, R1, R0, R2, Zero, Zero, Zero)),
                np.hstack((R3, R2, R1, R0, R1, R2, R3)),
                np.hstack((Zero, Zero, Zero, R1, R0, R1, R2)),
                np.hstack((Zero, Zero, Zero, R2, R1, R0, R1)),
                np.hstack((Zero, Zero, Zero, R3, R2, R1, R0))))

# R = np.vstack((np.hstack((R0, R1, R2, R3, R4, R5, R6, R7, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #17.5
#                np.hstack((R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #18.5
#                np.hstack((R2, R1, R0, R1, R2, R3, R4, R5, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)), #19.5
#                np.hstack((R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero)), #newpos1
#                np.hstack((R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero)), #newpos2
#                np.hstack((R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero)), #corner1
#                np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero)), #newpos3
#                np.hstack((R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero)), #newpos4
#                np.hstack((Zero, Zero, Zero, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, Zero)),  # 25o5
#                np.hstack((Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, Zero)),  # 25o6
#                np.hstack((Zero, Zero, Zero, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero)),  # 25o7
#                np.hstack((Zero, Zero, Zero, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, Zero)),  # 25o8
#                np.hstack((Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5)),  # newpos5
#                np.hstack((Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4)),  # newpos6
#                np.hstack((Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3)),  # corner2
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2)),  # newpos7
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1)),  # newpos8
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R5, R4, R3, R2, R1, R0))))  # 31.0

# R2 = np.vstack((np.hstack((R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero)),  # 17.5
#                np.hstack((R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero, Zero)),  # 18.5
#                np.hstack((R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero, Zero)),  # 19.5
#                np.hstack((R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero, Zero)),  # newpos1
#                np.hstack((R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero, Zero)),  # newpos2
#                np.hstack((R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero, Zero)),  # corner1
#                np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero, Zero)),  # newpos3
#                np.hstack((R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, Zero)),  # newpos4
#                np.hstack((R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9)),  # 25o5
#                np.hstack((R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7, R8)),  # 25o6
#                np.hstack((Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, R7)),  # 25o7
#                np.hstack((Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6)),  # 25o8
#                np.hstack((Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5)),  # newpos5
#                np.hstack((Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4)),  # newpos6
#                np.hstack((Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3)),  # corner2
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1, R2)),  # newpos7
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0, R1)),  # newpos8
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0))))  # 31.0

# R = np.vstack((np.hstack((R0, R1, R2, R4, Zero, Zero, Zero, Zero, Zero, Zero)),     # 17.5
#                np.hstack((R1, R0, R1, R3, Zero, Zero, Zero, Zero, Zero, Zero)),     # 18.5
#                np.hstack((R2, R1, R0, R2, Zero, Zero, Zero, Zero, Zero, Zero)),     # 19.5
#                np.hstack((Zero, Zero, R2, R0, R2, Zero, Zero, Zero, Zero, Zero)),   # corner1
#                np.hstack((Zero, Zero, Zero, R2, R0, R1, R2, R3, R5, Zero)),         # 25o5
#                np.hstack((Zero, Zero, Zero, R3, R1, R0, R1, R2, R4, Zero)),         # 25o6
#                np.hstack((Zero, Zero, Zero, R4, R2, R1, R0, R1, R3, Zero)),         # 25o7
#                np.hstack((Zero, Zero, Zero, R5, R3, R2, R1, R0, R2, Zero)),         # 25o8
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, R2, R0, R2)),   # corner2
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R2, R0))))# 31.0

# R = np.vstack((np.hstack((R0, R1, R2, R7, Zero, Zero, Zero, Zero, Zero, Zero)),  # 17.5
#                np.hstack((R1, R0, R1, R6, Zero, Zero, Zero, Zero, Zero, Zero)),  # 18.5
#                np.hstack((R2, R1, R0, R5, Zero, Zero, Zero, Zero, Zero, Zero)),  # 19.5
#                np.hstack((R7, R6, R5, R0, R5, R6, R7, R8, R9, Zero)),  # corner1
#                np.hstack((Zero, Zero, Zero, R5, R0, R1, R2, R3, R4, Zero)),  # 25o5
#                np.hstack((Zero, Zero, Zero, R6, R1, R0, R1, R2, R3, Zero)),  # 25o6
#                np.hstack((Zero, Zero, Zero, R7, R2, R1, R0, R1, R2, Zero)),  # 25o7
#                np.hstack((Zero, Zero, Zero, R8, R3, R2, R1, R0, R1, Zero)),  # 25o8
#                np.hstack((Zero, Zero, Zero, R9, R8, R7, R6, R5, R0, R5)),  # corner2
#                np.hstack((Zero, Zero, Zero, Zero, Zero, Zero, Zero, Zero, R5, R0))))  # 31.0

# R = np.vstack((np.hstack((R0, R1, R2, R7, R8, R9, Zero, Zero, Zero, Zero)),  # 17.5
#                np.hstack((R1, R0, R1, R6, R7, R8, R9, Zero, Zero, Zero)),  # 18.5
#                np.hstack((R2, R1, R0, R5, R6, R7, R8, R9, Zero, Zero)),  # 19.5
#                np.hstack((R7, R6, R5, R0, R5, R6, R7, R8, R9, Zero)),  # corner1
#                np.hstack((R8, R7, R6, R5, R0, R1, R2, R3, R8, R9)),  # 25o5
#                np.hstack((R9, R8, R7, R6, R1, R0, R1, R2, R7, R8)),  # 25o6
#                np.hstack((Zero, R9, R8, R7, R2, R1, R0, R1, R6, R7)),  # 25o7
#                np.hstack((Zero, Zero, R9, R8, R3, R2, R1, R0, R5, R6)),  # 25o8
#                np.hstack((Zero, Zero, Zero, R9, R8, R7, R6, R5, R0, R5)),  # corner2
#                np.hstack((Zero, Zero, Zero, Zero, R9, R8, R7, R6, R5, R0))))  # 31.0

# R = np.vstack((np.hstack((R0, R1, R2, R3, R4, R5, R6, R7, R8, R9)),   # 17.5
#                np.hstack((R1, R0, R1, R2, R3, R4, R5, R6, R7, R8)),   # 18.5
#                np.hstack((R2, R1, R0, R1, R2, R3, R4, R5, R6, R7)),   # 19.5
#                np.hstack((R3, R2, R1, R0, R1, R2, R3, R4, R5, R6)),   # corner1
#                np.hstack((R4, R3, R2, R1, R0, R1, R2, R3, R4, R5)),   # 25o5
#                np.hstack((R5, R4, R3, R2, R1, R0, R1, R2, R3, R4)),   # 25o6
#                np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3)),   # 25o7
#                np.hstack((R7, R6, R5, R4, R3, R2, R1, R0, R1, R2)),   # 25o8
#                np.hstack((R8, R7, R6, R5, R4, R3, R2, R1, R0, R1)),   # corner2
#                np.hstack((R9, R8, R7, R6, R5, R4, R3, R2, R1, R0))))  # 31.0

# R = np.vstack((np.hstack((R0, R1, R2, R3)),
#                np.hstack((R1, R0, R1, R2)),
#                np.hstack((R2, R1, R0, R1)),
#                np.hstack((R3, R2, R1, R0))))

fig, ax = plt.subplots(6, 6, figsize=(6, 6))
dets = []
for i in range(101, 101 + R0.shape[0]):
    dets.append(i)

k = 0
for j in range(6):
    for i in range(6):
        matfig = ax[i][j].imshow(R0[k].reshape(12, 12), extent=[-6, 6, -6, 6], origin='lower')
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax[i][j])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(matfig, cax=cax)

        # plot the pipe
        innerradius = innerradius / 12 * 12  # in px
        outerradius = outerradius / 12 * 12  # in px
        center = (0, 0)  # in px
        circle1 = plt.Circle(center, innerradius, color='r', fill=False)
        circle2 = plt.Circle(center, outerradius, color='r', fill=False)
        ax[i][j].add_artist(circle1)
        ax[i][j].add_artist(circle2)

        # labels
        ax[i][j].set_xlabel('x (cm)')
        ax[i][j].set_title('Det {}'.format(dets[k]))
        k += 1
        ax[i][0].set_ylabel('y (cm)')


# Why use svd to compute the step step-size of fista
u, s, vt = svds(R, 1, which='LM')  # svd
L = s[0] ** 2

# solving
# initialize
N = np.shape(R)[1]
x = z = np.random.rand(N) * 0

#L = 10

# ADAM and FISTA producing same results, all that matters is the stepsize and convergence criteria -> FISTA, since less order complexity
# Overfitting includes sparsing the solution w/ lambda (similar to l1 approximation?), which in addition under-approximates
# So, you have to find a balance between sparsing the data too little and too much (much is bad as the pixels which should have equal activity become further apart)
# Kinda difficult with only 3 sources, also hard to cross-validate given you have you actually observe the graphed solution and not just the calculated activity.

t = 1
lam = 10E-8  # tune this parameter to get better result
max_iter = 2000
results = []
err = np.zeros(max_iter)
for k in np.arange(0, max_iter):
    # print(k)
    xp = z - 1 / L * R.T @ (R @ z - Out)
    xp = np.maximum(xp - lam / L, 0).ravel()
    tp = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
    z = xp + (t - 1) / tp * (xp - x)
    # update
    t = tp
    x = xp
    err[k] = np.linalg.norm(Out - R.dot(x))
    if k % 10 == 0:
        results.append(x)
    # Stops loop when error difference is small -- Prevents overfitting??
    if np.abs(err[k] - err[k-1])/err[k] <= 10 ** -4.5: #or err[k] <= 10E-3 and k >= 1:
        break
In = results[-1]

fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
# ax1.set_title("Reconstruction\nFiltered back projection")
ax1.plot(np.arange(0, max_iter), err)
# ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('$||Ax-y||$')

In = In*max
print("The activity using FISTA is {} Bq".format(sum(In.ravel())))

In = np.array_split(In, np.shape(R)[1]/144)
In = np.asarray(In)

startingz = currentz = 0
x = 2
y = 4

fig, ax = plt.subplots(x, y, figsize=(10, 10))
for i in range(x):
    for j in range(y):
        if i == 1 and j == 3:
            break
        matfig = ax[i][j].imshow(In[currentz + startingz].reshape(12, 12), extent=[-6, 6, -6, 6],
                                 origin='lower', vmin=0, vmax=np.amax(In))
        # plot the pipe
        center = (0, 0)  # in px
        circle1 = plt.Circle(center, innerradius, color='r', fill=False)
        circle2 = plt.Circle(center, outerradius, color='r', fill=False)
        ax[i][j].add_artist(circle1)
        ax[i][j].add_artist(circle2)

        # labels
        ax[i][j].set_xlabel('x (cm)')
        ax[i][j].set_ylabel('y (cm)')
        ax[i][j].set_title(Sheets[currentz])


        fig.tight_layout()
        currentz += 1

fig.subplots_adjust(right=0.8, top=0.93)
# put colorbar at desire position
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(matfig, cax=cbar_ax)

plt.show()