import numpy as np
import scipy.optimize as op
import pandas as pd
from itertools import chain
import sys
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm

np.set_printoptions(threshold=sys.maxsize)

# Filling out lists of calculated responses and out
Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']
xl = pd.ExcelFile('Data/Resp_z_0_6singleSource.xls')
Out = []

def fillOut(Out, sheet):
    df = xl.parse(sheet)
    Out.append(df.iloc[0, :])

fillOut(Out, Sheets[0])
fillOut(Out, Sheets[1])
fillOut(Out, Sheets[2])
fillOut(Out, Sheets[3])
fillOut(Out, Sheets[4])
fillOut(Out, Sheets[5])

Out = list(chain.from_iterable(Out))
Out = np.asarray(Out)

R0 = []
R1 = []
R2 = []
R3 = []
R4 = []
R5 = []
R6 = []
xl = pd.ExcelFile('Data/Resp_z_0_6.xls')

Sheets = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6']
def fillResponse(Rx, sheet):
    df = xl.parse(sheet)
    for x in range(2, 39, 3):
        Rx.append(df.iloc[:, x])

fillResponse(R0, Sheets[0])
fillResponse(R1, Sheets[1])
fillResponse(R2, Sheets[2])
fillResponse(R3, Sheets[3])
fillResponse(R4, Sheets[4])
fillResponse(R5, Sheets[5])
fillResponse(R6, Sheets[6])


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
            for x in range(0,13):
                del R0[x][index]
                del R1[x][index]
                del R2[x][index]
                del R3[x][index]
                del R4[x][index]
                del R5[x][index]
                del R6[x][index]

        index += 1


R0 = np.transpose(R0)
R1 = np.transpose(R1)
R2 = np.transpose(R2)
R3 = np.transpose(R3)
R4 = np.transpose(R4)
R5 = np.transpose(R5)
R6 = np.transpose(R6)

R0 = np.transpose(R0)
R1 = np.transpose(R1)
R2 = np.transpose(R2)
R3 = np.transpose(R3)
R4 = np.transpose(R4)
R5 = np.transpose(R5)
R6 = np.transpose(R6)

Row1 = np.stack((R6, R5, R4, R3, R2))
Zero = np.zeros((13,80))

Row1 = np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero, Zero))
Row2 = np.hstack((Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero))
Row3 = np.hstack((Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero))
Row4 = np.hstack((Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero))
Row5 = np.hstack((Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero))
Row6 = np.hstack((Zero, Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6))

R = np.vstack((Row1, Row2, Row3, Row4, Row5, Row6))
print(R.shape)
## convert your array into a dataframe
df = pd.DataFrame (R)

## save to xlsx file

filepath = '../Data/R_cut.xlsx'

df.to_excel(filepath, index=False)

# Calculating In with a Bounded-Variable Least-Squares algorithm

In = op.lsq_linear(R, Out, (0, np.inf),
                       #method='bvls',
                       method='trf',
                       tol=1e-30,
                       max_iter=400,
                       verbose=0)['x']

activity = np.sum(In)
print("The activity using BLVS is {} Bq".format(activity))

In = np.array_split(In, 18)
In = np.asarray(In)

'''
# Creating a 12x12 matrix Z with the elements of In; needed for a square contour plot
def graph(var):
    global innerradius
    global outerradius
    indexIn = 0
    Z = np.zeros((N, N))
    for b in range(0, N):
        for a in range(0, N):
            if (b, a) in zero_pixels:
                Z[b][a] = 0
            else:
                Z[b][a] = In[var][indexIn]
                indexIn += 1

    # Plot a MxN square matrix plot
    plt.figure(figsize=(7, 5))
    matfig = plt.matshow(Z, fignum=1, origin='lower', norm=LogNorm(vmin=0.01, vmax=10E8))
    x = np.arange(0, W, W/N)  # the grid to which your data corresponds
    nx = x.shape[0]
    step_x = int(nx / (no_labels))  # step between consecutive labels
    x_positions = np.arange(0, nx, step_x)  # pixel count at label position
    x_labels = np.round(x[::step_x], 2)  # labels you want to see
    plt.xticks(x_positions, x_labels)
    y = np.arange(0, W, W/N)  # the grid to which your data corresponds
    ny = y.shape[0]
    step_y = int(ny / (no_labels))  # step between consecutive labels
    y_positions = np.arange(0, ny, step_y)  # pixel count at label position
    y_labels = np.round(y[::step_y], 2)  # labels you want to see
    plt.yticks(y_positions, y_labels)
    plt.xlabel('x-axis (cm)')
    plt.ylabel('y-axis (cm)')
    # plt.xticks(np.arange(0, M, step=M/4))
    # plt.yticks(np.arange(0, N, step=N/4))
    plt.gca().xaxis.tick_bottom()
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plt.colorbar(matfig, format=ticker.FuncFormatter(fmt))

    # plot the pipe
    innerradius = innerradius / W * N  # in px
    outerradius = outerradius / W * N  # in px
    center = ((N-1) / 2, (N - 1) / 2)  # in px
    circle1 = plt.Circle(center, innerradius, color='r', fill=False)
    circle2 = plt.Circle(center, outerradius, color='r', fill=False)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    #save figure
    # plt.savefig("fig")
    plt.show()

for x in range(0,18):
    #graph(x)
'''