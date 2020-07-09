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
matplotlib.rcParams.update({'font.size': 12})
num = int(input("Which z to graph? (0, 1, 2, 3, 4, 5, 6, 7, 8, 9): "))
det = int(input("Which detector to graph? (1, 2, 3, ..., 34, 35, 36): "))
Source = input("What source would you like to use? (pointSource, source1, source2, or pipeSource): ")

xl = pd.ExcelFile('../Data/NewRespMatr.xlsx')

Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6', 'Sheet7', 'Sheet8', 'Sheet9', 'Sheet10']

# x = 0 by average of x=-0.5 and x=0.5
def fillResponse(sheet):
    Rx = []
    df = xl.parse(sheet)
    for x in range(36):
        bruh = []
        for y in range(5, 6 + 12 * 11, 12):
            bruh.append((df.iloc[x, y]+df.iloc[x, y+1])/2)
        Rx.append(bruh)
    return np.array(Rx).tolist()

# x = -0.5
# def fillResponse(sheet):
#     Rx = []
#     df = xl.parse(sheet)
#     for x in range(35):
#         bruh = []
#         for y in range(5, 6 + 12 * 11, 12):
#             bruh.append((df.iloc[x, y]))
#         Rx.append(bruh)
#     return np.array(Rx).tolist()


R0 = fillResponse(Sheets[0])
R1 = fillResponse(Sheets[1])
R2 = fillResponse(Sheets[2])
R3 = fillResponse(Sheets[3])
R4 = fillResponse(Sheets[4])
R5 = fillResponse(Sheets[5])
R6 = fillResponse(Sheets[6])
R7 = fillResponse(Sheets[7])
R8 = fillResponse(Sheets[8])
R9 = fillResponse(Sheets[9])

R = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9]
y = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

fig, ax = plt.subplots(1,figsize=(6,6))

i=0
for Rx in R:
    for j in range(1):
        #ax[i].scatter(y, Rx[j], label="det"+str(j))
        if i == num:
            ax.plot(y, Rx[j], label="det"+str(j+1))
            ax.scatter(y, Rx[j])

    i += 1

ax.set_xlabel('y-coordinate (cm)')
ax.set_title('z = {}'.format(num))
ax.set_ylabel('Responses')
ax.grid('on')
plt.xticks(np.arange(min(y), max(y) + 1, 1.0))
fig.legend()

fig, ax = plt.subplots(1,figsize=(6,6))

j = 0
for Rx in R:
    ax.plot(y, np.transpose(Rx[det-1]), label="z = "+str(j))
    ax.scatter(y, np.transpose(Rx[det-1]))
    j += 1

ax.set_xlabel('y-coordinate (cm)')
ax.set_title('Detector {}'.format(det))
ax.set_ylabel('Responses')
plt.xticks(np.arange(min(y), max(y) + 1, 1.0))
ax.grid('on')

fig.legend()
xl = pd.ExcelFile('../Data/'+Source+'.xls')
Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']
Out = []

def fillOut(Out, sheet):
    df = xl.parse(sheet)
    Out.append(df.iloc[:, 0])

if Source == "pipeSource":
    Sheets = ["17.5", "18.5", "19.5", "25o5", "25o6", "25o7", "25o8", "31.0"]

    for i in range(len(Sheets)):
        fillOut(Out, Sheets[i])

else:
    fillOut(Out, Sheets[0])
    fillOut(Out, Sheets[1])
    fillOut(Out, Sheets[2])
    fillOut(Out, Sheets[3])
    fillOut(Out, Sheets[4])
    fillOut(Out, Sheets[5])

fig, ax = plt.subplots(1,figsize=(6,6))

y = []
for x in range (1, 37):
    y.append(x)

if Source == "pipeSource":
    for i in range(8):
        ax.plot(y, Out[i], label=Sheets[i])
        ax.scatter(y, Out[i])

else:
    for i in range(6):
        ax.plot(y, Out[i], label="z = "+str(i))
        ax.scatter(y, Out[i])

ax.set_xlabel('Detector')
ax.set_title(Source + " counts")
ax.set_ylabel('Counts')
ax.grid('on')
plt.xticks(np.arange(1, 37, 1.0))
fig.legend()


plt.show()
