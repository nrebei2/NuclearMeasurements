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

Out = [0.00000219, 0.0000028, 0.00000286]

R = []
xl = pd.ExcelFile('C:\\Users\\rebei\PycharmProjects\Measurements\Data/Resp_z_0_6peak.xls')

Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6', 'Sheet7']
def fillResponse(Rx, sheet):
    df = xl.parse(sheet)
    Rx.append(df.iloc[:, 2])
    Rx.append(df.iloc[:, 20])
    Rx.append(df.iloc[:, 38])

fillResponse(R, Sheets[0])
print(np.array(R).tolist())

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
            for k in range(0,3):
                del R[k][index]
        index += 1

print(R)
print(np.array(R))

## convert your array into a dataframe
df = pd.DataFrame (R)
## save to xlsx file
#filepath = '../Data/3DetR_cut.xlsx'
#df.to_excel(filepath, index=False)

# Calculating In with a Bounded-Variable Least-Squares algorithm
In = op.lsq_linear(R, Out, (0, np.inf),
                       #method='bvls',
                       method='trf',
                       tol=1e-30,
                       max_iter=400,
                       verbose=2)['x']

activity = np.sum(In)
#print("The activity using BLVS is {} Bq".format(activity))

#print(abs(Out - np.matmul(R, In))/Out)
#print(Out)
#print(np.matmul(R, In))

indexIn = 0
Z = np.zeros((N, N))
for b in range(0, N):
    for a in range(0, N):
        if (b, a) in zero_pixels:
            Z[b][a] = 0
        else:
            Z[b][a] = In[indexIn]
            indexIn += 1


plt.imshow(Z, extent=[-6,6,-6,6], origin='lower')
# plot the pipe
center = (0, 0)  # in px
plt.Circle(center, innerradius, color='r', fill=False)
plt.Circle(center, outerradius, color='r', fill=False)
plt.show()
