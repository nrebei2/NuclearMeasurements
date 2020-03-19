import numpy as np
import scipy.optimize as op
import pandas as pd
from itertools import chain
import sys
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm


# In[ ]:


# # Info of the pipe
innerradius = 10.2 / 2  # in cm
outerradius = 11.4 / 2  # in cm

N = 12
# In[ ]:



#np.set_printoptons(threshold=sys.maxsize)

# Filling out lists of calculated responses and out
Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']
xl = pd.ExcelFile('../Data/Resp_z_0_6singleSource.xls')
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


# In[ ]:


# xl = pd.ExcelFile('/content/drive/My Drive/Holdup/data/Resp_z_0_6.xls')
# Sheets = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6']
xl = pd.ExcelFile('../Data/Resp_z_0_6.xls')
Sheets = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6']
def fillResponse(sheet):
    Rx=[]
    singleDec=[]
    df = xl.parse(sheet)
    for x in range(2, 39, 3):
        Rx.append(df.iloc[:, x])
    return np.array(Rx)

# responses to voxels outside the pipe should be 0
# create a boolean matrix
zero_pixels = []
G = np.zeros((12,12))
for y in range(0, 12):
    y_p = y - 5.5
    for x in range(0, 12):
        x_p = x - 5.5
        if (x_p)**2 + (y_p)**2 <= innerradius**2:
            G[x,y] = 1
        else:
            #G[x,y] = 0
            zero_pixels.append((x, y))
G = np.ravel(G)

R0 = fillResponse(Sheets[0])

R1 = fillResponse(Sheets[1])

R2 = fillResponse(Sheets[2])

R3 = fillResponse(Sheets[3])

R4 = fillResponse(Sheets[4])
R5 = fillResponse(Sheets[5])

R6 = fillResponse(Sheets[6])
R0 = np.multiply(R0,G)
R1 = np.multiply(R1,G)
R2 = np.multiply(R2,G)
R3 = np.multiply(R3,G)
R4 = np.multiply(R4,G)
R5 = np.multiply(R5,G)
R6 = np.multiply(R6,G)

Zero = np.zeros((13,144))

Row1 = np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero, Zero))
Row2 = np.hstack((Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero))
Row3 = np.hstack((Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero))
Row4 = np.hstack((Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero))
Row5 = np.hstack((Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero))
Row6 = np.hstack((Zero, Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6))
# Row1 = np.hstack((R2, R1, R0, R1, R2, Zero, Zero, Zero, Zero, Zero))
# Row2 = np.hstack((Zero, R2, R1, R0, R1, R2, Zero, Zero, Zero, Zero))
# Row3 = np.hstack((Zero, Zero, R2, R1, R0, R1, R2, Zero, Zero, Zero))
# Row4 = np.hstack((Zero, Zero, Zero, R2, R1, R0, R1, R2, Zero, Zero))
# Row5 = np.hstack((Zero, Zero, Zero, Zero, R2, R1, R0, R1, R2, Zero))
# Row6 = np.hstack((Zero, Zero, Zero, Zero, Zero, R2, R1, R0, R1, R2))

R = np.vstack((Row1, Row2, Row3, Row4, Row5, Row6))
print(R.shape)
# ## convert your array into a dataframe
df = pd.DataFrame (R)
# ## save to xlsx file

filepath = '../Data/R.xlsx'

df.to_excel(filepath, index=False)


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(2,5,figsize=(10,10))

dets=np.array([0,6,12])
for j, detj in enumerate(dets):
    for i in np.arange(2):
        if i == 0:
            matfig = ax[0][j].imshow((R0[detj]).reshape(12,12), extent=[-6,6,-6,6], origin='lower')
        else:
            matfig = ax[1][j].imshow((np.multiply(R0,G))[detj].reshape(12,12), extent=[-6,6,-6,6], origin='lower')
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
        ax[i][j].add_artist(circle1)
        ax[i][j].add_artist(circle2)

        # labels
        ax[i][j].set_xlabel('x (cm)')
        ax[i][j].set_ylabel('y (cm)')
        ax[i][j].set_title('Det {}'.format(detj))
# plt.savefig("matfig")


# ## Plot the response matrix

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
ig, ax = plt.subplots(1,1,figsize=(12,8))
imfig = ax.imshow(R,norm=colors.SymLogNorm(linthresh=0.0001, linscale=0.0001,
                                              vmin=0.0001, vmax=0.001))
ax.set_aspect('auto')
fig.colorbar(imfig, ax=ax, extend='both')


# # Check with 3 detectors
# - keep det0, 6, 12
# - measurement at z = 0 cm

# ## Sovle equation:
# \begin{equation}
#     \text{Out} = R \text{ In},\\
#     \text{Out}\in \mathbb{R}^{78}, R \in \mathbb{R}^{78\times2592}, \text{In}\in \mathbb{R}^{2592}
# \end{equation}

# In[ ]:


In = op.lsq_linear(R, Out, (0, np.inf),
                       #method='bvls',
                       method='trf',
                       tol=1e-30,
                       max_iter=400,
                       verbose=0)['x']
#In[In == 0.1] = 0
activity = np.sum(In)
#activity = activity * 10000000
print("The activity using BLVS is {} Bq".format(activity))
#print(abs(Out - np.matmul(R, In))/Out)
print(np.matmul(R, In))
print(Out)
# In = np.array_split(In, 10)
# In = np.asarray(In)


# In[ ]:


#In = np.matmul(np.linalg.pinv(R),Out)
#sum(In)


# In[ ]:





# In[ ]:

In = np.array_split(In, 18)
In = np.asarray(In)

# plt.imshow(In[4].reshape(12,12))
fig,ax = plt.subplots(6,3,figsize=(12,18))
for i in range(6):
    for j in range(3):
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
        ax[i][j].set_title('z = {} cm'.format(n-6))


df = pd.DataFrame(In)
# ## save to xlsx file

filepath = '../Data/In.xlsx'

df.to_excel(filepath, index=False)

plt.show()

# In[ ]:


# import numpy as np
# import scipy.optimize as op
# import pandas as pd
# from itertools import chain
# import sys
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# from matplotlib.colors import LogNorm
# #np.set_printoptons(threshold=sys.maxsize)

# # Filling out lists of calculated responses and out
# Sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']
# xl = pd.ExcelFile('Data/Resp_z_0_6singleSource.xls')
# Out = []

# def fillOut(Out, sheet):
#     df = xl.parse(sheet)
#     Out.append(df.iloc[0, :])

# fillOut(Out, Sheets[0])
# fillOut(Out, Sheets[1])
# fillOut(Out, Sheets[2])
# fillOut(Out, Sheets[3])
# fillOut(Out, Sheets[4])
# fillOut(Out, Sheets[5])

# Out = list(chain.from_iterable(Out))
# Out = np.asarray(Out)

# R0 = []
# R1 = []
# R2 = []
# R3 = []
# R4 = []
# R5 = []
# R6 = []
# xl = pd.ExcelFile('Data/Resp_z_0_6.xls')

# Sheets = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6']
# def fillResponse(Rx, sheet):
#     df = xl.parse(sheet)
#     for x in range(2, 39, 3):
#         Rx.append(df.iloc[:, x])

# fillResponse(R0, Sheets[0])
# fillResponse(R1, Sheets[1])
# fillResponse(R2, Sheets[2])
# fillResponse(R3, Sheets[3])
# fillResponse(R4, Sheets[4])
# fillResponse(R5, Sheets[5])
# fillResponse(R6, Sheets[6])


# # # M * N is the number of pixels
# N = 12
# no_labels = 5  # how many labels to see on axis x and y.
#                 # M, N is divisible by no_labels.
# W = 12 # 10 cm

# # Info of the pipe
# innerradius = 10.2 / 2  # in cm
# outerradius = 11.4 / 2  # in cm

# '''
# # Creating list of calculated responses for each pixel
# zero_pixels = []
# index = 0
# for y in range(0, N):
#     for x in range(0, 12):
#         x_p = -5.5 + (x) * 1
#         y_p = -5.5 + (y) * 1
#         if (x_p) ** 2 + (y_p) ** 2 > innerradius ** 2:
#             zero_pixels.append((x, y))
#             for x in range(0,13):
#                 del R0[x][index]
#                 del R1[x][index]
#                 del R2[x][index]
#                 del R3[x][index]
#                 del R4[x][index]
#                 del R5[x][index]
#                 del R6[x][index]

#         index += 1
# '''

# R0 = np.transpose(R0)
# R1 = np.transpose(R1)
# R2 = np.transpose(R2)
# R3 = np.transpose(R3)
# R4 = np.transpose(R4)
# R5 = np.transpose(R5)
# R6 = np.transpose(R6)

# R0 = np.transpose(R0)
# R1 = np.transpose(R1)
# R2 = np.transpose(R2)
# R3 = np.transpose(R3)
# R4 = np.transpose(R4)
# R5 = np.transpose(R5)
# R6 = np.transpose(R6)

# Zero = np.zeros((13,144))

# Row1 = np.hstack((R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero, Zero))
# Row2 = np.hstack((Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero, Zero))
# Row3 = np.hstack((Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero, Zero))
# Row4 = np.hstack((Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero, Zero))
# Row5 = np.hstack((Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6, Zero))
# Row6 = np.hstack((Zero, Zero, Zero, Zero, Zero, R6, R5, R4, R3, R2, R1, R0, R1, R2, R3, R4, R5, R6))

# R = np.vstack((Row1, Row2, Row3, Row4, Row5, Row6))
# print(R.shape)
# ## convert your array into a dataframe
# df = pd.DataFrame (R)

# ## save to xlsx file

# filepath = 'R.xlsx'

# df.to_excel(filepath, index=False)

# Calculating In with a Bounded-Variable Least-Squares algorithm
'''
In = op.lsq_linear(R, Out, (0, np.inf),
                       method='bvls',
                       #method='trf',
                       tol=1e-30,
                       max_iter=400,
                       verbose=0)['x']

activity = np.sum(In)
print("The activity using BLVS is {} Bq".format(activity))

In = np.array_split(In, 18)
In = np.asarray(In)


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


# In[ ]:




