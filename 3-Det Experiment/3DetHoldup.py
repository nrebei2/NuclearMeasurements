#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np


# In[6]:
# # Calibration

# In[ ]:
from scipy.sparse.linalg import svds


def get_spectrum(filepath, time):
    counts = []
    with open(filepath) as f:
        for line in f:
            counts.append(int(line.strip())/float(time))
    return np.array(counts)


# In[ ]:


raw_paths = ["../Data/CH_0@DT5730_1770_Espectrum_CalibrationCH1 FINAL_2_20191113_231257.txt",
             "../Data/CH_1@DT5730_1770_Espectrum_CalibrationCH1_20191101_233347.txt",
             "../Data/CH_2@DT5730_1770_Espectrum_CalibrationCH2_20191111_223737.txt"]
bk_paths = ["../Data/CH_0@DT5730_1770_Espectrum_CH0 Background_20191118_205027.txt",
            "../Data/CH_1@DT5730_1770_Espectrum_CH1 CH2 Background_3_20191114_214529.txt",
            "../Data/CH_2@DT5730_1770_Espectrum_CH1 CH2 Background_3_20191114_214529.txt"]
raw_spectra = np.array([get_spectrum(rawf, 5) for rawf in raw_paths])  # 5 hours
bk_spectra = np.array([get_spectrum(bkf, 5) for bkf in bk_paths])  # 5 hours


# In[ ]:


cali_coeffs = [661.7/431, 661.7/1644, 661.7/550]
cali_spectra = np.array([np.array([c * np.arange(0, len(raw)), raw - bk]).T for c, raw, bk in zip(cali_coeffs, raw_spectra, bk_spectra)])


# In[230]:


xlims = [[0,1000],[0,2000],[0,1000]]
ylims=[[0,1000],[0,400],[0,800]]
fig,ax = plt.subplots(3,2,figsize = (12,10))
for n in np.arange(3):
    ax[n][0].plot(raw_spectra[n], label='Raw')
    ax[n][0].plot(bk_spectra[n] , label='Background')
    ax[n][0].plot(raw_spectra[n]-bk_spectra[n], label='Backgound subtracted')
    ax[n][0].set_xlim(xlims[n])
    ax[n][0].set_ylim(ylims[n])
    ax[n][0].set_title('CH%s'%n)
    ax[n][0].set_xlabel('Channel number')
    ax[n][0].set_ylabel('Count rate ($h^{-1}$)')
    ax[n][0].legend()
    ax[n][1].plot(cali_spectra[n,:,0],cali_spectra[n,:,1], label='Calibrated')
    ax[n][1].set_title('CH%s'%n)
    ax[n][1].set_xlabel('Energy (keV)')
    ax[n][1].set_ylabel('Count rate ($h^{-1}$)')
    ax[n][1].set_xlim([0,1000])
    ax[n][1].set_ylim(ylims[n])
    ax[n][1].legend()


# # Intrinsic Efficiency

# ## Sum of counts under the peak

# In[ ]:


def peak_sum(spectrum, ROI=[600,723]):
    # spectrum is a 2d array with first element being the enrergy bin and 
    # second element being the counts in that bin
    # ROI = [low_energy_bound, upper_energy_bound], keV
    summ = 0
    low_index  = up_index = 0
    for i in np.arange(0, len(spectrum) - 1):
        if (spectrum[i, 0] - ROI[0]) * (spectrum[i+1, 0] - ROI[0]) < 0:
            low_index = i
        elif (spectrum[i, 0] - ROI[1]) * (spectrum[i+1, 0] - ROI[1]) < 0:
            up_index = i
    # sum of counts under the peak
    for i in np.arange(low_index, up_index+1):
        summ = summ + spectrum[i,1]
    # Compton continuum
    compton_sum = (spectrum[low_index, 1]+spectrum[up_index, 1])*(up_index - low_index) / 2
    summ = summ - compton_sum
    return summ, compton_sum, up_index, low_index


# In[ ]:


# sum of counts under the peak, after subtracting the Compton continuum
peak_sums=[]
for n in np.arange(3):
    summ,_,_,_ = peak_sum(cali_spectra[n])
    peak_sums.append(summ)
peak_sums = np.array(peak_sums)


# In[232]:


# print(peak_sums)


# In[4]:


# plot
fig,ax = plt.subplots(3,1,figsize = (12,10))
for n in np.arange(3):
    summ, compton_sum, up_index, low_index = peak_sum(cali_spectra[n], ROI=[600,723])
    ax[n].plot(cali_spectra[n,:,0],cali_spectra[n,:,1])
    # ax[n].plot(np.array([cali_spectra[n,low_index,0],cali_spectra[n,low_index,0],cali_spectra[n,up_index,0],cali_spectra[n,up_index,0]]),
    #            np.array([0,cali_spectra[n,low_index,1],cali_spectra[n,up_index,1],0]),'o-',label='Compton')
    xs=cali_spectra[n,low_index:up_index,0]
    ys=cali_spectra[n,low_index:up_index,1]
    ax[n].fill_between(xs,ys, alpha=0.4, label='Net = %.1f'%summ)
    ax[n].fill_between(np.array([cali_spectra[n,low_index,0],cali_spectra[n,low_index,0],cali_spectra[n,up_index,0],cali_spectra[n,up_index,0]]),
               np.array([0,cali_spectra[n,low_index,1],cali_spectra[n,up_index,1],0]),alpha=0.4,label='Compton=%.1f'%compton_sum)
    ax[n].set_title('CH%s'%n)
    ax[n].set_xlabel('Energy (keV)')
    ax[n].set_ylabel('Count rate ($h^{-1}$)')
    ax[n].set_xlim([0,1000])
    ax[n].set_ylim(ylims[n])
    ax[n].legend()


# ## Efficiency

# In[235]:


f = 0.1 # attenuation by the lead
area = ((0.4**2) * np.pi) + f * (((5.08 / 2)**2 * np.pi) -
                                    ((0.4**2) * np.pi)) #effective area, cm^2

r = np.sqrt(area / np.pi) # effective radius, cm
h = 2.54 * 2 # h is the length of the detector, cm
# source to detector center distance, cm
c = np.array([16.2 + 2.54 + 1.79, #CH0
     12.5 + 2.54 + 1.79, #CH1
     16.75 + 2.54 + 1.79]) #CH2
BR = 1
BF = 0.851
a = 35400 #Bq
t = 3600 #Aquisition time, seconds

d = np.sqrt(r**2 / 2 + h**2 / 12 + np.power(c,2))
omega = 0.5 * (1 - d / np.sqrt(np.power(d,2) + r**2))
# omega = area / (4*np.pi * np.power(c-1.79,2))
intrinsic_eff = peak_sums / (omega * BF * BR * t * a)

# print(c)
# print(omega)
print("Intrinsic efficiency:", intrinsic_eff)


# # Response Matrix

# In[ ]:


# N * N is the number of pixels
N = 12
no_labels = 5  # num of labels to put on axis x and y.
                # M, N is divisible by no_labels.
W = 12 # region of interest, cm; covers the cross section of the pipe

# Info of the pipe
innerradius = 10.2 / 2  # in cm
outerradius = 11.4 / 2  # in cm

# Coordinates of detector, [x, y], cm
coords = np.array([[0,-24.13], # Detector (2'x2' (S/N 60002-6146-1))
                   [23.33,-2.45], # Detector (2'x2' (S/N 3290))
                   [-24.03,-2.45]]) # Detector (2'x2' (S/N 00876))

miu_air = 1.043E-4  # cm^(-1) # Air attenuation coefficient (Lambda)
att_pipe = 1 / 1.10292  # attenuation by the pipe, const for all pixels
BR = 0.85  # branching ratio

# Source strength
A_0 = 1

# # Distance between two points
# def distance(x1, y1, x2, y2):
#     return np.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

# Response to one pixel
def response(A, area, eff_i, c, h):
    # A is the source activity
    # area is the detector surface area
    # eff_i  is the intrinsic efficiency
    # d is the distance between the detector center and pixel center

    # return A * ((a * e) / (4 * np.pi * (r**2))) * (np.e**(-l * r))
    # omega = area / (4 * np.pi * d**2)
    r = np.sqrt(area / np.pi)
    d = np.sqrt(r**2 / 2 + h**2 / 12 + c**2)  # h is the length of the detector
    omega = 0.5 * (1 - d / np.sqrt(d**2 + r**2))
    return A * BR * eff_i * omega * att_pipe


# Creating list of calculated responses for each pixel
respMatrix = []
zero_pixels = []
for y in range(0, N):
    for x in range(0, N):
        # x_p = x * 12 / N
        # y_p = y * 12 / N
        x_p = -W / 2+ (x+0.5) * W / N # coordinates of pixel center
        y_p = -W / 2+ (y+0.5) * W / N
        # print(x_p)
        # print(y_p)
        # detector center to pixel center
        c = np.array([np.linalg.norm(x) for x in coords-[x_p,y_p]])
        # print(c)
        # response to one pixel
        resp = response(A_0, area, intrinsic_eff, c, h)
        # print(resp)
        if (x_p)**2 + (y_p)**2 > innerradius**2:
            #zero_pixels.append((x, y))
            respMatrix.append([0, 0, 0]) ## for plot only
        else:
            respMatrix.append(resp)
            # c_sum_list.append(c_1 + c_2 + c_3)
respMatrix = np.array(respMatrix)
respMatrix = np.transpose(respMatrix)

# In[238]:

'''
# uncomment line 61 in the previous cell
from mpl_toolkits.axes_grid1 import make_axes_locatable

respMatrix=respMatrix.reshape(N,N,3)
fig,ax = plt.subplots(1,3,figsize = (12,10))
for i in np.arange(3):
    matfig = ax[i].matshow(respMatrix[:,:,i])
    ax[i].set_title("Detector {}, [{},{}]".format(i,coords[i,0],coords[i,1]))
    # fig.colorbar(matfig, ax=ax[i])

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(matfig, cax=cax)
fig.tight_layout()
'''

# # Out Matrix

# ## Spectra

# In[ ]:


raw_paths = ['../Data/CH_0@DT5730_1770_Espectrum_Pipe Measurement 3 Detectors 2 cs-137_20191018_150203.txt',
             '../Data/CH_1@DT5730_1770_Espectrum_Pipe Measurement 3 Detectors 2 cs-137_20191018_150207.txt',
             '../Data/CH_2@DT5730_1770_Espectrum_Pipe Measurement 3 Detectors 2 cs-137_20191018_150159.txt']
bbk_paths = ["../Data/CH_0@DT5730_1770_Espectrum_3 Detector Collimated Background Count_20191014_231251.txt",
            "../Data/CH_1@DT5730_1770_Espectrum_3 Detector Collimated Background Count_20191014_231251.txt",
            "../Data/CH_2@DT5730_1770_Espectrum_3 Detector Collimated Background Count_20191014_231251.txt"]
raw_spectra = np.array([get_spectrum(rawf, 5) for rawf in raw_paths])  # 5 hours
bk_spectra  = np.array([get_spectrum(bkf, 5) for bkf in bk_paths])  # 5 hours
cali_coeffs = [661.7/427, 661.7/1688, 661.7/543]
cali_spectra = np.array([np.array([c * np.arange(0, len(raw)), raw - bk]).T for c, raw, bk in zip(cali_coeffs, raw_spectra, bk_spectra)])


# In[241]:


xlims = [[0,1000],[0,2000],[0,1000]]
ylims = [[0,1800],[0,250],[0,700]]
fig,ax = plt.subplots(3,2,figsize = (12,10))
for n in np.arange(3):
    ax[n][0].plot(raw_spectra[n], label='Raw')
    ax[n][0].plot(bk_spectra[n] , label='Background')
    ax[n][0].plot(raw_spectra[n]-bk_spectra[n], label='Backgound subtracted')
    ax[n][0].set_xlim(xlims[n])
    ax[n][0].set_ylim(ylims[n])
    ax[n][0].set_title('CH%s'%n)
    ax[n][0].set_xlabel('Channel number')
    ax[n][0].set_ylabel('Count rate ($h^{-1}$)')
    ax[n][0].legend()
    ax[n][1].plot(cali_spectra[n,:,0],cali_spectra[n,:,1], label='Calibrated')
    ax[n][1].set_title('CH%s'%n)
    ax[n][1].set_xlabel('Energy (keV)')
    ax[n][1].set_ylabel('Count rate ($h^{-1}$)')
    ax[n][1].set_xlim([0,1000])
    ax[n][1].set_ylim(ylims[n])

    ax[n][1].legend()


# ## Sum of counts in the peak

# In[ ]:


# sum of counts under the peak, after subtracting the Compton continuum
peak_sums=[]
for n in np.arange(3):
    summ,_,_,_ = peak_sum(cali_spectra[n])
    peak_sums.append(summ)
peak_sums = np.array(peak_sums)


# In[243]:


# plot
fig,ax = plt.subplots(3,1,figsize = (12,10))
for n in np.arange(3):
    summ, compton_sum, up_index, low_index = peak_sum(cali_spectra[n], ROI=[600,750])
    ax[n].plot(cali_spectra[n,:,0],cali_spectra[n,:,1])
    # ax[n].plot(np.array([cali_spectra[n,low_index,0],cali_spectra[n,low_index,0],cali_spectra[n,up_index,0],cali_spectra[n,up_index,0]]),
    #            np.array([0,cali_spectra[n,low_index,1],cali_spectra[n,up_index,1],0]),'o-',label='Compton')
    xs=cali_spectra[n,low_index:up_index,0]
    ys=cali_spectra[n,low_index:up_index,1]
    ax[n].fill_between(xs,ys, alpha=0.4, label='Net = %.1f'%summ)
    ax[n].fill_between(np.array([cali_spectra[n,low_index,0],cali_spectra[n,low_index,0],cali_spectra[n,up_index,0],cali_spectra[n,up_index,0]]),
               np.array([0,cali_spectra[n,low_index,1],cali_spectra[n,up_index,1],0]),alpha=0.4,label='Compton=%.1f'%compton_sum)
    ax[n].set_title('CH%s'%n)
    ax[n].set_xlabel('Energy (keV)')
    ax[n].set_ylabel('Count rate ($h^{-1}$)')
    ax[n].set_xlim([0,1000])
    ax[n].set_ylim(ylims[n])
    ax[n].legend()


# # Solve the Inverse Problem

# In[246]:


# comment line 61 in `Response Matrix` section, 
# and re-calculate the response matrix
print(respMatrix)
# print(respMatrix.shape)


# In[331]:


Out = peak_sums
print(Out)
# print(Out.shape)


# ## Built-in

# In[339]:


import scipy.optimize as op

In = op.lsq_linear(respMatrix,
                  Out, (0, np.inf),
                    method='bvls',
                    #method='trf',
                    tol=1e-30,
                    max_iter=10000,    verbose=2
                   )['x']

# calculate the activity
activity = np.sum(In)/3600
print("The activity using TRF is {} Bq".format(activity))


# $A_{cal} = 61801.8 \text{ Bq}, A_{real} = 70800 \text{ Bq}$

# In[ ]:


# Creating a NxN matrix Z with the elements of In
indexIn = 0
Z = np.zeros((N, N))
for b in range(0, N):
    for a in range(0, N):
        if (b, a) in zero_pixels:
            Z[b][a] = 0
        else:
            Z[b][a] = In[indexIn]
            indexIn += 1


# In[342]:

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure()
ax = plt.gca()
matfig = ax.imshow(Z, extent=[-6,6,-6,6], origin='lower')

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(matfig, cax=cax)

# plot the pipe
innerradius = innerradius / W * N  # in px
outerradius = outerradius / W * N  # in px
center = (0, 0)  # in px
circle1 = plt.Circle(center, innerradius, color='r', fill=False)
circle2 = plt.Circle(center, outerradius, color='r', fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)

# labels
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.savefig("matfig")


# In[330]:


# ## FISTA

# Let $f(x) = \frac{1}{2}\|y-Ax\|^2$, find:
# \begin{equation}
#     \hat{x} = \text{arg }\underset{x\geq 0}{\text{min}} f(x)
# \end{equation}
# [FISTA](https://github.com/tiepvupsu/FISTA): a fast iterative shrinkage-thresholding algorithm to find $\hat{x}$.
# In this case: 
# \begin{equation}
#     \nabla f(x) = A^TAx-A^Ty
# \end{equation}
# \begin{equation}
#     \Rightarrow L(f) = \text{max eigenvalue of }A^TA
# \end{equation}

# In[ ]:


#initialization

A = respMatrix
u, s, vt = svds(A, 1, which='LM')  # svd
L = s[0] ** 2
# In[ ]:


x = z = np.ones(A.shape[1])
y = Out
t = 1
lam = 1E-8
max_iter = 2000
err = np.zeros(max_iter)
for k in np.arange(0,max_iter):
    xp = z - 1/L * A.T@(A@z-y)
    xp = np.maximum(xp-lam/L, 0)
    tp = (1+np.sqrt(1+4* t**2)) / 2
    z = xp + (t-1) / tp *(xp-x)
    # update
    t = tp
    x = xp
    # err[k] = np.linalg.norm(y - A@x)
    err[k] = abs(np.sum(x)/3600 - 70800) / 70800
    # Stops loop when error difference is small -- Prevents overfitting??
    #if np.abs(err[k] - err[k-1])/err[k] <= 10 ** -4.5: #or err[k] <= 10E-3 and k >= 1:
     #   break
    if k == 250:
        break
# In[334]:


indexIn = 0
Z = np.zeros((N, N))
for b in range(0, N):
    for a in range(0, N):
        if (b, a) in zero_pixels:
            Z[b][a] = 0
        else:
            Z[b][a] = x[indexIn]
            indexIn += 1

fig,ax = plt.subplots(2,1,figsize=(12,10))
ax[0].plot(np.arange(0,max_iter), 100 * err)
ax[0].set_title('Error')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('abs$(A_{cal}-A_{real})/ A_{real}$ (%)')

matfig = ax[1].matshow(Z)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(matfig, cax=cax)

# $A_{cal} = 61867.7 \text{ Bq}$

# In[336]:

activity = np.sum(x)/3600
print("The activity using FISTA is {} Bq".format(activity))

import seaborn as sns
sns.set()

from mpl_toolkits.axes_grid1 import make_axes_locatable

indexIn = 0
Z = np.zeros((N, N))
for b in range(0, N):
    for a in range(0, N):
        if (b, a) in zero_pixels:
            Z[b][a] = 0
        else:
            Z[b][a] = x[indexIn]
            indexIn += 1

plt.figure()
ax = plt.gca()
matfig = ax.imshow(Z, extent=[-6,6,-6,6], origin='lower')

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(matfig, cax=cax)

# plot the pipe
innerradius = innerradius / W * N  # in px
outerradius = outerradius / W * N  # in px
center = (0, 0)  # in px
circle1 = plt.Circle(center, innerradius, color='r', fill=False)
circle2 = plt.Circle(center, outerradius, color='r', fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)

# labels
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.savefig("matfig")


#plt.show()


A = respMatrix
b = Out

# import scipy.linalg as linalg
# import numpy as np  # same matrix A and B as in LU decomposition
# print(A)
# print(b)
# q, r = np.linalg.qr(A)
# p = np.dot(q.T, b)
# print(np.dot(np.linalg.inv(r), p))

# print(A.shape)
# print(np.linalg.matrix_rank(A))
# print("The activity using closed form least squares solution is {} Bq".format(np.sum(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b)))/3600))



max = Out.max()
Out = Out/max

def noisy_val_grad(theta_hat, data_, label_, deg, lamba):
    gradient = np.zeros_like(theta_hat)
    loss = 0

    for i in range(data_.shape[0]):
        x_ = data_[i, :].reshape(-1, 1)
        y_ = label_[i, 0]
        err = np.matmul(np.transpose(x_), theta_hat) - y_
        grad = 2 * (np.matmul(np.transpose(x_), theta_hat) - y_) * x_ * (
                    np.absolute(np.matmul(np.transpose(x_), theta_hat) - y_) ** (deg - 2)) \
               #+ lamba * np.abs(theta_hat[i])/theta_hat[i]
        #print(grad.shape)
        l = np.abs(err) ** deg
        loss += l / data_.shape[0]
        gradient += np.divide(grad, data_.shape[0])

    return loss, gradient

Out = np.asarray([[i] for i in Out])
lr = 0.2
max_iter = 50000
theta_init = np.random.random((np.shape(respMatrix)[1], 1)) * 0
deg_ = 2
data_num = np.shape(respMatrix)[0]
batch_size = data_num
beta_1 = 0.9
beta_2 = 0.999
m = 0
nu = 0
e = 10E-8
G = 0

lam = 10E-9
#u, s, vt = svds(respMatrix, 1, which='LM')  # svd
#L = s[0] ** 2
#lr = 1/L

# initialize momentum
mew = 0.9999
v = 0

In = theta_init.copy()
err = np.zeros(max_iter)
stuff = []
for t in range(max_iter):
    #print(t)
    idx = np.random.choice(data_num, batch_size)
    train_loss, gradient = noisy_val_grad(In, respMatrix, Out, deg_, 0)
    #err[t] = train_loss
    err[t] = np.linalg.norm(Out - respMatrix.dot(In))


    # ADAGRAD
    # stuff.append(gradient ** 2)
    # G = sum(stuff)
    # In = In - (gradient * (lr / (np.sqrt(G + e))))

    # GD
    #In = In - lr * gradient

    # ADAM
    m = beta_1 * m + (1 - beta_1) * gradient
    nu = beta_2 * nu + (1 - beta_2) * np.square(gradient)
    m_bar = m / (1 - beta_1 ** (t + 1))
    nu_bar = nu / (1 - beta_2 ** (t + 1))
    In = In - lr * m_bar / (np.sqrt(nu_bar) + e)

    # Nesterov accelerated gradient
    # _, gradient = noisy_val_grad(In - mew * v, R, Out, deg_, 0)
    # v = mew * v + lr * gradient
    # In = In - v

    In = np.asarray([[i] for i in np.maximum(In, 0).ravel()])
    # 10E-4 works best for source1
    #if np.abs(err[t] - err[t-1])/err[t] <= 10E-4: # or err[t] <= 10E-3 and t >= 1:
        #break

fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
# ax1.set_title("Reconstruction\nFiltered back projection")
ax1.plot(np.arange(0, max_iter), err)
# ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('$||Ax-y||$')
#print(np.matmul(R,In) - Out)
#print(np.matmul(R,In) - Out)

In = In*max
print("The activity using ADAM is {} Bq".format(sum(In.ravel())/3600))

plt.show()