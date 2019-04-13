import numpy as np
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt
from numpy.linalg import norm

def compute_data_cost(I1, I2, num_disp_values, Tau):
    """data_cost: a 3D array of sixe height x width x num_disp_value;
    data_cost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    The cost is min(1/3 ||I1[y,x]-I2[y,x-l]||_1, Tau)."""
    h,w,_ = I1.shape
    dataCost=np.zeros((h,w,num_disp_values))

    for lp in range(num_disp_values):
        dataCost[:, :, lp] = np.minimum(1./3*norm(I1[:, :] - np.roll(I2, -lp, axis=1), axis=2, ord=1), Tau*np.ones((h, w)))

    return dataCost

def compute_energy(dataCost,disparity,Lambda):
    """dataCost: a 3D array of sixe height x width x num_disp_values;
    dataCost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    disparity: array of size height x width containing disparity of each pixel.
    (an integer between 0 and num_disp_values-1)
    Lambda: a scalar value.
    Return total energy, a scalar value"""
    return 0

def update_msg(msgUPrev,msgDPrev,msgLPrev,msgRPrev,dataCost,Lambda):
    """Update message maps.
    dataCost: 3D array, depth=label number.
    msgUPrev,msgDPrev,msgLPrev,msgRPrev: 3D arrays (same dims) of old messages.
    Lambda: scalar value
    Return msgU,msgD,msgL,msgR: updated messages"""
    msgU=np.zeros(dataCost.shape)
    msgD=np.zeros(dataCost.shape)
    msgL=np.zeros(dataCost.shape)
    msgR=np.zeros(dataCost.shape)

    # msg = [msgU, msgD, msgL, msgR]
    # neighbors = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    h,w,num_disp_values = dataCost.shape

    # npqU = np.zeros((h, w, num_disp_values))
    # npqD = np.zeros((h, w, num_disp_values))
    # npqL = np.zeros((h, w, num_disp_values))
    # npqR = np.zeros((h, w, num_disp_values))

    msg_incoming_from_U = np.roll(msgDPrev, 1, axis=0)
    msg_incoming_from_L = np.roll(msgRPrev, 1, axis=1)
    msg_incoming_from_D = np.roll(msgUPrev, -1, axis=0)
    msg_incoming_from_R = np.roll(msgLPrev, -1, axis=1)

    # npqU = dataCost + np.roll(msgUPrev, -1, axis=0) + np.roll(msgLPrev, -1, axis=1) + np.roll(msgRPrev, 1, axis=1)
    # npqR = dataCost + np.roll(msgUPrev, -1, axis=0) + np.roll(msgDPrev, 1, axis=0) + np.roll(msgRPrev, 1, axis=1)
    # npqD = dataCost + np.roll(msgUPrev, 1, axis=0) + np.roll(msgLPrev, -1, axis=1) + np.roll(msgRPrev, 1, axis=1)
    # npqL = dataCost + np.roll(msgDPrev, 1, axis=0) + np.roll(msgUPrev, -1, axis=0) + np.roll(msgRPrev, 1, axis=1)

    npqU = dataCost + msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_R
    npqL = dataCost + msg_incoming_from_U + msg_incoming_from_D + msg_incoming_from_R
    npqD = dataCost + msg_incoming_from_L + msg_incoming_from_U + msg_incoming_from_R
    npqR = dataCost + msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_U

    spqU = np.amin(npqU, axis=2)#, keepdims=True)
    spqL = np.amin(npqL, axis=2)#, keepdims=True)
    spqD = np.amin(npqD, axis=2)#, keepdims=True)
    spqR = np.amin(npqR, axis=2)#, keepdims=True)

    # msgU = np.minimum(npqU, Lambda + spqU, axis=2)
    # msgL = np.minimum(npqL, Lambda + spqL, axis=2)
    # msgD = np.minimum(npqD, Lambda + spqD, axis=2)
    # msgR = np.minimum(npqR, Lambda + spqR, axis=2)

    for lp in range(num_disp_values):
        msgU[:, :, lp] = np.minimum(npqU[:, :, lp], Lambda + spqU)
        msgL[:, :, lp] = np.minimum(npqL[:, :, lp], Lambda + spqL)
        msgD[:, :, lp] = np.minimum(npqD[:, :, lp], Lambda + spqD)
        msgR[:, :, lp] = np.minimum(npqR[:, :, lp], Lambda + spqR)

    # for lp in range(num_disp_values):
    #     msgU[:, :, lp] = np.amin(npqU[:, :, lp], Lambda + spqU, axis=2)
    #     msgD[:, :, lp] = np.amin(npqD[:, :, lp], Lambda + spqD)
    #     msgL[:, :, lp] = np.amin(npqL[:, :, lp], Lambda + spqL)
    #     msgR[:, :, lp] = np.amin(npqR[:, :, lp], Lambda + spqR)
    # for y in range(h):
    #     for x in range(w):
    #         for lp in range(num_disp_values):
    #             for dq in neighbors:
    #                 s = 0
    #                 for dr in neighbors:
    #                     if dr != dq:
    #                         xr, yr = x+dr[0], y+dr[1]
    #                         xq, yq = x+dq[0], y+dq[1]
    #                         s += msg[yr, xr, lp]
    #             npq = compute_energy(dataCost,lp,Lambda) +



    return msgU,msgD,msgL,msgR

def normalize_msg(msgU,msgD,msgL,msgR):
    """Subtract mean along depth dimension from each message"""
    avg=np.mean(msgU,axis=2)
    msgU -= avg[:,:,np.newaxis]
    avg=np.mean(msgD,axis=2)
    msgD -= avg[:,:,np.newaxis]
    avg=np.mean(msgL,axis=2)
    msgL -= avg[:,:,np.newaxis]
    avg=np.mean(msgR,axis=2)
    msgR -= avg[:,:,np.newaxis]
    return msgU,msgD,msgL,msgR

def compute_belief(dataCost,msgU,msgD,msgL,msgR):
    """Compute beliefs, sum of data cost and messages from all neighbors"""
    beliefs=dataCost.copy()
    return beliefs

def MAP_labeling(beliefs):
    """Return a 2D array assigning to each pixel its best label from beliefs
    computed so far"""
    return np.zeros((beliefs.shape[0],beliefs.shape[1]))

def stereo_bp(I1,I2,num_disp_values,Lambda,Tau=15,num_iterations=60):
    """The main function"""
    dataCost = compute_data_cost(I1, I2, num_disp_values, Tau)
    energy = np.zeros((num_iterations)) # storing energy at each iteration
    # The messages sent to neighbors in each direction (up,down,left,right)
    h,w,_ = I1.shape
    msgU=np.zeros((h, w, num_disp_values))
    msgD=np.zeros((h, w, num_disp_values))
    msgL=np.zeros((h, w, num_disp_values))
    msgR=np.zeros((h, w, num_disp_values))

    for iter in range(num_iterations):
        msgU,msgD,msgL,msgR = update_msg(msgU,msgD,msgL,msgR,dataCost,Lambda)
        msgU,msgD,msgL,msgR = normalize_msg(msgU,msgD,msgL,msgR)
        # Next lines unused for next iteration, could be done only at the end
        # beliefs = compute_belief(dataCost,msgU,msgD,msgL,msgR)
        # disparity = MAP_labeling(beliefs)
        # energy[iter] = compute_energy(dataCost,disparity,Lambda)
    disparity = []
    return disparity,energy

# Input
img_left =imageio.imread('tsukuba/imL.png')
img_right=imageio.imread('tsukuba/imR.png')
# plt.subplot(121)
# plt.imshow(img_left)
# plt.subplot(122)
# plt.imshow(img_right)
# plt.show()
compute_data_cost(img_left, img_right, 15, 1)
# stereo_bp(img_left, img_right, 15, 1)
# Convert as float gray images
img_left=img_left.astype(float)
img_right=img_right.astype(float)

# Parameters
num_disp_values=16 # these images have disparity between 0 and 15.
Lambda=10.0

# Gaussian filtering
I1=scipy.ndimage.filters.gaussian_filter(img_left, 0.6)
I2=scipy.ndimage.filters.gaussian_filter(img_right,0.6)

disparity,energy = stereo_bp(I1,I2,num_disp_values,Lambda)
imageio.imwrite('disparity_{:g}.png'.format(Lambda),disparity)

# Plot results
plt.subplot(121)
plt.plot(energy)
plt.subplot(122)
plt.imshow(disparity,cmap='gray',vmin=0,vmax=num_disp_values-1)
plt.show()
