import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pylab as pl
from numpy.random import uniform, seed
from matplotlib import cm
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    #print(size)
    assert (size == len(mu) and (size, size) == sigma.shape), "dims of input do not match"
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        assert det!=0, "covariance matrix cannot be singular"

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result

def get_true(y):
    true = []
    for i in y:
        if(i=="Alaska"):
            true.append(1)
        else:
            true.append(0)
    return true

def seperate_data(x, y):
    alaska = []
    canada = []
    for i,val in enumerate(y):
        if(val=='Alaska'):
            alaska.append(x[i])
        else:
            canada.append(x[i])
    return np.array(alaska), np.array(canada)


def min_max_normalize(x):
    global x1_min, x2_min, x1_max, x2_max
    x1_max = np.max(x[:,0])
    x1_min = np.min(x[:,0])

    x2_max = np.max(x[:,1])
    x2_min = np.min(x[:,1])
    x[:,0] = (x[:,0]-x1_min)/(x1_max-x1_min)
    x[:,1] = (x[:,1]-x2_min)/(x2_max-x2_min)

    return x

def compute_mu(x_a, x_c):
    mu1 = [np.sum(x_a[:,0])/x_a.shape[0], np.sum(x_a[:,1])/x_a.shape[0]]
    mu2 = [np.sum(x_c[:,0])/x_c.shape[0], np.sum(x_c[:,1])/x_c.shape[0]]

    return np.array(mu1), np.array(mu2)

def compute_covar(x_a, mu1):
    sub = x_a - mu1
    covar = np.zeros((2,2))
    for i in sub:
        dot = np.dot(i.reshape(2,1), i.reshape(2,1).T)
        covar = covar + dot
    return covar/x_a.shape[0]

def descion_boundary(mu1, covar_a, mu2, covar_c):
    X1, X2 = np.mgrid[-3:3:100j, -3:3:100j]
    #print(X1)
    #print(X2)
    #print(X1.shape)
    #print(X2.shape)
    #exit()
    x1_ravel = X1.ravel()
    x2_ravel = X2.ravel()
    rav_data = []
    for rav1, rav2 in zip(x1_ravel,x2_ravel):
        rav_data.append([rav1, rav2])


    dif = []
    for every in rav_data:
        p_a = norm_pdf_multivariate(every, np.squeeze(mu1), matrix(covar_a))
        p_c = norm_pdf_multivariate(every, np.squeeze(mu2), matrix(covar_c))
        dif.append([p_a-p_c])

    dif = np.array(dif)
    dif = dif.reshape(X1.shape)

    return X1, X2, dif

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd

def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(x,y,z):
    # define grid.
    xi = np.linspace(-2.1, 2.1, 100)
    yi = np.linspace(-2.1, 2.1, 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test (%d points)' % npts)
    #plt.show()






x = np.loadtxt("input.dat")
X = normalize(x)
print("Dataset shape:", X.shape)
plt.scatter(X[:,0], X[:,1])
#plt.show()




# Create a grid for visualization purposes
x = np.linspace(np.min(X[...,0])-1,np.max(X[...,0])+1,100)
y = np.linspace(np.min(X[...,1])-1,np.max(X[...,1])+1,80)
X_,Y_ = np.meshgrid(x,y)
pos = np.array([X_.flatten(),Y_.flatten()]).T
print(pos.shape)
print(np.max(pos[...,1]))

k = 2
weights = np.ones(k)/k
print("the weights selected are:\n", weights)
means = np.random.choice(X.flatten(), (k,X.shape[1]))
print("\n the means selected are:\n", means)
cov = []
for i in range(k):
    cov.append(make_spd_matrix(X.shape[1]))
cov = np.array(cov)
print("\n cov shape:", cov.shape)
print("the cov matrix selected are:\n", cov)


colors = ['tab:blue', 'tab:orange', 'tab:green', 'magenta', 'yellow', 'red', 'brown', 'grey']
eps=1e-8

# run GMM for 40 steps
for step in range(40):

  # visualize the learned clusters
  if step % 1 == 0:
    plt.figure(figsize=(12,int(8)))
    plt.title("Iteration {}".format(step))
    axes = plt.gca()

    likelihood = []
    for j in range(k):
      likelihood.append(multivariate_normal.pdf(x=pos, mean=means[j], cov=cov[j]))
    likelihood = np.array(likelihood)
    predictions = np.argmax(likelihood, axis=0)

    for c in range(k):
      pred_ids = np.where(predictions == c)
      plt.scatter(pos[pred_ids[0],0], pos[pred_ids[0],1], color=colors[c], alpha=0.2, edgecolors='none', marker='s')

    plt.scatter(X[...,0], X[...,1], facecolors='none', edgecolors='grey')

    for j in range(k):
      plt.scatter(means[j][0], means[j][1], color=colors[j])

    #plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
    #plt.show()

  likelihood = []
  # Expectation step
  for j in range(k):
    likelihood.append(multivariate_normal.pdf(x=X, mean=means[j], cov=cov[j]))
  likelihood = np.array(likelihood)
  assert likelihood.shape == (k, len(X))

  b = []
  # Maximization step
  for j in range(k):
    # use the current values for the parameters to evaluate the posterior
    # probabilities of the data to have been generanted by each gaussian
    b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))

    # updage mean and variance
    means[j] = np.sum(b[j].reshape(len(X),1) * X, axis=0) / (np.sum(b[j]+eps))
    cov[j] = np.dot((b[j].reshape(len(X),1) * (X - means[j])).T, (X - means[j])) / (np.sum(b[j])+eps)

    # update the weights
    weights[j] = np.mean(b[j])

    assert cov.shape == (k, X.shape[1], X.shape[1])
    assert means.shape == (k, X.shape[1])


print("cov:\n",cov)
print("\nmeans:\n", means)

plt.clf()
seed(1234)
npts = 1000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
for i in range(k):
    z = gauss(x, y, Sigma=cov[i], mu=means[i])
    plot_countour(x, y, z)

#plt.colorbar() # draw colorbar
#plt.show()
