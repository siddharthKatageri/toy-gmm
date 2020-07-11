from sklearn.mixture import GaussianMixture
import numpy as np
import math
from numpy import *
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from numpy.random import uniform, seed
from scipy.interpolate import griddata
from matplotlib import cm
from scipy.stats import multivariate_normal
from sklearn.datasets import make_moons
from matplotlib.patches import Ellipse



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)



def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    #print(size)
    #print(len(mu))
    #print()
    #print(sigma.shape)
    assert (size == len(mu) and (size, size) == sigma.shape), "dims of input do not match"
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        assert det!=0, "covariance matrix cannot be singular"

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd

def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def gauss_new(X,Sigma,mu):
    #X=np.vstack((x,y)).T
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


#X, ymoon = make_moons(200, noise=.05, random_state=0)

#plt.scatter(X[:, 0], X[:, 1])
#plt.show()

#loading and normalizing my custom dataset
x = np.loadtxt("input.dat")
X = normalize(x)

'''
#----------------------------------------------------------------------------------------
#AIC and BIC plot

n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', n_init=5).fit(X)
          for n in n_components]

plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
plt.xticks(np.arange(1, 21, 2))
plt.show()

#----------------------------------------------------------------------------------------
'''








print("Dataset shape:", X.shape)

# this number is the number of guassian you want to fit on the data in the latent space
# decided by looking at the plot of aic and bic curve
k = 2

gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=100, n_init=100, verbose=False).fit(X)
print("cov's are :\n", gmm.covariances_)
print("means are:\n", gmm.means_)
prediction_gmm = gmm.predict(X)
probs = gmm.predict_proba(X)
centers = np.zeros((k,X.shape[1]))
for i in range(k):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]

plt.figure(figsize = (10,8))
plt.scatter(X[:, 0], X[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);

plot_gmm(gmm, X, label=True)

#test datapoints, to calculate the probability of the datapoints coming from this distribution
my_val = np.array([[0.6, -0.6], [-0.9, 0.8], [0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5], [2, 2], [5, 5], [10, 10], [15, 15], [20, 20]])

gmm.weights_ = gmm.weights_.reshape(1,len(gmm.weights_))
print("weights are:\n", gmm.weights_)
print()
gen_probs = np.zeros((len(my_val), k))
for ind, l in enumerate(range(k)):
    var = multivariate_normal(mean=gmm.means_[l], cov=gmm.covariances_[l])  #multivariate_normal is sklearn's implementation for my
    gen_probs[:,ind] = var.pdf(my_val)                                      # norm_pdf_multivariate function defined above
print("generative probs:\n", gen_probs)
for ele in gen_probs:
    print("prob(x):\n", np.sum(np.multiply(gmm.weights_, ele))) # printing prob of each datapoint coming from this distribution
plt.show()                                                      # observe far away datapoints have less prob



