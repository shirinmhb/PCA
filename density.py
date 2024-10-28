import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from matplotlib import cm
from scipy.stats import multivariate_normal

class Density:

  def __init__(self, mean0, cov0, numData, bw):

    self.x0, self.x1 = np.random.multivariate_normal(mean0, cov0, numData).T
    self.mu = mean0
    self.sigma = cov0
    self.numData = numData
    self.minX0 = min(self.x0)
    self.minX1 = min(self.x1)
    self.maxX0 = max(self.x0)
    self.maxX1 = max(self.x1)
    self.numEstimate = 100

  def knn(self, sample0, sample1, k):
    

  def pknn_function(self, x0, x):
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        # z = self.parzenWindow(x0[i][j], x1[i][j], h)
        z = self.knn(x0[i][j], x1[i][j], k, 0.5)
        zRow += [z]
      zValue += [zRow]
    return zValue

  def plotKNNDe(self, k):
    x0 = np.linspace(self.minX0, self.maxX0, self.numEstimate)
    x1 = np.linspace(self.minX1, self.maxX1, self.numEstimate)

    x0, x1 = np.meshgrid(x0, x1)
    j = np.array(self.pknn_function(x0, x1, k))
  
    fig = plt.figure(figsize=plt.figaspect(0.3))

    j_plot = fig.add_subplot(1, 3, 1, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 0)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 2, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 90)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 3, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(45, 45)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")
    plt.show()

  def kU(self, u):
    if abs(u) < 1/2:
      return 1
    return 0

  def kG(self, u, sigma):
    # return (1 / ( sigma * (2 * np.pi)**0.5) ) * np.exp(-1*((u)/(2 * sigma**2)))
    return(((sigma*(2*np.pi)**0.5)**-1) * np.exp(-0.5 * u**2) )

  def GaussiKernel(self, sample0, sample1, h, sigma):
    sumKD0 = 0
    sumKD1 = 0
    for i in range(len(self.x0)):
      sumKD0 += self.kG((sample0 - self.x0[i])/h, sigma)
      sumKD1 += self.kG((sample1 - self.x1[i])/h, sigma)
    sumKD0 = sumKD0 / (self.numData * h**2)
    sumKD1 = sumKD1 / (self.numData * h**2)
    return (sumKD0 * sumKD1)
  
  def parzenWindow(self, sample0, sample1, h):
    sumKD0 = 0
    sumKD1 = 0
    for i in range(len(self.x0)):
      sumKD0 += self.kU((sample0 - self.x0[i])/h)
      sumKD1 += self.kU((sample1 - self.x1[i])/h)
    sumKD0 = sumKD0 / (self.numData * h**2)
    sumKD1 = sumKD1 / (self.numData * h**2)
    return (sumKD0 * sumKD1)

  def parzen_function(self, x0, x1, h):
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        # z = self.parzenWindow(x0[i][j], x1[i][j], h)
        z = self.GaussiKernel(x0[i][j], x1[i][j], h, 0.5)
        zRow += [z]
      zValue += [zRow]
    return zValue

  def plotParzenWindowDe(self, h):
    x0 = np.linspace(self.minX0, self.maxX0, self.numEstimate)
    x1 = np.linspace(self.minX1, self.maxX1, self.numEstimate)

    x0, x1 = np.meshgrid(x0, x1)
    j = np.array(self.parzen_function(x0, x1, h))
  
    fig = plt.figure(figsize=plt.figaspect(0.3))

    j_plot = fig.add_subplot(1, 3, 1, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 0)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 2, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 90)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 3, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(45, 45)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")
    plt.show()

  def histogramEstimateDensity(self, x0, x1, binWidth):
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        z = self.findP(x0[i][j], x1[i][j], binWidth)
        zRow += [z]
      zValue += [zRow]
    return zValue
  
  def findP(self, x, y, binWidth):
      x0Index = int((x - self.minX0) // binWidth )
      x1Index = int((y - self.minX1) // binWidth )
      return(self.count[x0Index][x1Index] / (self.numData * binWidth))

  def histogram(self, binWidth):
    nx0 = int((self.maxX0 - self.minX0) // binWidth) + 1
    nx1 = int((self.maxX1 - self.minX1) // binWidth) + 1
    count = [ [0] * nx1 for _ in range(nx0)]
    for i in range(len(self.x0)):
      x = self.x0[i]
      y = self.x1[i]
      x0Index = int((x - self.minX0) // binWidth )
      x1Index = int((y - self.minX1) // binWidth )
      count[x0Index][x1Index] += 1
    self.count = count

  def plotHistogramDe(self, binWidth):
    self.histogram(binWidth)
    x0 = np.linspace(self.minX0, self.maxX0, self.numEstimate)
    x1 = np.linspace(self.minX1, self.maxX1, self.numEstimate)

    x0, x1 = np.meshgrid(x0, x1)
    j = np.array(self.histogramEstimateDensity(x0, x1, binWidth))
    fig = plt.figure(figsize=plt.figaspect(0.3))

    j_plot = fig.add_subplot(1, 3, 1, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 0)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 2, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 90)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 3, projection='3d')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x0, x1, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(45, 45)
    j_plot.set_xlabel('x0')
    j_plot.set_ylabel('x1')
    j_plot.set_zlabel("Px")
    plt.show()


  def plotRealD(self):
    X = np.linspace(self.minX0, self.maxX0, self.numData)
    Y = np.linspace(self.minX0, self.maxX0, self.numData)
    X, Y = np.meshgrid(X, Y)
    # Mean vector and covariance matrix
    mu = np.array(self.mu)
    Sigma = np.array(self.sigma)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    def multivariate_gaussian(pos, mu, Sigma):
      """Return the multivariate Gaussian distribution on array pos."""

      n = mu.shape[0]
      Sigma_det = np.linalg.det(Sigma)
      Sigma_inv = np.linalg.inv(Sigma)
      N = np.sqrt((2*np.pi)**n * Sigma_det)
      # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
      # way across all the input variables.
      fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

      return np.exp(-1 *fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    fig = plt.figure(figsize=plt.figaspect(0.3))

    j_plot = fig.add_subplot(1, 3, 1, projection='3d')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 0)
    j_plot.set_xlabel('X')
    j_plot.set_ylabel('Y')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 2, projection='3d')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 90)
    j_plot.set_xlabel('X')
    j_plot.set_ylabel('Y')
    j_plot.set_zlabel("Px")

    j_plot = fig.add_subplot(1, 3, 3, projection='3d')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(45, 45)
    j_plot.set_xlabel('X')
    j_plot.set_ylabel('Y')
    j_plot.set_zlabel("Px")
    plt.show()
   
  def plotPDF(self):
    # x = np.linspace(self.minX0, self.maxX0, self.numData)
    # y = np.linspace(self.minX1, self.maxX1, self.numData)
    # x, y = np.meshgrid(x, y)
    mean0 = np.array(self.mu)
    x, y = np.mgrid[-3.0:10.0:30j, -7.5:12.5:30j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])
    z0 = multivariate_normal.pdf(xy, mean=mean0, cov=self.sigma)
    # Reshape back to a (30, 30) grid.
    z0 = z0.reshape(x.shape)
    fig = plt.figure()
    j_plot = fig.add_subplot(1, 3, 1, projection='3d')
    j_plot.plot_surface(x, y, z0, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.plot_surface(x, y, z0, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    j_plot.view_init(0, 0)
    j_plot.set_xlabel('x')
    j_plot.set_ylabel('y')
    j_plot.set_zlabel("Px")
    ax = fig.add_subplot(111, projection='3d')
    plt.show()

mean = [2, 5]
cov = [[2, 0], [0, 2]]  

d = Density(mean, cov, 500, 0.5)
# d.plotHistogramDe(binWidth=0.5)
d.plotParzenWindowDe(h=0.5)


   # plot data
    # fig1 = plt.figure()
    # plt.plot(self.x0, self.x1, '.r')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # fig2 = plt.figure()
    # plt.hist2d(self.x0, self.x1, bins=10)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Counts')
    # plt.show()

    # data = np.array(data)
    # length = data.shape[0]
    # width = data.shape[1]
    # length = 3
    # width = 5
    # x, y = np.meshgrid(np.arange(length), np.arange(width))
    # print("x", x)
    # print("y", y)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.plot_surface(x, y, data)
    # plt.show()