import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from matplotlib import cm
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix 

class Bayesian():

  def __init__(self, mean0, cov0, mean1, cov1, mean2, cov2, numData, numTrain):
    
    x0, y0 = np.random.multivariate_normal(mean0, cov0, numData).T
    x1, y1 = np.random.multivariate_normal(mean1, cov1, numData).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, numData).T

    self.cov0 = cov0
    self.cov1 = cov1
    self.cov2 = cov2
    self.mean0 = mean0
    self.mean1 = mean1
    self.mean2 = mean2

    self.meanClass0 = np.array( [(sum(x0[:numTrain])/numTrain), (sum(y0[:numTrain])/numTrain)] )
    self.meanClass1 = np.array( [(sum(x1[:numTrain])/numTrain), (sum(y1[:numTrain])/numTrain)] )
    self.meanClass2 = np.array( [(sum(x2[:numTrain])/numTrain), (sum(y2[:numTrain])/numTrain)] )
    self.phiClass0 = 1/3
    self.phiClass1 = 1/3
    self.phiClass2 = 1/3
    self.countData = numTrain * 3

    features = []
    label =[]
    demeanX0 = []
    demeanX1 = []
    demeanX2 = []
    testFeature = []
    testLabel = []
    for i in range(numTrain):
      features.append([x0[i], y0[i]]) 
      label.append([0])
      demeanX0.append([x0[i] - self.meanClass0[0], y0[i] - self.meanClass0[1]])
      features.append([x1[i], y1[i]]) 
      label.append([1])
      demeanX1.append([x1[i] - self.meanClass1[0], y1[i] - self.meanClass1[1]])
      features.append([x2[i], y2[i]])  
      label.append([2])
      demeanX2.append([x2[i] - self.meanClass2[0], y2[i] - self.meanClass2[1]])

    for j in range(numTrain, numData):
      testFeature.append([x0[j], y0[j]]) 
      testLabel.append([0])
      testFeature.append([x1[j], y1[j]]) 
      testLabel.append([1])
      testFeature.append([x2[j], y2[j]]) 
      testLabel.append([2])

    self.features = np.array(features)
    self.label = np.array(label)
    self.testFeature = np.array(testFeature)
    self.testLabel = np.array(testLabel)

    demeanX0 = np.array(demeanX0)
    demeanX1 = np.array(demeanX1)
    demeanX2 = np.array(demeanX2)
    demeanX0 = demeanX0/np.linalg.norm(demeanX0, ord=2, axis=1, keepdims=True)
    demeanX1 = demeanX1/np.linalg.norm(demeanX1, ord=2, axis=1, keepdims=True)
    demeanX2 = demeanX2/np.linalg.norm(demeanX2, ord=2, axis=1, keepdims=True)
    self.sigma0 = np.round( (1/self.countData) * (demeanX0.T @ demeanX0) , 5)
    self.sigma1 = np.round( (1/self.countData) * (demeanX1.T @ demeanX1) , 5)
    self.sigma2 = np.round( (1/self.countData) * (demeanX2.T @ demeanX2) , 5)

  def classifier(self, n, dataX, dataY):
    predict = []
    coefficient0 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma0))) )
    coefficient1 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma1))) )
    coefficient2 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma2))) )
    correct = 0
    for i in range(len(dataX)):

      demean0 = dataX[i] - self.meanClass0
      demean1 = dataX[i] - self.meanClass1
      demean2 = dataX[i] - self.meanClass2

      sigmaInvers0 = np.linalg.inv(self.sigma0)
      sigmaInvers1 = np.linalg.inv(self.sigma1)
      sigmaInvers2 = np.linalg.inv(self.sigma2)

      pXY0 = coefficient0 * (np.exp(  -0.5 * ( (demean0.T @ sigmaInvers0) @ demean0 )  ))
      pXY1 = coefficient1 * (np.exp(  -0.5 * ( (demean1.T @ sigmaInvers1) @ demean1 )  ))
      pXY2 = coefficient2 * (np.exp(  -0.5 * ( (demean2.T @ sigmaInvers2) @ demean2 )  ))

      l0 = pXY0 * self.phiClass0
      l1 = pXY1 * self.phiClass1
      l2 = pXY2 * self.phiClass2
      
      res = max(l0,l1,l2)

      if res == l0:
        predict.append(0)
        if dataY[i][0] == 0:
          correct += 1
      elif res == l1:
        predict.append(1)
        if dataY[i][0] == 1:
          correct += 1
      elif res == l2:
        predict.append(2)
        if dataY[i][0] == 2:
          correct += 1

    accuracy = correct / len(dataX) * 100

    print(accuracy, "%")

# print("dataSet results")
# mean0 = [2, 5]
# cov0 = [[2, 0], [0, 2]]  
# mean1 = [8, 1]
# cov1 = [[3, 1], [1, 3]]
# mean2 = [5, 3]
# cov2 = [[2, 1], [1, 2]]
# b = Bayesian(mean0, cov0, mean1, cov1, mean2, cov2, numData=500, numTrain=450)
# print("accuracy train")
# b.classifier(n=2, dataX=b.features, dataY=b.label)
# print("accuracy test")
# b.classifier(n=2, dataX=b.testFeature, dataY=b.testLabel)

class GaussianBayesian():
  def __init__(self, mean0, cov0, mean1, cov1, mean2, cov2):
    x0, y0 = np.random.multivariate_normal(mean0, cov0, 500).T
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T

    self.x0Class1 = x0[:450]
    self.x1Class1 = y0[:450]
    self.x0Class2 = x1[:450]
    self.x1Class2 = y1[:450]
    self.x0Class3 = x2[:450]
    self.x1Class3 = y2[:450]

    self.testX0 = x0[450:].tolist() + x1[450:].tolist() + x2[450:].tolist()
    self.testX1 = y0[450:].tolist() + y1[450:].tolist() + y2[450:].tolist()
    self.labelTest = 50 * [1] + 50 * [2] + 50 * [3]

    self.cov0 = cov0
    self.cov1 = cov1
    self.cov2 = cov2
    self.mean0 = mean0
    self.mean1 = mean1
    self.mean2 = mean2


  def newK(self, u, sigma):
    a = u * sigma * u
    return ((2*np.pi)**-1) * np.exp(-0.5 * a)


  def gaussian_kernel(self, x0Train, x1Train, sample0, sample1, h, sigma):
    sumKD0 = 0
    sumKD1 = 0
    numTrain = len(x0Train)
    for i in range(len(x0Train)):
      sumKD0 += self.newK((sample0 - x0Train[i])/h, sigma)
      sumKD1 += self.newK((sample1 - x1Train[i])/h, sigma)

    sumKD0 = (sumKD0 / (100 * h**2)) / 2
    sumKD1 = (sumKD1 / (100 * h**2)) / 2
    return (sumKD0 * sumKD1)

  def classifier(self):
    cor = 0
    pred = []
    for i in range(len(self.testX0)):
      prob1 = self.gaussian_kernel(x0Train=self.x0Class1, x1Train=self.x1Class1, sample0=self.testX0[i], sample1=self.testX1[i], h=0.6, sigma=0.6)
      prob2 = self.gaussian_kernel(x0Train=self.x0Class2, x1Train=self.x1Class2, sample0=self.testX0[i], sample1=self.testX1[i], h=0.6, sigma=0.6)
      prob3 = self.gaussian_kernel(x0Train=self.x0Class3, x1Train=self.x1Class3, sample0=self.testX0[i], sample1=self.testX1[i], h=0.6, sigma=0.6)
      probs = [prob1, prob2, prob3]
      if self.labelTest[i] == (probs.index(max(probs))+1):
        cor += 1
    acc = (cor*100)/len(self.testX0)
    print ("accuracy test is ", acc)

    allTrainX0 = self.x0Class1.tolist() + self.x0Class2.tolist() + self.x0Class3.tolist()
    allTrainX1 = self.x1Class1.tolist() + self.x1Class2.tolist() + self.x1Class3.tolist()
    allTrainLabel = 450 * [1] + 450 * [2] + 450 * [3]
    cor = 0
    pred = []
    for i in range(len(allTrainX0)):
      prob1 = self.gaussian_kernel(x0Train=self.x0Class1, x1Train=self.x1Class1, sample0=allTrainX0[i], sample1=allTrainX1[i], h=0.6, sigma=0.6)
      prob2 = self.gaussian_kernel(x0Train=self.x0Class2, x1Train=self.x1Class2, sample0=allTrainX0[i], sample1=allTrainX1[i], h=0.6, sigma=0.6)
      prob3 = self.gaussian_kernel(x0Train=self.x0Class3, x1Train=self.x1Class3, sample0=allTrainX0[i], sample1=allTrainX1[i], h=0.6, sigma=0.6)
      probs = [prob1, prob2, prob3]
      if allTrainLabel[i] == (probs.index(max(probs))+1):
        cor += 1
    acc = (cor*100)/len(allTrainX0)
    print ("accuracy train is ", acc)
    
# mean0 = [2, 5]
# cov0 = [[2, 0], [0, 2]]  
# mean1 = [8, 1]
# cov1 = [[3, 1], [1, 3]]
# mean2 = [5, 3]
# cov2 = [[2, 1], [1, 2]]

# gb = GaussianBayesian(mean0, cov0, mean1, cov1, mean2, cov2)
# gb.classifier()

class DensityEstimator:
  def __init__(self, mean0, cov0, numData):
    self.x0, self.x1 = np.random.multivariate_normal(mean0, cov0, numData).T
    self.mu = mean0
    self.sigma = cov0
    self.x0Train = self.x0[:400]
    self.x1Train = self.x1[:400]
    self.x0Test = self.x0[400:]
    self.x1Test = self.x1[400:]
    self.minX0 = min(self.x0Train)
    self.minX1 = min(self.x1Train)
    self.maxX0 = max(self.x0Train)
    self.maxX1 = max(self.x1Train)
    self.numTest = 100
    self.numTrain = 400

  def plot(self, x0, x1, Px, title):
    fig = plt.figure(figsize=plt.figaspect(0.3))
    Px_plot = fig.add_subplot(1, 3, 1, projection='3d')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.view_init(0, 0)
    Px_plot.set_xlabel('x0')
    Px_plot.set_ylabel('x1')
    Px_plot.set_zlabel("Px")

    Px_plot = fig.add_subplot(1, 3, 2, projection='3d')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.view_init(0, 90)
    Px_plot.set_xlabel('x0')
    Px_plot.set_ylabel('x1')
    Px_plot.set_zlabel("Px")
    Px_plot.set_title(title)

    Px_plot = fig.add_subplot(1, 3, 3, projection='3d')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.plot_surface(x0, x1, Px, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    Px_plot.view_init(45, 45)
    Px_plot.set_xlabel('x0')
    Px_plot.set_ylabel('x1')
    Px_plot.set_zlabel("Px")
    plt.show()

  def countHistogramGrid(self, binWidth):
    nx0 = int((self.maxX0 - self.minX0) // binWidth) + 1
    nx1 = int((self.maxX1 - self.minX1) // binWidth) + 1
    count = [ [0] * nx1 for _ in range(nx0)]
    for i in range(len(self.x0Train)):
      x = self.x0Train[i]
      y = self.x1Train[i]
      x0Index = int((x - self.minX0) // binWidth )
      x1Index = int((y - self.minX1) // binWidth )
      count[x0Index][x1Index] += 1
    return count

  def getPHistogram(self, x, y, binWidth, count):
      x0Index = int((x - self.minX0) // binWidth )
      x1Index = int((y - self.minX1) // binWidth )
      return(count[x0Index][x1Index] / (self.numTest * binWidth*10))

  def histogram(self, binWidth):
    countGrid = self.countHistogramGrid(binWidth)
    x0 = np.linspace(self.minX0, self.maxX0, self.numTest)
    x1 = np.linspace(self.minX1, self.maxX1, self.numTest)
    x0, x1 = np.meshgrid(x0, x1)
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        z = self.getPHistogram(x0[i][j], x1[i][j], binWidth, countGrid)
        zRow += [z]
      zValue += [zRow]
    Px = np.array(zValue)
    title = "histogram with bin width: " + str(binWidth)
    self.plot(x0, x1, Px, title)

  def kParzenWindow(self, u):
    if abs(u) < 1/2:
      return 1
    return 0

  def kGaussiKernel(self, u, sigma):
    return(((sigma*(2*np.pi)**0.5)**-1) * np.exp(-0.5 * u**2) )

  def newK(self, u, sigma):
    a = u * sigma * u
    return ((2*np.pi)**-1) * np.exp(-0.5 * a)

  def getPParzenWindow(self, sample0, sample1, h, isGaussi, sigma):
    sumKD0 = 0
    sumKD1 = 0
    for i in range(len(self.x0Train)):
      if isGaussi:
        sumKD0 += self.newK((sample0 - self.x0Train[i])/h, sigma)
        sumKD1 += self.newK((sample1 - self.x1Train[i])/h, sigma)
      else:
        sumKD0 += self.kParzenWindow((sample0 - self.x0Train[i])/h)
        sumKD1 += self.kParzenWindow((sample1 - self.x1Train[i])/h)
    sumKD0 = (sumKD0 / (self.numTrain * h**2 *10)) /2
    sumKD1 = (sumKD1 / (self.numTrain * h**2 *10)) /2
    return (sumKD0 * sumKD1)

  def parzenWindow(self, h, isGaussi, sigma):
    x0 = np.linspace(self.minX0, self.maxX0, self.numTest)
    x1 = np.linspace(self.minX1, self.maxX1, self.numTest)
    x0, x1 = np.meshgrid(x0, x1)
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        z = self.getPParzenWindow(x0[i][j], x1[i][j], h, isGaussi, sigma)
        zRow += [z]
      zValue += [zRow]
    Px = np.array(zValue)
    if isGaussi:
      title = "gaussi kernel with h: " + str(h) + " and sigma: " + str(sigma)
    else:
      title = "parzen window with h: " + str(h)
    self.plot(x0, x1, Px, title)

  def getPKnn(self, sample0, sample1, k):
    distances = []
    for i in range(len(self.x0Train)):
      d = ((sample0 - self.x0Train[i])**2 + (sample1 - self.x1Train[i])**2 )**0.5
      distances.append(d)
    distances.sort()
    R = distances[k-1]
    v = np.pi * R**2
    px = k/(self.numTrain * v)
    return px

  def kNearestNeighbour(self, k):
    x0 = np.linspace(self.minX0, self.maxX0, self.numTest)
    x1 = np.linspace(self.minX1, self.maxX1, self.numTest)
    x0, x1 = np.meshgrid(x0, x1)
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        z = self.getPKnn(x0[i][j], x1[i][j], k)
        zRow += [z]
      zValue += [zRow]
    Px = np.array(zValue)
    title = "k nearest neighbour with k: " + str(k)
    self.plot(x0, x1, Px, title)

  def multivariate_gaussian(self, pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-1 *fac / 2) / N

  def realDensity(self):
    x0 = np.linspace(self.minX0, self.maxX0, self.numTest)
    x1 = np.linspace(self.minX1, self.maxX1, self.numTest)
    x0, x1 = np.meshgrid(x0, x1)
    zValue = []
    for i in range(len(x0)):
      zRow = []
      for j in range(len(x0[i])):
        z = self.multivariate_gaussian([ x0[i][j], x1[i][j] ], np.array(self.mu), np.array(self.sigma))
        zRow += [z]
      zValue += [zRow]
    Px = np.array(zValue)
    title = "real gaussian pdf"
    self.plot(x0, x1, Px, title)

  
  def gaussian_kernel(self, x0Train, x1Train, sample0, sample1, h, sigma):
    sumKD0 = 0
    sumKD1 = 0
    numTrain = len(x0Train)
    for i in range(len(x0Train)):
      sumKD0 += self.newK((sample0 - x0Train[i])/h, sigma)
      sumKD1 += self.newK((sample1 - x1Train[i])/h, sigma)

    sumKD0 = (sumKD0 / (100 * h**2)) / 2
    sumKD1 = (sumKD1 / (100 * h**2)) / 2
    return (sumKD0 * sumKD1)
  
  def getAccuracy(self, x0Test, x1Test, x0Train, x1Train, sigma, h):
    mean0 = sum(x0Train) / len(x0Train)
    mean1 = sum(x1Train) / len(x1Train)
    mu = [mean0, mean1]
    demeanX = []
    for i in range(len(x0Train)):
      temp = []
      temp.append(self.x0Train[i] - mean0)
      temp.append(self.x1Train[i] - mean1)
      demeanX.append(temp)
    demeanX = np.array(demeanX)
    demeanX = demeanX/np.linalg.norm(demeanX, ord=2, axis=1, keepdims=True)
    cov = (1/len(x0Train)) * (demeanX.T @ demeanX)

    diff = 0
    for i in range(len(x0Test)):
      realDensity = self.multivariate_gaussian([ x0Test[i], x1Test[i] ], np.array(mu), np.array(cov))
      nonParamGaussiDesnsity = self.gaussian_kernel(x0Train, x1Train, x0Test[i], x1Test[i], h, sigma)
      # print(realDensity, nonParamGaussiDesnsity)
      diff += ((realDensity - nonParamGaussiDesnsity) **2)
    return diff

  def kfold(self, h):
    sigma = 0.6
    acc1 = self.getAccuracy(self.x0[:100], self.x1[:100], self.x0[100:], self.x1[100:], sigma, h)
    temp0 = np.append(self.x0[200:], self.x0[:100])
    temp1 = np.append(self.x1[200:], self.x1[:100]) 
    acc2 = self.getAccuracy(self.x0[100:200], self.x1[100:200], temp0, temp1, sigma, h)
    temp0 = np.append(self.x0[300:], self.x0[:200])
    temp1 = np.append(self.x1[300:], self.x1[:200])  
    acc3 = self.getAccuracy(self.x0[200:300], self.x1[200:300], temp0, temp1, sigma, h)
    temp0 = np.append(self.x0[400:], self.x0[:300])
    temp1 = np.append(self.x1[400:], self.x1[:300])  
    acc4 = self.getAccuracy(self.x0[300:400], self.x1[300:400], temp0, temp1, sigma, h)
    acc5 = self.getAccuracy(self.x0[400:], self.x1[400:], self.x0[:400], self.x1[:400], sigma, h)
    return (acc1+acc2+acc3+acc4+acc5)/5

  def find_best_h(self):
    h = [0.6, 0.3, 0.09]
    error = []
    error.append(de.kfold(h[0]))
    error.append(de.kfold(h[1]))
    error.append(de.kfold(h[2]))
    indexMinH = error.index(min(error))
    return(h[indexMinH])


mean = [2, 5]
cov = [[2, 0], [0, 2]]  
de = DensityEstimator(mean, cov, 500)
# de.histogram(0.09)
# de.histogram(0.3)
# de.histogram(0.6)
# de.parzenWindow(0.09, False, 0.09)
# # de.parzenWindow(0.3, False, 0.3)
# de.parzenWindow(0.6, False, 0.6)
# de.parzenWindow(0.09, True, 0.2)
# de.parzenWindow(0.09, True, 0.6)
# de.parzenWindow(0.09, True, 0.9)
# de.parzenWindow(0.3, True, 0.2)
# de.parzenWindow(0.3, True, 0.6)
# de.parzenWindow(0.3, True, 0.9)
# de.parzenWindow(0.6, True, 0.2)
# de.parzenWindow(0.6, True, 0.6)
# de.parzenWindow(0.6, True, 0.9)
# de.kNearestNeighbour(1)
# de.kNearestNeighbour(99)
# de.realDensity()
# print(de.find_best_h())



      
