import numpy as np

class NormalDistribution(object):
  def __init__(self, mean, covariance):
    dim = mean.shape[0]
    assert(mean.shape == (dim,))
    assert(covariance.shape == (dim,dim))
    assert(np.all(covariance.T == covariance))
    self.__mean = mean
    self.__covariance = covariance
    self.dim = dim

  def sample(self, n):
    (zvar, zrot) = np.linalg.eig(self.__covariance)
    return np.dot(np.random.randn(n ,self.dim)*np.sqrt(zvar),zrot) + self.__mean


class GMM(object):
  def __init__(self, n):
    self.__data = []
    self.__dists = []
    self.__n = n
    self.__ns = []
    self.__ratios = [1.]
