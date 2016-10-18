#coding:utf-8
import numpy as np

class KNearestNeighbor:
  """ 基于欧几里得距离的knn分类器 """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    训练分类器，对于k临近算法记忆训练集

    输入：
    X - 一个维度为x的数组，其中每一行都是训练点
    y - 一个长度与X相同的向量，其中y[i]是X[i,:]的分类标签

    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    使用分类器对测试集的进行预测和分类

    输入：
    X - 一个维度为x的数组，其中每一行都是测试点
    k - 用来投票的邻近值的数量（取奇数）
    num_loops - 决定了用哪种方式去计算训练点和测试点之间的距离

    输出：
    y - 一个长度与X相同的向量，其中y[i]是通过分类器预测出来的，对X[i,:]的分类标签（结果）
    
    """

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    通过一个嵌套循环，计算训练集和测试集之间的距离

    输入：
    X - 一个维度为x的数组，其中每一行都是测试点

    输出：
    dists - 计算出来的距离，为一个数组，长度为x

    """
    num_test = X.shape[0] #获得X的长度
    num_train = self.X_train.shape[0] #获得self.X的长度
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j, :] - X[i, :])))
        #计算距离
    return dists

  def compute_distances_one_loop(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis = 1))
    return dists

  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    test_square = np.array([np.sum(np.square(X), axis = 1)] * num_train).transpose()
    train_square = np.array([np.sum(np.square(self.X_train), axis = 1)] * num_test)
    dists = np.sqrt(X.dot(self.X_train.transpose()) * (-2) + test_square + train_square)
    return dists

  def predict_labels(self, dists, k=1):
    """
    传入计算好的数据，并对其进行预测

    输入：
    dists - 计算得出来的测试点与训练点之间的距离

    输出：
    y - 最终分类的结果

    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      closest_y = []
      #使用距离矩阵去寻找出训练集k个最邻近的点，使用y标签集去寻找这些临近点的分类，并存储在closest_y中
      indexs = np.argsort(dists[i, :])#argsort() 排序后，最后从小到大，返回索引值，比如[3,1,2]返回[1,2,0]
      #打印出 indexs[0:k]
      for w in range(0, k):
        closest_y.append(self.y_train[indexs[w]])
      #找出了k个最邻近点的分类后，在closest_y中找到最相同的分类，把这些分类存储在y_pred[i]中
      labels = []
      count = []
      count_index = -1
      prev = -1
      for j in np.sort(closest_y):
        if j != prev:
          count.append(1)
          count_index += 1
          prev = j
          labels.append(j)
        else:
          count[count_index] += 1
      y_pred[i] = labels[np.argmax(count)]


    return y_pred

