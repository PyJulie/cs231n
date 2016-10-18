import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  参数说明:
    W: numpy生成的权重矩阵，大小为[D,C]
    X: numpy生成的图像训练集矩阵，大小为[D,N]，其中N表示样本的数量，D表示样本的维度
    y: numpy生成的标签，为训练集分类
    reg: 正则化

  返回的元组:
    单浮点型的损失
    遵从权重得到的梯度，一个和W维度相同的数组

  """
  dW = np.zeros(W.shape) #初始化梯度，全部置0

  #计算损失和梯度
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    count = 0
    index = 0
    for j in xrange(num_classes):
      if j == y[i]:
        index = j
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count += 1
        dW[j, :] += X[:, i]
        loss += margin
      else:
        dW[j, :] += np.array([0] * X.shape[0])
    dW[index, :] += - count * X[:, i] 

  #至今所有训练集的损失已经全部加和，现在让它变成平均值
  #全部除以一个训练数量
  loss /= num_train
  dW /= num_train

  #正则化损失
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #计算梯度损失并把它存放在dW中
  #首先计算损失，然后计算导数
  #计算损失的同时计算导数可能会简单一点
  #最后你可能需要修改一些代码去计算梯度

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  创建SVM损失函数，实现向量化，参数和返回值都与naive相同
  """
  loss = 0.0
  dW = np.zeros(W.shape) #初始化梯度，全部置0


  # 实现一个向量化版本的结构化SVM损失,存储损失的结果
  num_classes = W.shape[0]
  num_train = X.shape[1]

  scores = W.dot(X)
  correct_class_scores = scores[y, range(num_train)]
  margins = np.maximum(0, scores - correct_class_scores + 1.0)
  margins[y, range(num_train)] = 0

  loss_cost = np.sum(margins) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost + loss_reg
  num_pos = np.sum(margins > 0, axis=0) # 正向损失的数量

  #实现一个向量化版本的结构SVM损失梯度，把结果存储在dW里

  dscores = np.zeros(scores.shape)
  dscores[margins > 0] = 1
  dscores[y, range(num_train)] = -num_pos

  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW
