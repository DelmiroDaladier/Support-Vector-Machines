import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target


svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)
svc2 = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X, y)

x_min, x_max = X[:,0].min() -1 , X[:,0].max() + 1
y_min, y_max = X[:,1].min() -1 , X[:,1].max() + 1
h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Training Set')

plt.plot()

plt.figure(1, figsize=(8,6))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal Lenght')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(),xx.max())
plt.title('SVC with RBF Kernel')
plt.show()

plt.figure(3, figsize=(8,6))

Z = svc2.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.contour


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal Lenght')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(),xx.max())
plt.title('SVC with Linear Kernel')
plt.show()