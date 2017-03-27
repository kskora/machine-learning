from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

# Print dataset description
# print(iris['DESCR'])

# shuffle and split data into training and test data sets
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
print("Training data set contains %i samples with %i features." % X_train.shape)
print("Test data set contains %i samples with %i features." % X_test.shape)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")

for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())

        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_xlabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i, j].set_visible(False)

# display created image
# plt.show()

# create and build machine learning model based on K nearest neighbours algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

import numpy as np
y_pred = knn.predict(X_test)

model_accuracy = knn.score(X_test, y_test)

print("Test set accuracy is: %3.2f%%" % (model_accuracy * 100))