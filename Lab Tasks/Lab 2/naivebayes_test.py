import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naivebayes import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy



iris = datasets.load_iris()

#df = pd.DataFrame(iris.data, columns=iris.feature_names) # Haven't added column names.
#df['category'] = iris.target
#df['category'] = df['category'].replace( { 0 : iris.target_names[0], 1 : iris.target_names[1], 2 : iris.target_names[2] })

#y = iris.target
#X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))