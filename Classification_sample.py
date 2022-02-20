############### 1st Part ############################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv') #change name of file
X = dataset.iloc[:, 3:-1].values  #change columns of X
y = dataset.iloc[:, -1].values #change column of Y (normally last, but could be first)


# Encoding categorical data  #(use when text variables in X)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #change country column pos 1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #change Sex  column pos 2
onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy variable for country
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #remove first column "dummy variable" to omit the dummy trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling                               #(depending on the model, some require some not)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  #trained scale on "train"
X_test = sc_X.transform(X_test)        #scale test, based on "train" scaling


############### 2nd Part ############################
#fitting the Model (choose one by one and test - restart kernel each time)
#Logistic Regression #(requires standardization of variables)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Knn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #p=2 is euclidean distance
classifier.fit(X_train, y_train)

#SVM  #(requires standardization of variables)
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0) #linear or rbf
classifier.fit(X_train, y_train)

#Naive Bayes #(requires standardization of variables)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#decision tree #(doesn’t requires standardization of variables)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

#Random Forest #(doesn’t requires standardization of variables)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0) #entropy or gini
classifier.fit(X_train, y_train)

#Xgboost classification #(doesn’t requires standardization of variables)
from xgboost import XGBClassifier
classifier = XGBClassifier()  #objective='multi:softmax' for multiclasses
classifier.fit(X_train, y_train)



############### 3rd Part ############################
#Prediction and metrics
#Prediction
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sum_first_diagonal = sum(cm[i][i] for i in range(len(cm)))
accuracy = sum_first_diagonal/len(X_test)

###############################################################
#3.b improve
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# tunning - Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, #C = penalty parameter
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.0001,0.001,0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

#after get the best, implement that model with best_parameters


############### 4th Part ############################
# Visualization
# Training results for 2D problems
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     # -1 and +1 to enlarge range and include all posible points
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1,
                               step=0.01))  # 0.01 resolution of pixels to make "continuous"
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')  # model name
plt.xlabel('Age')  # 1st col name
plt.ylabel('Estimated Salary')  # 2nd col name
plt.legend()
plt.show()

# Test results for 2D problems
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')  # model name
plt.xlabel('Age')  # 1st col name
plt.ylabel('Estimated Salary')  # 2nd col name
plt.legend()
plt.show()

# AUC - ROC plot
