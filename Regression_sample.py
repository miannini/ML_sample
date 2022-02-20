############### 1st Part ############################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data  #(use when text variables in X)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #change country column pos 1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #change Sex  column pos 2
onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy variable for country
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #remove first column "dummy variable" to omit the dummy trap

#another encoding option
#from sklearn.compose import ColumnTransformer
#From sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') #column [3] only
#X = np.array(ct.fit_transform(X)) #convert to np.array because will be used for the models later
#print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling                               #(depending on the model, some require some not)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  #trained scale on "train"
X_test = sc_X.transform(X_test)        #scale test, based on "train" scaling
#if y requires to be scaled
y = sc_y.fit_transform(y.reshape(-1,1)) #reshape is required when only 1 variable


############### 2nd Part ############################
#fitting the Model (choose one by one and test - restart kernel each time)
#Simple Linear Regression  #(doesn’t requires standardization of variables)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train

# Multiple Linear Regression #(doesn’t requires standardization of variables)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Polynomial Regression #(doesn’t requires standardization of variables)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #degree = power 2
X_poly = poly_reg.fit_transform(X) #new dataset with variables (power 2)
regressor = LinearRegression()
regressor.fit(X_poly, y)

#SVR   #(requires standardization of variables)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Decision tree regressor  #(doesn’t requires standardization of variables) / plot with grid
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Random Forest regressor   #(doesn’t requires standardization of variables) / plot with grid
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

############### 3rd Part ############################
#Prediction and metrics
#Prediction
y_pred = regressor.predict(X_test)
#in case of y scaled
y_pred = sc_y.inverse_transform(y_pred)

#metrics

###############################################################
#3.b improve
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# tunning - Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, #C = penalty parameter
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.0001,0.001,0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = regressor,
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
