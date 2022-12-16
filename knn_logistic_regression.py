# Develop a machine learning model that can predict whether people have diabetes when their characteristics are specified.

# In data set;
# The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

# Variables
#
# Pregnancies
# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI
# DiabetesPedigreeFunction
# Age
# Outcome


# Diabetes Prediction with Logistic Regression

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model


import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)


# 1. Exploratory Data Analysis - EDA

df = pd.read_csv('dataset/diabetes.csv')

df.head()
df.shape
df.describe().T
df['Outcome'].value_counts()


# 2. Data Preprocessing & Feature Engineering

y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

# In distance-based methods, variables need to be standardized.
X_scaled = StandardScaler().fit_transform(X)

# There is no column information, numpy array should be converted to dataframe.
X = pd.DataFrame(X_scaled, columns=X.columns)


# 3. Model & Prediction

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=23)  #choose a random user

knn_model.predict(random_user)


# 4. Model Evaluation

# y_pred for the confusion matrix (prediction for all observation units):
y_pred = knn_model.predict(X)

# y_probe for AUC:
y_prob = knn_model.predict_proba(X)[:, 1]   # 1.index values, probability of belonging to class 1

print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model,
                            X,
                            y,
                            cv=5,
                            scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()  # 0.73
cv_results['test_f1'].mean()        # 0.59
cv_results['test_roc_auc'].mean()   # 0.78

# to increase achievement scores;
# n_neighbors hypermarameter in KNN method can be changed.
knn_model.get_params()


# 5. Hyperparameter Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()

# The goal is to find the optimum value by changing the n_neighbors hypermarameter value!

knn_params = {'n_neighbors': range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                          knn_params,
                          cv=5,
                          n_jobs=-1,
                          verbose=1).fit(X, y)

knn_gs_best.best_params_


# 6. Final Model

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

# Now we have to look at the test error of the final model:

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()  # 0.77
cv_results['test_f1'].mean()        # 0.62
cv_results['test_roc_auc'].mean()   # 0.81

# prediction
random_user = X.sample(3)       # 3 random users
knn_final.predict(random_user)  # prediction of the probability that 3 random users will have diabetes




