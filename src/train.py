from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import pickle

def train_lr(X_train, y_train):

    lr = LogisticRegression(random_state=42) 
    lr.fit(X_train,y_train)

    return lr


def train_gnb(X_train, y_train):

    gnb = GaussianNB() 
    gnb.fit(X_train,y_train)

    return gnb

def train_rf(X_train, y_train):

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state = 42),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/rf_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_


def train_gb(X_train, y_train):

    param_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state = 42),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/gb_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_


def train_xgb(X_train, y_train):

    param_grid = {
        "max_depth": [2, 3, 4],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
        "reg_alpha": [0, 1],
        "reg_lambda": [1, 10],
        "gamma": [0, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state = 42, n_jobs = 1),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/xgb_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_

def train_knn(X_train, y_train):

    param_grid = {
        'n_neighbors': range(1, 31, 2),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(metric="minkowski"),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/knn_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_


def train_support_vector(X_train, y_train):

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/svc_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_


def train_decision_tree(X_train, y_train):

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [5, 7, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    with open('models/dt_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return grid_search.best_estimator_


import config
from dataset import load_data
from dataset import separate_data
from features import select_features
from predict import load_model
from predict import predict
from plots import correlation_matrix
from plots import plot_feature_importance
from plots import plot_dt
from plots import plot_confusion_matrix


def main():
    df = load_data(config.DATA_PATH)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = separate_data(X, y, config.TEST_SIZE, config.RANDOM_STATE)
    
    #Save correlation Matrix
    correlation_matrix(df)

    #Logistic Regression Classifier
    model = train_lr(X_train, y_train)
    y_pred = predict(model, X_test)
    print("Test f1 Logistic Regression:", f1_score(y_test, y_pred))

    #Gaussian NB Classifier
    model = train_gnb(X_train, y_train)
    y_pred = predict(model, X_test)
    print("Test f1 Gaussian NB:", f1_score(y_test, y_pred))

    #Random Forest Classifier
    #model = train_rf(X_train, y_train)
    model = load_model("models/rf_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 Random Forest Classifier:", f1_score(y_test, y_pred))

    #Gradient Boosting Classifier
    #model = train_gb(X_train, y_train)
    model = load_model("models/gb_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 Gradient Boosting:", f1_score(y_test, y_pred))

    #XGBoost Classifier
    #model = train_xgb(X_train, y_train)
    model = load_model("models/xgb_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 XGBoost:", f1_score(y_test, y_pred))

    #KNN Classifier
    #model = train_knn(X_train, y_train)
    model = load_model("models/knn_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 KNN:", f1_score(y_test, y_pred))

    #Support Vector Classifier
    #model = train_support_vector(X_train, y_train)
    model = load_model("models/svc_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 Support Vector Classifier:", f1_score(y_test, y_pred))
    
    #Decision Tree
    #model = train_decision_tree(X_train, y_train)
    model = load_model("models/dt_model.pkl")
    y_pred = predict(model, X_test)
    print("Test f1 Decision Tree:", f1_score(y_test, y_pred))

    #Plot decision tree feature importance (found in figures folder)
    plot_feature_importance(X, model, "DecisionTreeFeatureImportance")
    #Plot decision tree map
    plot_dt(X, model, "DecisionTree")
    #Plot decision tree confusion matrix
    plot_confusion_matrix(y_test, y_pred, model, "DecisionTree")
if __name__ == "__main__":
    main()