from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle

def train_lr(X_train, y_train):

    lr = LogisticRegression(random_state=42) 
    lr.fit(X_train,y_train)

    with open('models/lr_model.pkl', 'wb') as file:
        pickle.dump(lr, file)
    return lr


def train_gnb(X_train, y_train):

    gnb = GaussianNB() 
    gnb.fit(X_train,y_train)

    with open('models/gnb_model.pkl', 'wb') as file:
        pickle.dump(gnb, file)
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
from predict import evaluate_models
from plots import correlation_matrix
from plots import generate_dt_plots
import argparse



def main():

    parser = argparse.ArgumentParser(description="Data Science Pipeline")

    parser.add_argument(
        "--mode", 
        choices=["train", "evaluate", "plot"], 
        required=True, 
        help="train = train models, evaluate = evaluate saved models, plot = generate visualizations"
    )
    
    args = parser.parse_args()

    #Load Data
    df = load_data(config.DATA_PATH)
    #Feature Selection
    X, y = select_features(df)
    #Make training/testing datasets
    X_train, X_test, y_train, y_test = separate_data(X, 
                                                     y, 
                                                     config.TEST_SIZE, 
                                                     config.RANDOM_STATE)

    #Train Models
    if args.mode == "train":
        trainers = [
            train_lr, #logistic regression
            train_gnb, #gaussian nb
            train_rf, #random forest
            train_gb, #gradient boost
            train_xgb, #xgboost
            train_knn, #k nearest neighbors
            train_support_vector, #support vector
            train_decision_tree, #decision tree
        ]
        for trainer in trainers:
            print(f"Training {trainer.__name__}...")
            trainer(X_train, y_train)
    
    #Load Models
    elif args.mode == "evaluate":
        results = evaluate_models(X_test, y_test, config.MODEL_PATHS)
        print(results)

    #Plot models (found in models directory)
    elif args.mode == "plot":
        #Save correlation Matrix
        correlation_matrix(df)
        #Save decision tree feature importance, map, confusion matrix
        generate_dt_plots(X, X_test, y_test, "models/dt_model.pkl")

if __name__ == "__main__":
    main()
