DATA_PATH = "data/raw/ACME-HappinessSurvey2020.csv"

TARGET = "Y"

FEATURES = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6"
]

RANDOM_STATE = 42
TEST_SIZE = 0.15

MODEL_PATHS = {
    "Logistic Regression": "models/lr_model.pkl",
    "Gaussian NB": "models/gnb_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Gradient Boosting": "models/gb_model.pkl",
    "XGBoost": "models/xgb_model.pkl",
    "KNN": "models/knn_model.pkl",
    "SVC": "models/svc_model.pkl",
    "Decision Tree": "models/dt_model.pkl",
}