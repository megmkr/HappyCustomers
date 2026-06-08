import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from predict import load_model
from predict import predict

def correlation_matrix(df):

    sns.heatmap(
        df.corr(),
        annot=True
    )
    plt.savefig("reports/figures/CorrelationMatrix", dpi=300, bbox_inches="tight")


def plot_confusion_matrix(y_test, y_pred, model_path, title):
    model = load_model(model_path)
    cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels=['unhappy', 'happy'])
    disp.plot()
    plt.savefig("reports/figures/confusionMatrix"+title, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(df, model_path, title = "Feature Importance"):
    model = load_model(model_path)
    feat_importances = pd.Series(model.feature_importances_, index=df.columns)
    sorted_importances = feat_importances.nlargest(3).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 1.2)))

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    sorted_importances.plot(kind="barh", color="skyblue", ax=ax)
    fig.savefig("reports/figures/"+title, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_dt(X, model_path, title):
    model = load_model(model_path)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    plot_tree(model, filled=True, ax=ax, feature_names=X.columns)
    plt.savefig("reports/figures/"+title, bbox_inches='tight')


def generate_dt_plots(X, X_test, y_test, model_path):

    plot_feature_importance(
        X,
        model_path,
        "DecisionTreeFeatureImportance"
    )

    plot_dt(
        X,
        model_path,
        "DecisionTree"
    )

    y_pred = predict(model_path, X_test)

    plot_confusion_matrix(
        y_test,
        y_pred,
        model_path,
        "DecisionTree"
    )