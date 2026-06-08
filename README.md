# HappyCustomers

This project aims to take customer feedback and create a predictive model, labeling a customer as "unhappy" or "happy" based on 6 features:

1. my order was delivered on time
2. contents of my order was as I expected
3. I ordered everything I wanted to order
4. I paid a good price for my order
5. I am satisfied with my courier
6. the app makes ordering easy for me

These features are rated 1 to 5 (5 = Agree, 1 = Disagree).

## Running the Project

This project uses a command-line interface to train models, evaluate saved models, and generate visualizations.

### Train Models

To train all machine learning models and save them to the `models/` directory:

```bash
python src/main.py --mode train
```

### Evaluate Models

To load the saved models and display their F1 scores on the test dataset:

```bash
python src/main.py --mode evaluate
```

This command evaluates the following classifiers:

- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree

### Generate Visualizations

To create and save project visualizations:

```bash
python src/main.py --mode plot
```

This command generates:

- Correlation Matrix
- Decision Tree Visualization
- Decision Tree Feature Importance Plot
- Decision Tree Confusion Matrix

Generated figures are saved to the `reports/figures/` directory.

### Prerequisites

Install all required dependencies before running the project:

```bash
pip install -r requirements.txt
```

### Project Workflow

A typical workflow is:

1. Train models

```bash
python src/main.py --mode train
```

2. Evaluate model performance

```bash
python src/main.py --mode evaluate
```

3. Generate visualizations

```bash
python src/main.py --mode plot
```
