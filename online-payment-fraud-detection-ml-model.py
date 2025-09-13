import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _(mo):
    mo.md(r"""# Import Packages""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, confusion_matrix
    return (
        ColumnTransformer,
        DecisionTreeClassifier,
        OneHotEncoder,
        Pipeline,
        accuracy_score,
        confusion_matrix,
        cross_validate,
        f1_score,
        mo,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Business Problem

    1. How to reduce fraud transaction in financial company?
    2. How to reduce revenue loss because of fraud transaction?
    3. How to automatically detect a fraudulent transaction?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Objective

    Build a classification machine learning model to predict whether the transaction is fraud or not.
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("online-payment-fraud-detection/online_payment_fraud.csv")

    df.head(10)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Missing Values""")
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Outliers""")
    return


@app.cell
def _(df):
    df_num = df.select_dtypes(["int64", "float64"])
    col_num = df_num.columns
    col_num
    return (col_num,)


@app.cell
def _(col_num, df):
    for col in col_num:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        high_fence = q3 + 1.5*iqr
        low_fence = q1 - 1.5*iqr
        outliers = df[(df[col] < low_fence) | (df[col] > high_fence)]
        print(col)
        print(outliers.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""there are lots of outliers in fraud cases, but special to fraud dataset the outliers maybe the character of the fraud itself""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Duplicated Values""")
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""# Exploratory Data Analysis""")
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    1. step column doesn't have meaningful value to determine wether the the record is fraud or not => remove
    2. nameOrig and nameDest too unique similar to primary_key => remove
    """
    )
    return


@app.cell
def _(df):
    df2 = df.drop(columns=["step", "nameOrig", "nameDest"])

    df2.head(10)
    return (df2,)


@app.cell
def _(mo):
    mo.md(r"""## Deciding Target isFraud or isFlaggedFraud""")
    return


@app.cell
def _(df):
    df["isFraud"].value_counts()
    return


@app.cell
def _(df):
    df["isFlaggedFraud"].value_counts()
    return


@app.cell
def _(df2):
    df3 = df2.drop(columns=["isFlaggedFraud"])

    df3.head(10)
    return (df3,)


@app.cell
def _(mo):
    mo.md(r"""## Univariate Analysis""")
    return


@app.cell
def _(df3):
    df3.describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    1. amount std is large, 3x of its mean
    2. oldbalanceDest std is 3x larger than its mean
    3. newbalanceDest std is 3x larger than its mean
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Insights**:    

    1. we can't use classification algorithm that assume the distribution is normal for example Logistic regression. We use Decision Tree, Random Forest, and XGBoost
    2. Because the data is too large, we dont plot the histogram to see the distribution of the data and scatterplot for multivariate analysis
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Multivariate Analysis""")
    return


@app.cell
def _(df3):
    df_num2 = df3.select_dtypes(["int64", "float64"])
    col_num2 = df_num2.columns.tolist()
    col_num2
    return (col_num2,)


@app.cell
def _(col_num2, df, sns):
    sns.heatmap(df[col_num2].corr(), annot=True)
    return


@app.cell
def _(mo):
    mo.md(r"""# Data Preprocessing""")
    return


@app.function
def preprocess_data(df):
    """
    Performs feature engineering, preprocessing, training, and evaluation of a
    Linear Regression model using a scikit-learn pipeline.

    Args:
        df (pd.DataFrame): The input dataframe containing the car data.
                           It is expected to have the original 26 features.

    Returns:
        tuple: A tuple containing:
            - Pipeline: The fitted scikit-learn pipeline object.
            - pd.DataFrame: X_test data.
            - pd.Series: y_test data.
    """


@app.cell
def _(mo):
    mo.md(r"""## Feature Engineering""")
    return


@app.cell
def _(df, df3):
    df3["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df3["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return


@app.cell
def _(df3):
    df3.head(10)
    return


@app.cell
def _(df3):
    df3.info()
    return


@app.cell
def _(df3):
    df3.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""# Modeling & Evaluation""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Decision Tree""")
    return


@app.cell
def _(
    ColumnTransformer,
    DecisionTreeClassifier,
    OneHotEncoder,
    Pipeline,
    accuracy_score,
    confusion_matrix,
    cross_validate,
    f1_score,
    np,
    plt,
    precision_score,
    recall_score,
    roc_auc_score,
    sns,
    train_test_split,
):
    def create_and_evaluate_Decision_Tree_model(df):
        """
        Performs preprocessing, training, and evaluation of a Decision Tree 
        model using a scikit-learn pipeline.

        Args:
            df (pd.DataFrame): The input dataframe containing features and the 
                               'isFraud' target variable.

        Returns:
            tuple: A tuple containing:
                - Pipeline: The fitted scikit-learn pipeline object.
                - np.ndarray: dtree_y_pred predictions on the test set.
                - pd.DataFrame: dtree_X_test data.
                - pd.Series: dtree_y_test data.
        """
        df_processed = df.copy()

        # Train Test Split
        X = df_processed.drop(columns=["isFraud"])
        y = df_processed["isFraud"]
        dtree_X_train, dtree_X_test, y_train, dtree_y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # Create a Scikit-learn Pipeline
        categorical_features = dtree_X_train.select_dtypes(include=['object', 'category']).columns
        numerical_features = dtree_X_train.select_dtypes(include=np.number).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        dtree = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # Modeling and Cross-Validation
        print("--- Cross-Validation Results (Decision Tree) ---")
        scoring_metrics = ['accuracy', 'recall', 'precision', 'f1']
        cv_results = cross_validate(dtree, dtree_X_train, y_train, cv=5, scoring=scoring_metrics)

        print(f"Average CV Accuracy: {np.mean(cv_results['test_accuracy']):.2f}")
        print(f"Average CV Recall: {np.mean(cv_results['test_recall']):.2f}")
        print(f"Average CV Precision: {np.mean(cv_results['test_precision']):.2f}")
        print(f"Average CV F1-Score: {np.mean(cv_results['test_f1']):.2f}")

        # Final Model Evaluation
        dtree.fit(dtree_X_train, y_train)
        dtree_y_pred = dtree.predict(dtree_X_test)
        y_prob = dtree.predict_proba(dtree_X_test)[:, 1]

        print("\n--- Final Model Evaluation on the Test Set (Decision Tree) ---")
        print(f"Accuracy: {accuracy_score(dtree_y_test, dtree_y_pred):.2f}")
        print(f"Recall: {recall_score(dtree_y_test, dtree_y_pred):.2f}")
        print(f"Precision: {precision_score(dtree_y_test, dtree_y_pred):.2f}")
        print(f"F1-Score: {f1_score(dtree_y_test, dtree_y_pred):.2f}")
        print(f"ROC AUC Score: {roc_auc_score(dtree_y_test, y_prob):.3f}")

        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(dtree_y_test, dtree_y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        return dtree, dtree_y_pred, dtree_X_test, dtree_y_test
    return (create_and_evaluate_Decision_Tree_model,)


@app.cell
def _(create_and_evaluate_Decision_Tree_model, df3):
    dtree, dtree_y_pred, dtree_X_test, dtree_y_test = create_and_evaluate_Decision_Tree_model(df3)
    return (dtree,)


@app.cell
def _(mo):
    mo.md(r"""# Evaluation""")
    return


@app.cell
def _(mo):
    mo.md(r"""NOTE: For negative cases like fraud and churn case, the really important metric is recall because the model need to capture how many fraud or churn that is actually a fraud or churn""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Final Model = Decision Tree

    1. Recall = 87%     => Out of all the actual fraud transactions that occurred, the model successfully identified 87% of them.
    2. Precision = 89%  => When the model predicts that a transaction is fraudulent, it is correct 89% of the time.
    3. in fraud case it is more tolerateable when non-fraud account/transaction predicted as fraud account/transaction than fraud account/transaction as non-fraud account/transaction
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Business Impact""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Assume for every 100 transactions, 5 will be fraud: 

    - Before the model:
          - The company dont know which transaction will be fraud, there is no fraud mitigation at all
          - Fraud cases = 5
          - Loss = 5 * 1Million = 5 Million Loss

    - After Modeling:
        - The company know which transaction more likely will be fraud
        - Fraud cases = 5 - (5 * Recall Score) = 5 - (5 * 0.87) = 1
        - Loss = 1 * 1Million = 1 Million Loss
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Export Selected Model""")
    return


@app.cell
def _(dtree):
    import joblib

    joblib.dump(dtree, './online-payment-fraud-detection/online-payment-fraud-detection-dtree.pkl')
    print("Model saved successfully!")
    return


@app.cell
def _(df3):
    fraud_counts_by_type = df3.groupby('type')['isFraud'].value_counts()
    print(fraud_counts_by_type)
    return


@app.cell
def _(mo):
    mo.md(r"""Note: In this dataset, fraudulent transactions only occur with 'TRANSFER' and 'CASH_OUT' types.""")
    return


if __name__ == "__main__":
    app.run()
