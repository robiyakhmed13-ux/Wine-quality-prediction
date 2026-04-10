# =============================================================================
# Wine Quality Prediction using Random Forest Classifier
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# 1. Data Collection & Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the wine quality dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Generate EDA visualizations: quality distribution, barplots, correlation heatmap."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wine Quality – Exploratory Data Analysis", fontsize=16)

    # Quality distribution
    sns.countplot(x="quality", data=df, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Quality Score Distribution")
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Count")

    # Volatile acidity vs Quality
    sns.barplot(x="quality", y="volatile acidity", data=df, ax=axes[1], palette="Reds_d")
    axes[1].set_title("Volatile Acidity vs Quality")

    # Citric acid vs Quality
    sns.barplot(x="quality", y="citric acid", data=df, ax=axes[2], palette="Greens_d")
    axes[2].set_title("Citric Acid vs Quality")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")

    # Correlation Heatmap
    plt.figure(figsize=(10, 10))
    correlation = df.corr()
    sns.heatmap(
        correlation,
        square=True,
        fmt=".2f",
        annot=True,
        annot_kws={"size": 8},
        cmap="Blues"
    )
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")


# =============================================================================
# 3. Data Preprocessing & Label Binarization
# =============================================================================

def preprocess_data(df: pd.DataFrame):
    """
    Separate features and binarize the quality label:
        quality >= 7  →  1  (Good quality)
        quality <  7  →  0  (Not good quality)
    """
    X = df.drop(columns="quality", axis=1)
    Y = df["quality"].apply(lambda val: 1 if val >= 7 else 0)

    print(f"Feature shape : {X.shape}")
    print(f"Label distribution:\n{Y.value_counts().rename({0: 'Not Good (0)', 1: 'Good (1)'})}")
    return X, Y


# =============================================================================
# 4. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    """Split data into training and test sets (stratified)."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        stratify=Y,
        random_state=random_state
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 5. Model Training
# =============================================================================

def train_model(X_train, Y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    print("Model training complete.")
    return model


# =============================================================================
# 6. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> None:
    """Print accuracy scores, classification report, and confusion matrix."""
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_acc = accuracy_score(Y_train, train_preds)
    test_acc  = accuracy_score(Y_test,  test_preds)

    print(f"\nTraining Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy     : {test_acc:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(Y_test, test_preds, target_names=["Not Good", "Good"]))

    # Feature Importance Plot
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=True).plot(
        kind="barh", figsize=(8, 6), color="steelblue", title="Feature Importances"
    )
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.show()
    print("Feature importances saved as 'feature_importances.png'")

    # Confusion Matrix
    cm = confusion_matrix(Y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")


# =============================================================================
# 7. Predictive System
# =============================================================================

def predict_wine_quality(model, input_data: tuple) -> str:
    """
    Predict the quality class for a single wine sample.

    Parameters
    ----------
    model      : trained Random Forest model
    input_data : tuple of 11 numeric values:
        (fixed acidity, volatile acidity, citric acid, residual sugar,
         chlorides, free sulfur dioxide, total sulfur dioxide, density,
         pH, sulphates, alcohol)

    Returns
    -------
    str: prediction message
    """
    input_array    = np.asarray(input_data).reshape(1, -1)
    prediction     = model.predict(input_array)

    if prediction[0] == 1:
        return "🍷 This wine is predicted to be GOOD quality (score ≥ 7)."
    else:
        return "🍶 This wine is predicted to be NOT good quality (score < 7)."


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    # ── 1. Load ──────────────────────────────────────────────────────────────
    DATA_PATH = "winequality-red.csv"   # update path if needed
    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    plot_eda(df)

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    X, Y = preprocess_data(df)

    # ── 4. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    model = train_model(X_train, Y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # ── 7. Predict a sample wine ──────────────────────────────────────────────
    # Format: (fixed acidity, volatile acidity, citric acid, residual sugar,
    #          chlorides, free sulfur dioxide, total sulfur dioxide, density,
    #          pH, sulphates, alcohol)
    sample_input = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
    result = predict_wine_quality(model, sample_input)
    print(f"\nSample Prediction:\n{result}")
