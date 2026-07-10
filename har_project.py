"""
Human Activity Recognition with Smartphones
============================================
Classify 30 volunteers' daily activities (6 classes) from 561 sensor-derived
features captured by a waist-mounted smartphone's accelerometer and gyroscope.

Pipeline:
  1. Load & explore data (EDA)
  2. Preprocess (scaling, label encoding)
  3. Train & compare several classifiers
  4. Evaluate the best model on the held-out test set
  5. Save figures and a metrics report
"""

import os
import time
import warnings

# joblib/loky fails when the system temp path contains non-ASCII characters,
# so redirect all temp folders to an ASCII-only location before importing them.
_ASCII_TMP = "C:\\har_tmp"
os.makedirs(_ASCII_TMP, exist_ok=True)
for _v in ("TMP", "TEMP", "JOBLIB_TEMP_FOLDER"):
    os.environ[_v] = _ASCII_TMP

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data():
    train = pd.read_csv(os.path.join(HERE, "train.csv"))
    test = pd.read_csv(os.path.join(HERE, "test.csv"))
    log(f"Train shape: {train.shape}   Test shape: {test.shape}")
    return train, test


# ----------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ----------------------------------------------------------------------
def run_eda(train):
    log("\n=== EDA ===")
    log(f"Missing values in train: {train.isna().sum().sum()}")
    log(f"Duplicate rows in train: {train.duplicated().sum()}")
    log(f"Number of features: {train.shape[1] - 2}")
    log(f"Number of subjects: {train['subject'].nunique()}")
    log("\nActivity distribution:\n" + str(train["Activity"].value_counts()))

    # Class distribution
    plt.figure(figsize=(8, 5))
    order = train["Activity"].value_counts().index
    sns.countplot(y="Activity", data=train, order=order, palette="viridis")
    plt.title("Activity class distribution (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "01_class_distribution.png"), dpi=120)
    plt.close()

    # Samples per subject
    plt.figure(figsize=(10, 5))
    sns.countplot(x="subject", data=train, palette="mako")
    plt.title("Number of samples per subject (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "02_samples_per_subject.png"), dpi=120)
    plt.close()

    # A discriminative feature: gravity mean on X separates static vs dynamic
    if "tBodyAccMag-mean()" in train.columns:
        feat = "tBodyAccMag-mean()"
        plt.figure(figsize=(9, 5))
        sns.boxplot(x="Activity", y=feat, data=train, order=order, palette="Set2")
        plt.xticks(rotation=30, ha="right")
        plt.title(f"{feat} by activity")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "03_feature_by_activity.png"), dpi=120)
        plt.close()

    log("EDA figures saved to outputs/")


# ----------------------------------------------------------------------
# 3. Preprocessing
# ----------------------------------------------------------------------
def preprocess(train, test):
    drop_cols = ["subject", "Activity"]
    X_train = train.drop(columns=drop_cols).values
    X_test = test.drop(columns=drop_cols).values

    le = LabelEncoder()
    y_train = le.fit_transform(train["Activity"])
    y_test = le.transform(test["Activity"])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, le


# ----------------------------------------------------------------------
# 4. PCA visualization
# ----------------------------------------------------------------------
def pca_plot(X_train_s, y_train, le):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    comp = pca.fit_transform(X_train_s)
    plt.figure(figsize=(9, 7))
    for cls in np.unique(y_train):
        mask = y_train == cls
        plt.scatter(comp[mask, 0], comp[mask, 1], s=8, alpha=0.5,
                    label=le.inverse_transform([cls])[0])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (2 components) of scaled features")
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "04_pca_scatter.png"), dpi=120)
    plt.close()
    log(f"PCA explained variance (2 comp): {pca.explained_variance_ratio_.sum():.3f}")


# ----------------------------------------------------------------------
# 5. Model training & comparison
# ----------------------------------------------------------------------
def train_models(X_train_s, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0,
                                                  random_state=RANDOM_STATE),
        "LinearSVC": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
        "RBF_SVM": SVC(kernel="rbf", C=10, gamma="scale",
                       random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200,
                                               random_state=RANDOM_STATE,
                                               n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    log("\n=== 5-fold CV on training set ===")
    for name, model in models.items():
        t0 = time.time()
        scores = cross_val_score(model, X_train_s, y_train, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        results[name] = (scores.mean(), scores.std())
        log(f"{name:20s} acc = {scores.mean():.4f} +/- {scores.std():.4f}"
            f"   ({time.time()-t0:.1f}s)")

    # Bar chart of CV results
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.figure(figsize=(9, 5))
    plt.barh(names, means, xerr=stds, color=sns.color_palette("crest", len(names)))
    plt.xlim(0.85, 1.0)
    plt.xlabel("CV accuracy")
    plt.title("Model comparison (5-fold CV)")
    for i, m in enumerate(means):
        plt.text(m + 0.001, i, f"{m:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "05_model_comparison.png"), dpi=120)
    plt.close()

    best_name = max(results, key=lambda k: results[k][0])
    log(f"\nBest CV model: {best_name}")
    return models, best_name, results


# ----------------------------------------------------------------------
# 6. Final evaluation on test set
# ----------------------------------------------------------------------
def evaluate(model, best_name, X_train_s, y_train, X_test_s, y_test, le):
    log(f"\n=== Training {best_name} on full train set and evaluating on test ===")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    log(f"Test accuracy : {acc:.4f}")
    log(f"Test macro-F1 : {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    log("\nClassification report:\n" + report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix - {best_name} (test acc={acc:.3f})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "06_confusion_matrix.png"), dpi=120)
    plt.close()

    # Save text report
    with open(os.path.join(OUT, "metrics_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro-F1: {f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
    return acc, f1


def main():
    train, test = load_data()
    run_eda(train)
    X_train_s, X_test_s, y_train, y_test, le = preprocess(train, test)
    pca_plot(X_train_s, y_train, le)
    models, best_name, _ = train_models(X_train_s, y_train)
    evaluate(models[best_name], best_name, X_train_s, y_train,
             X_test_s, y_test, le)
    log("\nDone. All outputs in: " + OUT)


if __name__ == "__main__":
    main()
