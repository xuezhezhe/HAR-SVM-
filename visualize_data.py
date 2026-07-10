"""
Human Activity Recognition - Data Visualization
===============================================
Generates a set of exploratory charts describing the dataset.
Outputs are saved to the `charts/` directory.
"""

import os
import warnings

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", font_scale=1.0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)

ACT_ORDER = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
             "SITTING", "STANDING", "LAYING"]
PALETTE = dict(zip(ACT_ORDER, sns.color_palette("tab10", len(ACT_ORDER))))


def save(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fig_name), dpi=130)
    plt.close()
    print("saved", fig_name, flush=True)


def main():
    train = pd.read_csv(os.path.join(HERE, "train.csv"))
    print("Loaded train:", train.shape, flush=True)

    # ------------------------------------------------------------------
    # 1. Class distribution (count + pie)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = train["Activity"].value_counts().reindex(ACT_ORDER)
    sns.barplot(x=counts.values, y=counts.index, ax=axes[0],
                palette=[PALETTE[a] for a in counts.index])
    axes[0].set_title("Activity sample counts")
    axes[0].set_xlabel("count")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 5, i, str(v), va="center")
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[PALETTE[a] for a in counts.index],
                startangle=90, wedgeprops=dict(width=0.45))
    axes[1].set_title("Activity proportion")
    save("chart_01_class_distribution.png")

    # ------------------------------------------------------------------
    # 2. Static vs Dynamic grouping
    # ------------------------------------------------------------------
    static = ["SITTING", "STANDING", "LAYING"]
    train["motion"] = train["Activity"].apply(
        lambda a: "Static" if a in static else "Dynamic")
    plt.figure(figsize=(7, 5))
    sns.countplot(x="motion", data=train, palette="Set2")
    plt.title("Static vs Dynamic activities")
    for i, v in enumerate(train["motion"].value_counts().reindex(
            ["Static", "Dynamic"]).values):
        plt.text(i, v + 5, str(v), ha="center")
    save("chart_02_static_vs_dynamic.png")

    # ------------------------------------------------------------------
    # 3. Samples per subject (stacked by activity)
    # ------------------------------------------------------------------
    ct = pd.crosstab(train["subject"], train["Activity"])[ACT_ORDER]
    ct.plot(kind="bar", stacked=True, figsize=(13, 6),
            color=[PALETTE[a] for a in ACT_ORDER])
    plt.title("Samples per subject (stacked by activity)")
    plt.ylabel("count")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save("chart_03_samples_per_subject.png")

    # ------------------------------------------------------------------
    # 4. Key magnitude features - boxplots
    # ------------------------------------------------------------------
    feats = ["tBodyAccMag-mean()", "tBodyGyroMag-mean()",
             "tBodyAccJerkMag-mean()", "fBodyAccMag-mean()"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, feat in zip(axes.ravel(), feats):
        sns.boxplot(x="Activity", y=feat, data=train, order=ACT_ORDER,
                    palette=[PALETTE[a] for a in ACT_ORDER], ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle("Signal-magnitude features by activity", fontsize=14)
    save("chart_04_feature_boxplots.png")

    # ------------------------------------------------------------------
    # 5. Distribution (KDE) of gravity angle - separates orientation
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for a in ACT_ORDER:
        sub = train[train["Activity"] == a]["angle(X,gravityMean)"]
        sns.kdeplot(sub, label=a, color=PALETTE[a], fill=False, linewidth=2)
    plt.title("Distribution of angle(X, gravityMean) by activity")
    plt.legend(fontsize=8)
    save("chart_05_gravity_angle_kde.png")

    # ------------------------------------------------------------------
    # 6. Correlation heatmap of top mean() features
    # ------------------------------------------------------------------
    mean_feats = [c for c in train.columns if c.endswith("mean()")][:25]
    corr = train[mean_feats].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
    plt.title("Correlation among 25 mean() features")
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    save("chart_06_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 7. PCA 2D
    # ------------------------------------------------------------------
    X = train.drop(columns=["subject", "Activity", "motion"]).values
    y = train["Activity"].values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(Xs)
    plt.figure(figsize=(10, 8))
    for a in ACT_ORDER:
        m = y == a
        plt.scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.5,
                    color=PALETTE[a], label=a)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA projection (2 components)")
    plt.legend(markerscale=2, fontsize=8)
    save("chart_07_pca_scatter.png")

    # ------------------------------------------------------------------
    # 8. PCA cumulative explained variance
    # ------------------------------------------------------------------
    pca_full = PCA(n_components=0.99, random_state=42).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(cum) + 1), cum, marker=".")
    plt.axhline(0.95, color="r", ls="--", label="95% variance")
    n95 = np.argmax(cum >= 0.95) + 1
    plt.axvline(n95, color="g", ls="--", label=f"{n95} components")
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    save("chart_08_pca_variance.png")

    # ------------------------------------------------------------------
    # 9. t-SNE 2D (subsample for speed)
    # ------------------------------------------------------------------
    idx = np.random.RandomState(42).choice(len(Xs), size=2000, replace=False)
    ts = TSNE(n_components=2, random_state=42, init="pca",
              perplexity=30).fit_transform(Xs[idx])
    plt.figure(figsize=(10, 8))
    ysub = y[idx]
    for a in ACT_ORDER:
        m = ysub == a
        plt.scatter(ts[m, 0], ts[m, 1], s=12, alpha=0.6,
                    color=PALETTE[a], label=a)
    plt.title("t-SNE projection (2000 samples)")
    plt.legend(markerscale=2, fontsize=8)
    save("chart_09_tsne_scatter.png")

    print("\nAll charts saved to:", OUT, flush=True)


if __name__ == "__main__":
    main()
