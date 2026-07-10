"""
Combine all generated charts into a single PDF report.
Output: charts_report.pdf
"""

import os
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(HERE, "charts")
PDF_PATH = os.path.join(HERE, "charts_report.pdf")

TITLES = {
    "chart_01_class_distribution.png": "1. Activity Class Distribution",
    "chart_02_static_vs_dynamic.png": "2. Static vs Dynamic Activities",
    "chart_03_samples_per_subject.png": "3. Samples per Subject",
    "chart_04_feature_boxplots.png": "4. Signal-Magnitude Features by Activity",
    "chart_05_gravity_angle_kde.png": "5. Gravity-Angle Distribution",
    "chart_06_correlation_heatmap.png": "6. Feature Correlation Heatmap",
    "chart_07_pca_scatter.png": "7. PCA Projection (2D)",
    "chart_08_pca_variance.png": "8. PCA Cumulative Explained Variance",
    "chart_09_tsne_scatter.png": "9. t-SNE Projection (2D)",
}


def main():
    files = sorted(glob.glob(os.path.join(CHARTS, "*.png")))
    if not files:
        raise SystemExit("No charts found in " + CHARTS)

    with PdfPages(PDF_PATH) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.62, "Human Activity Recognition",
                 ha="center", size=22, weight="bold")
        fig.text(0.5, 0.56, "with Smartphones", ha="center", size=18)
        fig.text(0.5, 0.48, "Data Visualization Report", ha="center", size=14)
        fig.text(0.5, 0.10, f"{len(files)} charts", ha="center",
                 size=11, style="italic", color="gray")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # One chart per page
        for f in files:
            img = mpimg.imread(f)
            h, w = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(TITLES.get(os.path.basename(f), os.path.basename(f)),
                         size=13, weight="bold", y=0.96)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "HAR Data Visualization Report"
        d["Subject"] = "Human Activity Recognition with Smartphones"

    print("PDF saved to:", PDF_PATH, flush=True)


if __name__ == "__main__":
    main()
