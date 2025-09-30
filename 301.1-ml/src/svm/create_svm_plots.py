import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, SVR, NuSVR
from sklearn.datasets import make_classification, make_regression
from sklearn.inspection import DecisionBoundaryDisplay


# Generate sample data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42,
)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Train SVM model
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# Create a mesh to plot the decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)

# Plot the dataset
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50)

# Highlight support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="k",
    linewidths=2,
    label="Support Vectors",
)

# Add labels and title
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("SVM Decision Boundary with Support Vectors")
ax.legend()

# Add margins
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res/svm/svm_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.show()

# Now create a plot showing the effect of different kernels
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
titles = [
    "Linear Kernel",
    "Polynomial Kernel (degree=3)",
    "RBF Kernel",
    "Sigmoid Kernel",
]

for i, (title, kernel) in enumerate(zip(titles, ["linear", "poly", "rbf", "sigmoid"])):
    # Generate non-linear data
    X2, y2 = make_classification(
        n_samples=150,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
        n_classes=2,
    )

    # Add some non-linearity
    X2 = np.c_[X2[:, 0], X2[:, 1] + (0.2 * X2[:, 0] ** 2)]

    ax = sub[i // 2, i % 2]

    # Train SVM with specified kernel
    clf = svm.SVC(kernel=kernel, gamma=1, C=1.0)
    clf.fit(X2, y2)

    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X2, plot_method="contour", levels=[0], alpha=0.4, ax=ax, colors=["k"]
    )

    # Plot dataset
    ax.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.coolwarm, s=50)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Highlight support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=2,
    )

plt.tight_layout()
plt.savefig("res/svm/svm_kernels_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Create a plot showing the effect of the C parameter
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
C_values = [0.01, 0.1, 1, 100]
titles_C = [
    "C = 0.01 (Soft Margin)",
    "C = 0.1",
    "C = 1 (Balanced)",
    "C = 100 (Hard Margin)",
]

for i, (C, title) in enumerate(zip(C_values, titles_C)):
    ax = sub[i // 2, i % 2]

    # Create dataset with some noise
    X3, y3 = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Add some noise to create overlapping classes
    X3[-5:] += np.random.uniform(-1, 1, (5, 2))

    # Train SVM with specified C value
    clf = svm.SVC(kernel="linear", C=C)
    clf.fit(X3, y3)

    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X3, plot_method="contour", levels=[0], alpha=0.4, ax=ax, colors=["k"]
    )

    # Plot dataset
    ax.scatter(X3[:, 0], X3[:, 1], c=y3, cmap=plt.cm.coolwarm, s=50)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Highlight support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=2,
    )

plt.tight_layout()
plt.savefig("res/svm/svm_c_parameter_effect.png", dpi=150, bbox_inches="tight")
plt.show()


# Create a plot showing the effect of gamma parameter for RBF kernel
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
gamma_values = [0.01, 0.1, 1, 10]
titles_gamma = [
    "Gamma = 0.01 (Low)",
    "Gamma = 0.1",
    "Gamma = 1 (Medium)",
    "Gamma = 10 (High)",
]

for i, (gamma, title) in enumerate(zip(gamma_values, titles_gamma)):
    # Create non-linear data
    X_gamma, y_gamma = make_classification(
        n_samples=150,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
        n_classes=2,
    )

    ax = sub[i // 2, i % 2]

    # Train SVM with specified gamma value
    clf = svm.SVC(kernel="rbf", gamma=gamma, C=1.0)
    clf.fit(X_gamma, y_gamma)

    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_gamma, plot_method="contour", levels=[0], alpha=0.4, ax=ax, colors=["k"]
    )

    # Plot dataset
    ax.scatter(X_gamma[:, 0], X_gamma[:, 1], c=y_gamma, cmap=plt.cm.coolwarm, s=50)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Highlight support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=2,
    )

plt.tight_layout()
plt.savefig("res/svm/svm_gamma_parameter_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# Create a plot showing the effect of regularization (C parameter) and gamma together on model complexity
fig, sub = plt.subplots(2, 3, figsize=(15, 10))
C_values = [0.1, 1, 100]
gamma_values = [0.01, 10]
titles_complexity = [
    "C=0.1, Gamma=0.01 (Underfitting)",
    "C=1, Gamma=0.01 (Balanced)",
    "C=100, Gamma=0.01 (Overfitting)",
    "C=0.1, Gamma=10 (Underfitting)",
    "C=1, Gamma=10 (Balanced but Complex)",
    "C=100, Gamma=10 (Overfitting)",
]

for i, (C, gamma) in enumerate(
    [(0.1, 0.01), (1, 0.01), (100, 0.01), (0.1, 10), (1, 10), (100, 10)]
):
    # Create non-linear data
    X_complex, y_complex = make_classification(
        n_samples=150,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
        n_classes=2,
    )

    # Add non-linear pattern
    X_complex = np.c_[X_complex[:, 0], X_complex[:, 1] + (0.5 * X_complex[:, 0] ** 2)]

    ax = sub[i // 3, i % 3]

    # Train SVM with specified C and gamma values
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    clf.fit(X_complex, y_complex)

    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_complex,
        plot_method="contour",
        levels=[0],
        alpha=0.4,
        ax=ax,
        colors=["k"],
    )

    # Plot dataset
    ax.scatter(
        X_complex[:, 0], X_complex[:, 1], c=y_complex, cmap=plt.cm.coolwarm, s=50
    )

    # Add title and labels
    ax.set_title(titles_complexity[i])
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Highlight support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=2,
    )

plt.tight_layout()
plt.savefig(
    "res/svm/svm_c_and_gamma_parameter_effect.png", dpi=150, bbox_inches="tight"
)
plt.show()

# Create a plot showing the effect of polynomial degree parameter
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
degree_values = [2, 3, 4, 5]
titles_degree = [
    "Polynomial Kernel (degree=2)",
    "Polynomial Kernel (degree=3)",
    "Polynomial Kernel (degree=4)",
    "Polynomial Kernel (degree=5)",
]

for i, (degree, title) in enumerate(zip(degree_values, titles_degree)):
    # Create non-linear data
    X_poly, y_poly = make_classification(
        n_samples=150,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
        n_classes=2,
    )

    # Add some non-linearity
    X_poly = np.c_[X_poly[:, 0], X_poly[:, 1] + (0.2 * X_poly[:, 0] ** 2)]

    ax = sub[i // 2, i % 2]

    # Train SVM with polynomial kernel of specified degree
    clf = svm.SVC(kernel="poly", degree=degree, gamma="scale", C=1.0)
    clf.fit(X_poly, y_poly)

    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_poly, plot_method="contour", levels=[0], alpha=0.4, ax=ax, colors=["k"]
    )

    # Plot dataset
    ax.scatter(X_poly[:, 0], X_poly[:, 1], c=y_poly, cmap=plt.cm.coolwarm, s=50)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Highlight support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=2,
    )

plt.tight_layout()
plt.savefig("res/svm/svm_polynomial_degree_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# Create a plot comparing different SVM variants (classification and regression)

# Generate sample classification data
X_var, y_var = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42,
)

# Add some overlap to make the comparison more interesting
X_var[y_var == 1] += 0.5

# Create sample regression data
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

# Define all SVM variants to compare - 2 classification and 2 regression
# Using different parameters to make C-SVM and Nu-SVM more visually distinct
variants = [
    (
        "C-SVM (Classification)",
        SVC(kernel="rbf", C=0.1, gamma="scale"),
        X_var,
        y_var,
        "classification",
    ),  # Soft margin C-SVM
    (
        "Nu-SVM (Classification)",
        NuSVC(kernel="rbf", nu=0.9, gamma="scale", random_state=42),
        X_var,
        y_var,
        "classification",
    ),  # High nu Nu-SVM
    (
        "Epsilon-SVR (Regression)",
        SVR(kernel="rbf", C=1, gamma=0.1, epsilon=0.1),
        X_reg,
        y_reg,
        "regression",
    ),  # Regression
    (
        "Nu-SVR (Regression)",
        NuSVR(kernel="rbf", C=1, gamma=0.1, nu=0.1),
        X_reg,
        y_reg,
        "regression",
    ),  # Regression
]

fig, sub = plt.subplots(2, 2, figsize=(12, 10))

for i, (title, model, X_data, y_data, variant_type) in enumerate(variants):
    ax = sub[i // 2, i % 2]

    # Train the model
    model.fit(X_data, y_data)

    if variant_type == "classification":
        # For classification - plot decision boundary
        disp = DecisionBoundaryDisplay.from_estimator(
            model,
            X_data,
            plot_method="contour",
            levels=[0],
            alpha=0.4,
            ax=ax,
            colors=["k"],
        )

        # Plot dataset
        ax.scatter(
            X_data[:, 0], X_data[:, 1], c=y_data, cmap=plt.cm.coolwarm, s=50, alpha=0.7
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Highlight support vectors for classification models
        if hasattr(model, "support_vectors_"):
            sv_plot = ax.scatter(
                model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=100,
                facecolors="none",
                edgecolors="k",
                linewidths=2,
                label="Support Vectors",
            )

            # Add text annotations to differentiate C-SVM and Nu-SVM
            if "C-SVM" in title:
                # For C-SVM, add annotation about the C parameter
                ax.text(
                    0.02,
                    0.98,
                    f"C={model.C:.1f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            elif "Nu-SVM" in title:
                # For Nu-SVM, add annotation about the nu parameter and estimated fraction of SVs
                n_support_ratio = (
                    (model.n_support_ / len(y_data)).mean()
                    if hasattr(model, "n_support_")
                    else "N/A"
                )
                ax.text(
                    0.02,
                    0.98,
                    f"Nu={model.nu:.1f}\nSVs: ~{n_support_ratio:.1%}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )
    else:  # regression
        # For regression - plot the regression line
        X_plot = np.linspace(X_data.min(), X_data.max(), 300).reshape(-1, 1)
        y_plot = model.predict(X_plot)

        ax.scatter(X_data, y_data, color="black", label="Data", alpha=0.6)
        ax.plot(X_plot, y_plot, color="red", label="SVR Prediction")

        # Show epsilon tube for Epsilon-SVR
        if "Epsilon" in title:
            ax.fill_between(
                X_plot.ravel(),
                model.predict(X_plot) - model.epsilon,
                model.predict(X_plot) + model.epsilon,
                color="red",
                alpha=0.2,
                label=f"Epsilon tube (ε={model.epsilon})",
            )

        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")

        # Highlight support vectors for regression models
        if hasattr(model, "support_"):
            sv_indices = model.support_
            ax.scatter(
                X_data[sv_indices],
                y_data[sv_indices],
                s=100,
                facecolors="none",
                edgecolors="k",
                linewidths=2,
                label="Support Vectors",
            )

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("res/svm/svm_variants_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
