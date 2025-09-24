import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay

# Generate sample data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Train SVM model
clf = svm.SVC(kernel='linear', C=1.0)
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
    label="Support Vectors"
)

# Add labels and title
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("SVM Decision Boundary with Support Vectors")
ax.legend()

# Add margins
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res/svm/svm_decision_boundary.png", dpi=150, bbox_inches='tight')
plt.show()

# Now create a plot showing the effect of different kernels
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
titles = [
    "Linear Kernel",
    "Polynomial Kernel (degree=3)",
    "RBF Kernel",
    "Sigmoid Kernel"
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
        n_classes=2
    )
    
    # Add some non-linearity
    X2 = np.c_[X2[:, 0], X2[:, 1] + (0.2 * X2[:, 0]**2)]
    
    ax = sub[i // 2, i % 2]
    
    # Train SVM with specified kernel
    clf = svm.SVC(kernel=kernel, gamma=1, C=1.0)
    clf.fit(X2, y2)
    
    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X2,
        plot_method="contour",
        levels=[0],
        alpha=0.4,
        ax=ax,
        colors=["k"]
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
        linewidths=2
    )

plt.tight_layout()
plt.savefig("res/svm/svm_kernels_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# Create a plot showing the effect of the C parameter
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
C_values = [0.01, 0.1, 1, 100]
titles_C = [
    "C = 0.01 (Soft Margin)",
    "C = 0.1",
    "C = 1 (Balanced)",
    "C = 100 (Hard Margin)"
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
        random_state=42
    )
    
    # Add some noise to create overlapping classes
    X3[-5:] += np.random.uniform(-1, 1, (5, 2))
    
    # Train SVM with specified C value
    clf = svm.SVC(kernel="linear", C=C)
    clf.fit(X3, y3)
    
    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X3,
        plot_method="contour",
        levels=[0],
        alpha=0.4,
        ax=ax,
        colors=["k"]
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
        linewidths=2
    )

plt.tight_layout()
plt.savefig("res/svm/svm_c_parameter_effect.png", dpi=150, bbox_inches='tight')
plt.show()

# Create a plot showing SVM regression
from sklearn.svm import SVR
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Fit regression model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_reg, y_reg)

# Plot the regression
fig, ax = plt.subplots(figsize=(10, 6))
X_plot = np.linspace(X_reg.min(), X_reg.max(), 300).reshape(-1, 1)
y_plot = svr.predict(X_plot)

ax.scatter(X_reg, y_reg, color='black', label='Data')
ax.plot(X_plot, y_plot, color='red', label='SVR')
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.set_title('SVM Regression Example')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res/svm/svm_regression.png", dpi=150, bbox_inches='tight')
plt.show()

# Create a plot showing the effect of gamma parameter for RBF kernel
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
gamma_values = [0.01, 0.1, 1, 10]
titles_gamma = [
    "Gamma = 0.01 (Low)",
    "Gamma = 0.1",
    "Gamma = 1 (Medium)",
    "Gamma = 10 (High)"
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
        n_classes=2
    )
    
    ax = sub[i // 2, i % 2]
    
    # Train SVM with specified gamma value
    clf = svm.SVC(kernel="rbf", gamma=gamma, C=1.0)
    clf.fit(X_gamma, y_gamma)
    
    # Plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_gamma,
        plot_method="contour",
        levels=[0],
        alpha=0.4,
        ax=ax,
        colors=["k"]
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
        linewidths=2
    )

plt.tight_layout()
plt.savefig("res/svm/svm_gamma_parameter_effect.png", dpi=150, bbox_inches='tight')
plt.show()