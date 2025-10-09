import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample classification data for SHAP explanations
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=1,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.Explainer(svm_model.predict_proba, X_train)

# Calculate SHAP values for test set (using a subset for efficiency)
X_test_sample = X_test[:100]  # Use first 100 samples
shap_values = explainer(X_test_sample)

# Convert to binary classification SHAP values (for positive class)
shap_values_binary = shap_values[..., 1]  # Take positive class

# 1. Partial Dependence Plot
print("Generating Partial Dependence Plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Create partial dependence for the first two features
features = [0, 1]
pd_results = partial_dependence(svm_model, X_train, features, kind="average")

# Plot partial dependence
PartialDependenceDisplay.from_estimator(
    svm_model,
    X_train,
    features,
    ax=ax,
    kind="average",
    subsample=50,
    random_state=42
)

ax.set_title("Partial Dependence Plot - Features 0 and 1")
plt.tight_layout()
plt.savefig("res/shap/partial_dependence_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. Waterfall SHAP Plot
print("Generating Waterfall SHAP Plot...")
# Select one instance for explanation
instance_idx = 0
instance = X_test_sample[instance_idx]

# Create waterfall plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(
    shap_values_binary[instance_idx],
    max_display=10,
    show=False
)
plt.title(f"Waterfall SHAP Plot - Instance {instance_idx}")
plt.tight_layout()
plt.savefig("res/shap/waterfall_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. Beeswarm SHAP Plot
print("Generating Beeswarm SHAP Plot...")
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(
    shap_values_binary,
    max_display=10,
    show=False
)
plt.title("Beeswarm SHAP Plot - Feature Importance Distribution")
plt.tight_layout()
plt.savefig("res/shap/beeswarm_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("All SHAP plots generated and saved!")