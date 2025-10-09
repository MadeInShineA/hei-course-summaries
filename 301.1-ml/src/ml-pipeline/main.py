import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Load breast cancer dataset for a real-life example
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM and KNN models
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

models = {'SVM': svm_model, 'K-NN': knn_model}
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# 1. Confidence Interval Plot
print("Generating Confidence Interval Plot...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx]
    means = []
    ci_lowers = []
    ci_uppers = []
    model_names = []

    for model_name, model in models.items():
        # Perform cross-validation to get scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring=metric)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        ci_lower = mean_score - z_score * (std_score / np.sqrt(len(cv_scores)))
        ci_upper = mean_score + z_score * (std_score / np.sqrt(len(cv_scores)))
        means.append(mean_score)
        ci_lowers.append(mean_score - ci_lower)
        ci_uppers.append(ci_upper - mean_score)
        model_names.append(model_name)

    bars = ax.bar(model_names, means, yerr=[ci_lowers, ci_uppers], capsize=5,
                  color=['#4e79a7', '#f28e2c'], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{name} with {confidence_level*100}% CI', fontsize=14, fontweight='bold')
    ax.set_ylabel(name, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add value labels
    for bar, mean, lower, upper in zip(bars, means, ci_lowers, ci_uppers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + upper + 0.02,
                f'{mean:.3f}\n[{mean-lower:.3f}, {mean+upper:.3f}]',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Performance Metrics Comparison on Breast Cancer Dataset', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("res/ml-pipeline/confidence_interval.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. ROC Curve
print("Generating ROC Curve...")
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
colors = ['#e74c3c', '#3498db']  # Red for SVM, Blue for K-NN
for i, (name, model) in enumerate(models.items()):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=3, alpha=0.8,
             label=f'{name} (AUC = {roc_auc:.3f})', marker='o', markersize=4, markevery=0.1)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('ROC Curves Comparison: SVM vs K-NN on Breast Cancer Dataset', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res/ml-pipeline/roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("All plots generated and saved!")