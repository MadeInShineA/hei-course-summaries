# Machine Learning Course Summary - Decision Trees and Random Forests

## Table of Contents

1. [Decision Trees](#decision-trees)
2. [Random Forests](#random-forests)
3. [Summary](#summary)

---

## <a name="decision-trees"></a>Decision Trees

### What are Decision Trees?

Decision Trees are supervised learning models that represent decisions and their possible consequences as a tree-like structure. They are used for both classification (predicting categories) and regression (predicting continuous values) tasks.

- **Tree structure**: Root node (full dataset) branches into decision nodes based on features, leading to leaf nodes (predictions)
- **Non-parametric**: No assumptions about data distribution
- **Interpretable**: Easy to visualize and understand decision paths

### Tree Structure

Decision Trees consist of hierarchical nodes that progressively refine the dataset:

- **Root Node**: The top node containing the entire training dataset; first split is chosen here.
- **Internal Nodes**: Non-leaf nodes representing decisions based on a feature threshold (e.g., "Age > 30?"). Each leads to two child nodes.
- **Leaf Nodes**: Terminal nodes where splitting stops; store the prediction (majority class for classification, mean/median for regression).

The tree's depth and branching reflect the complexity of decision boundaries.

#### Components Visualization

```mermaid
graph LR
    ROOT["Root Node<br/>Full Dataset<br/>Best Split: Feature X > Threshold"] -->|"Yes"| INTERNAL1["Internal Node<br/>Subset Data<br/>Next Split: Feature Y <= Value"]
    ROOT -->|"No"| INTERNAL1
    INTERNAL1 -->|"Yes"| LEAF1["Leaf Node<br/>Class A (Majority Vote)<br/>or Mean Value"]
    INTERNAL1 -->|"No"| LEAF2["Leaf Node<br/>Class B"]
    
    style ROOT fill:#2563eb40,stroke:#2563eb,stroke-width:3px
    style INTERNAL1 fill:#7c3aed20,stroke:#7c3aed
    style LEAF1 fill:#16a34a20,stroke:#16a34a
    style LEAF2 fill:#16a34a20,stroke:#16a34a
```

### How Decision Trees Work

Decision Trees build by recursively splitting the dataset:

1. **Root Selection**: Choose the best feature to split the data at the root
2. **Splitting Criteria**: For classification, use Gini impurity or entropy; for regression, mean squared error (MSE) or mean absolute error (MAE)
3. **Recursion**: Repeat splitting on subsets until stopping criteria (e.g., max depth)
4. **Prediction**: For classification, majority class in leaf; for regression, mean value in leaf
5. **Pruning**: Post-build trimming to reduce overfitting by removing unnecessary branches

#### Detailed Explanation of Splitting Criteria

Splitting criteria determine the best way to divide the dataset at each node, maximizing purity (for classification) or minimizing error (for regression). The choice affects tree structure and performance.

##### Classification Criteria

- **Gini Impurity**  
  Measures the probability of misclassifying a random sample based on class distribution.  
  Formula:  
  $Gini = 1 - \sum_{i=1}^{C} p_i^2$  
  where $C$ is the number of classes, and $p_i$ is the proportion of class $i$.  

  - Ranges from 0 (pure subset) to 0.5 (max impurity for binary classes).  
  - For a split: $Gini_{split} = \frac{n_{left}}{n} Gini_{left} + \frac{n_{right}}{n} Gini_{right}$.  
  - Goal: Minimize $Gini_{split}$.  
  - Pros: Computationally efficient, less sensitive to class imbalance.  

  **Example**: Balanced dataset (50% A, 50% B): $Gini = 1 - (0.5^2 + 0.5^2) = 0.5$. Perfect split: $Gini_{split} = 0$.

- **Entropy (Information Gain)**  
  Quantifies uncertainty in the data.  
  Formula:  
  $Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)$  
  Ranges from 0 (pure) to 1 (max for binary).  

  - Information Gain: $IG = Entropy_{parent} - \left( \frac{n_{left}}{n} Entropy_{left} + \frac{n_{right}}{n} Entropy_{right} \right)$.  
  - Goal: Maximize $IG$.  
  - Pros: Emphasizes significant uncertainty reduction.  
  - Cons: More computationally expensive due to logs.  

  **Example**: Balanced dataset: $Entropy = - (0.5 \log_2(0.5) + 0.5 \log_2(0.5)) = 1$. Perfect split: $IG = 1$.

Gini and Entropy often produce similar trees; Gini is faster for large datasets.

##### Regression Criteria

- **Mean Squared Error (MSE)**  
  Measures variance around the mean prediction.  
  Formula:  
  $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$  

  - For a split: Weighted average of children's MSEs.  
  - Goal: Minimize to reduce variance.  
  - Cons: Sensitive to outliers (penalizes large errors heavily).  

  **Example**: Prices {100k, 150k, 200k}, mean=150k: $MSE = \frac{(100-150)^2 + (150-150)^2 + (200-150)^2}{3} \approx 1000k^2$. Split reduces overall MSE.

- **Mean Absolute Error (MAE)**  
  Measures average absolute deviation.  
  Formula:  
  $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \bar{y}|$  

  - For a split: Weighted sum of children's MAEs.  
  - Goal: Minimize absolute errors.  
  - Pros: Robust to outliers.  

In scikit-learn, defaults are Gini (classification) and MSE (regression). Choose based on data (e.g., MAE for outliers) and computational needs.

#### Example (Classification)

For iris dataset classifying flower species:

- Root split: Petal length > 2.5 cm?
  - Yes → Split on petal width → Leaf: Versicolor
  - No → Leaf: Setosa

For regression (house prices):

- Root split: Size > 1000 sq ft?
  - Yes → Mean price: $300k
  - No → Mean price: $150k

#### Detailed Example: Building a Decision Tree with Gini Impurity

This comprehensive example demonstrates the step-by-step construction of a decision tree using the Gini impurity criterion for a binary classification task. We'll use a toy dataset of 8 samples with two numerical features (Age and Income) and a binary target (Buy: Yes or No). The goal is to predict whether a person will buy based on age and income. We'll detail every calculation using LaTeX math, explain the rationale behind each split, and visualize the tree's growth. This mirrors the mathematical rigor found in notebooks, ensuring transparency in how impurity is minimized at each node.

**Dataset Overview:**

The dataset is small for clarity but representative of real-world tabular data. Samples are sorted by feature for threshold selection.

| Sample | Age | Income (k) | Buy  |
|--------|-----|------------|------|
| 1      | 25  | 40         | No   |
| 2      | 30  | 50         | No   |
| 3      | 35  | 60         | Yes  |
| 4      | 40  | 70         | Yes  |
| 5      | 45  | 80         | Yes  |
| 6      | 50  | 90         | No   |
| 7      | 55  | 100        | Yes  |
| 8      | 60  | 110        | Yes  |

Class distribution: 5 Yes, 3 No (62.5% Yes, 37.5% No). Features are continuous, so we test binary splits at midpoints between sorted values.

**Step 1: Compute Gini Impurity for the Root Node**

The root node contains all 8 samples. Gini impurity quantifies class mixing:  
$Gini = 1 - \sum_{i=1}^{C} p_i^2$  
where $C = 2$ (classes: Yes, No), $p_{\text{Yes}} = 5/8 = 0.625$, $p_{\text{No}} = 3/8 = 0.375$.  

$Gini = 1 - (0.625^2 + 0.375^2) = 1 - (0.390625 + 0.140625) = 1 - 0.53125 = 0.46875$  

This value (0.46875) indicates moderate impurity. A pure node has Gini = 0; maximum for binary is 0.5. Lower Gini means better separation.

**Step 2: Evaluate Potential Splits to Minimize Weighted Gini**

We test splits on each feature by choosing thresholds between consecutive sorted values (e.g., for Age: midpoints like 27.5, 32.5, etc.). For each threshold, compute:  

- Left and right child subsets.  
- Gini for each child: $Gini_{\text{left}} = 1 - \sum p_{i,\text{left}}^2$, similarly for right.  
- Weighted Gini for the split: $Gini_{\text{split}} = \frac{n_{\text{left}}}{n} Gini_{\text{left}} + \frac{n_{\text{right}}}{n} Gini_{\text{right}}$.  
Select the split with the lowest $Gini_{\text{split}}$ (maximizing purity gain).  

**Substep 2.1: Splits on Age**  
Sorted Ages: 25, 30, 35, 40, 45, 50, 55, 60. Possible thresholds: 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5.  

- **Threshold: Age ≤ 32.5** (Left: Samples 1-2; Right: 3-8)  
  Left (n=2): 2 No → $p_{\text{No}} = 1$  
  $Gini_{\text{left}} = 1 - 1^2 = 0$  
  Right (n=6): 5 Yes, 1 No → $p_{\text{Yes}} = 5/6 \approx 0.833$, $p_{\text{No}} = 1/6 \approx 0.167$.  
  $Gini_{\text{right}} = 1 - (0.833^2 + 0.167^2) = 1 - (0.6944 + 0.0278) = 0.2778$ (some impurity due to minority class; calculated as 1 - (0.6944 + 0.0278) = 0.2778).  
  $Gini_{\text{split}} = (2/8) \times 0 + (6/8) \times 0.2778 = 0 + 0.20835 = 0.20835$ (weighted by subset sizes).  
  Gain = $0.46875 - 0.20835 = 0.2604$ (significant reduction, good split).

  **Hypothetical Tree for Age ≤32.5:**  

  ```mermaid
  graph TD
      ROOT["Root<br/>Age <= 32.5?<br/>Gain=0.2604"] -->|"Yes"| L["Leaf: No<br/>(Pure)"]
      ROOT -->|"No"| R["Node: 5 Yes, 1 No<br/>Gini=0.2778"]
      style ROOT fill:#2563eb40,stroke:#2563eb,stroke-width:3px
      style L fill:#16a34a20,stroke:#16a34a
      style R fill:#7c3aed20,stroke:#7c3aed
  ```

**Substep 2.2: Splits on Income**  
Sorted Incomes: 40, 50, 60, 70, 80, 90, 100, 110. Thresholds: 45, 55, 65, 75, 85, 95, 105.  

- **Threshold: Income ≤ 55k** (Left: 1-2; Right: 3-8)  
  Left (n=2): 2 No → $p_{\text{No}} = 1$  
  $Gini_{\text{left}} = 1 - 1 ^ 2 = 0$  
  Right (n=6): 5 Yes, 1 No → $p_{\text{Yes}} = 5/6 \approx 0.833$, $p_{\text{No}} = 1/6 \approx 0.167$.  
  $Gini_{\text{right}} = 1 - (0.833^2 + 0.167^2) = 1 - (0.6944 + 0.0278) = 0.2778$  
  $Gini_{\text{split}} = (2/8) \times 0 + (6/8) \times 0.2778 = 0 + 0.20835 = 0.20835$  
  Gain = $0.46875 - 0.20835 = 0.2604$  

  **Hypothetical Tree for Income ≤55k:**  

  ```mermaid
  graph TD
      ROOT["Root<br/>Income <= 55k?<br/>Gain=0.2604"] -->|"Yes"| L["Leaf: No<br/>(Pure)"]
      ROOT -->|"No"| R["Node: 5 Yes, 1 No<br/>Gini=0.2778"]
      style ROOT fill:#2563eb40,stroke:#2563eb,stroke-width:3px
      style L fill:#16a34a20,stroke:#16a34a
      style R fill:#7c3aed20,stroke:#7c3aed
  ```

Overall best split: Tie between Age ≤32.5 and Income ≤55k. We choose Age ≤32.5 (arbitrary but consistent). This reduces root Gini from 0.46875 to 0.20835, a gain of 0.2604.

**Tree After Root Split:**

The tree now branches based on Age ≤32.5. Left is pure (No), right needs further splitting.

```mermaid
graph TD
    ROOT["Root<br/>Age <= 32.5?<br/>Gain=0.2604"] -->|"Yes"| LEFT["Leaf: No<br/>(2 No, Pure<br/>Gini=0)"]
    ROOT -->|"No"| RIGHT["Internal Node<br/>Age > 32.5<br/>(5 Yes, 1 No)<br/>Gini=0.2778"]
    
    style ROOT fill:#2563eb40,stroke:#2563eb,stroke-width:3px
    style LEFT fill:#16a34a20,stroke:#16a34a
    style RIGHT fill:#7c3aed20,stroke:#7c3aed
```

**Step 3: Evaluate Child Nodes for Further Splitting**

- **Left Child**: Pure (Gini=0) → Becomes a leaf: Predict No for new samples with Age ≤32.5.  
- **Right Child**: Gini=0.2778 >0 → Not pure; recurse to split further. Subset: Samples 3-8 (Ages 35-60, Incomes 60k-110k, 5 Yes, 1 No).

**Step 4: Split the Right Child Node**

Repeat the process on the right subset (n=6). Compute Gini for this node:  
$Gini_{\text{right}} = 1 - \left( (5/6)^2 + (1/6)^2 \right) = 1 - (0.6944 + 0.0278) = 0.2778$ (same as before).  

**Substep 4.1: Splits on Age in Right Subset**  
Sorted Ages in subset: 35,40,45,50,55,60. Thresholds: 37.5,42.5,47.5,52.5,57.5.  

- **Threshold: Age ≤37.5** (Left: 3; Right: 4-8)  
  Left (n=1): 1 Yes → $p_{\text{Yes}} = 1$
  $Gini_{\text{left}} = 1 - 1 ^ 2 = 0$  
  Right (n=5): 4 Yes, 1 No → $p_{\text{Yes}} = 4/5 = 0.8$, $p_{\text{No}} = 1/5 = 0.2$.  
  $Gini_{\text{right}} = 1 - (0.8^2 + 0.2^2) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32$  
  $Gini_{\text{split}} = (1/6) \times 0 + (5/6) \times 0.32 \approx 0.2667$  
  Gain = $0.2778 - 0.2667 = 0.0111$  

  **Hypothetical Tree for Age ≤37.5 (Right Node):**  

  ```mermaid
  graph TD
      ROOT["Right Node<br/>Age <= 37.5?<br/>Gain=0.0111"] -->|"Yes"| L["Leaf: Yes<br/>(Pure)"]
      ROOT -->|"No"| R["Node: 4 Yes, 1 No<br/>Gini=0.32"]
      style ROOT fill:#7c3aed40,stroke:#7c3aed,stroke-width:3px
      style L fill:#16a34a20,stroke:#16a34a
      style R fill:#7c3aed20,stroke:#7c3aed
  ```

**Substep 4.2: Splits on Income in Right Subset**  
Sorted Incomes: 60,70,80,90,100,110. Thresholds: 65,75,85,95,105.  

- **Threshold: Income ≤65k** (Left: 3; Right: 4-8)  
  Left (n=1): 1 Yes → $p_{\text{Yes}} = 1$  
  $Gini_{\text{left}} = 1 - 1 ^ 2 = 0$.  
  Right (n=5): 4 Yes, 1 No → $p_{\text{Yes}} = 4/5 = 0.8$, $p_{\text{No}} = 1/5 = 0.2$.  
  $Gini_{\text{right}} = 1 - (0.8^2 + 0.2^2) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32$.  
  $Gini_{\text{split}} = (1/6) \times 0 + (5/6) \times 0.32 \approx 0.2667$ (same as Age ≤37.5).  
  Gain = $0.2778 - 0.2667 = 0.0111$.

  **Hypothetical Tree for Income ≤65k (Right Node):**  

  ```mermaid
  graph TD
      ROOT["Right Node<br/>Income <= 65k?"] -->|"Yes"| L["Leaf: Yes<br/>(Pure)"]
      ROOT -->|"No"| R["Node: 4 Yes, 1 No<br/>Gini=0.32"]
      style ROOT fill:#7c3aed40,stroke:#7c3aed,stroke-width:3px
      style L fill:#16a34a20,stroke:#16a34a
      style R fill:#7c3aed20,stroke:#7c3aed
  ```

Choose Age ≤37.5 (tie-breaker). This reduces right node's Gini from 0.2778 to 0.2667.

**Step 5: Finalize the Tree Structure**

The right node splits into:  

- Left (Age ≤37.5): Sample 3 (Yes) → Pure leaf: Predict Yes.  
- Right (Age >37.5): Samples 4-8 (4 Yes, 1 No) → Not pure, but for this example, we stop (as if min_samples_leaf=1). Predict Yes (majority).  

The tree is now complete (max depth=2). Overall impurity reduction: From 0.46875 to weighted average of leaves (0 for most, small for right leaf).

**Final Tree Visualization:**

```mermaid
graph TD
    ROOT["Root<br/>Age <= 32.5?"] -->|"Yes"| LEAF1["Predict: No<br/>(Pure)"]
    ROOT -->|"No"| INTERNAL["Age <= 37.5?"]
    INTERNAL -->|"Yes"| LEAF2["Predict: Yes<br/>(Pure)"]
    INTERNAL -->|"No"| LEAF3["Predict: Yes<br/>(4 Yes, 1 No)"]
    
    style ROOT fill:#2563eb40,stroke:#2563eb,stroke-width:3px
    style INTERNAL fill:#7c3aed20,stroke:#7c3aed
    style LEAF1 fill:#16a34a20,stroke:#16a34a
    style LEAF2 fill:#16a34a20,stroke:#16a34a
    style LEAF3 fill:#16a34a20,stroke:#16a34a
```

**Step 6: Prediction Example and Insights**

For a new sample (Age=28, Income=45k):  

- Age=28 ≤32.5 → Follow "Yes" branch → Predict No.  

For (Age=38, Income=75k):  

- Age=38 >32.5 → "No" branch → Age=38 >37.5 → "No" subbranch → Predict Yes.  

This tree is interpretable: Young low-income → No; Older → Yes. Gini ensured balanced, purity-focused splits. In practice, use cross-validation to tune depth and avoid overfitting. This example highlights how recursive partitioning with Gini builds accurate, explainable models from data.

#### Decision Tree Prediction Flow

```mermaid
flowchart TD
    INPUT["New Sample<br/>Features"] --> ROOT["Root Node<br/>First Split"]
    ROOT --> DEC1{"Condition Met?<br/>e.g., Feature X > Threshold"}
    DEC1 -->|"Yes"| BRANCH1["Next Node/Split"]
    DEC1 -->|"No"| BRANCH2["Next Node/Split"]
    BRANCH1 --> LEAF["Leaf Node<br/>Prediction: Class/Value"]
    BRANCH2 --> LEAF
    LEAF --> OUTPUT["Final Prediction"]
    
    style INPUT fill:#f3f4f620,stroke:#6b7280
    style ROOT fill:#2563eb20,stroke:#2563eb
    style DEC1 fill:#7c3aed20,stroke:#7c3aed
    style BRANCH1 fill:#10b98120,stroke:#10b981
    style BRANCH2 fill:#10b98120,stroke:#10b981
    style LEAF fill:#16a34a20,stroke:#16a34a
    style OUTPUT fill:#16a34a40,stroke:#16a34a
```

### Pruning and Regularization

Pruning prevents overfitting by simplifying the tree:

- **Pre-pruning** (Early Stopping): Halt growth if max depth reached, minimum samples per split/leaf not met, or no significant impurity reduction.
- **Post-pruning** (Cost-Complexity Pruning): Grow full tree, then remove subtrees that increase validation error minimally. Uses a complexity parameter α to balance fit and simplicity.

| Pruning Type | Pros | Cons |
|--------------|------|------|
| **Pre-pruning** | Faster training; avoids deep trees | May underfit if stopped too early |
| **Post-pruning** | Better accuracy; explores full structure | More computationally intensive |

Regularization parameters like min_samples_leaf smooth leaves and reduce overfitting.

### Algorithm Pipeline (Decision Tree)

```mermaid
flowchart TD
    A["Load Dataset<br/>Features + Targets"] --> B["Select Best Split<br/>Gini/Entropy/MSE"]
    B --> C{"Create Node?<br/>Stopping Criteria?"}
    C -->|"Yes"| D["Split Data<br/>Left/Right Subsets"]
    C -->|"No"| E["Leaf Node<br/>Make Prediction"]
    D --> F["Recurse on Subsets"]
    F --> B
    E --> G["Full Tree Built"]
    
    style A fill:#2563eb20,stroke:#2563eb,stroke-width:2px
    style B fill:#7c3aed20,stroke:#7c3aed,stroke-width:2px
    style C fill:#d9770620,stroke:#d97706,stroke-width:2px
    style D fill:#10b98120,stroke:#10b981,stroke-width:2px
    style F fill:#d9770620,stroke:#d97706,stroke-width:2px
    style E fill:#16a34a20,stroke:#16a34a,stroke-width:2px
    style G fill:#16a34a40,stroke:#16a34a,stroke-width:2px
```

### Parameters and Tuning (Decision Tree)

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **Max Depth** | Maximum tree levels | Deeper trees fit more but overfit |
| **Min Samples Split** | Min data for internal node | Higher values prevent overfitting |
| **Min Samples Leaf** | Min data for leaf | Smooths predictions, reduces overfitting |

Tuning: Use grid search and cross-validation to find optimal values.

---

## <a name="random-forests"></a>Random Forests

### What are Random Forests?

Random Forests are an ensemble learning method that combines multiple decision trees to improve predictive performance and control overfitting. Each tree is trained on a random subset of the data and features, and predictions are aggregated for robustness. They excel in classification and regression, often outperforming single trees on noisy or high-dimensional data.

- **Ensemble Structure**: Collection of diverse decision trees voting or averaging their outputs
- **Bagging + Randomness**: Reduces variance through bootstrap sampling and feature randomization
- **Robust and Accurate**: Handles overfitting better than single trees while maintaining interpretability via feature importances

### How Random Forests Work

Random Forests extend decision trees by introducing randomness and aggregation:

1. **Bootstrap Sampling (Bagging)**: Create multiple subsets of the training data by sampling with replacement (each ~63% unique samples). This introduces diversity among trees.
2. **Random Feature Selection**: At each split in a tree, consider only a random subset of features (e.g., √n for classification, n/3 for regression), preventing dominance by strong predictors and decorrelating trees.
3. **Tree Construction**: Build each decision tree independently using the same splitting criteria as single trees (e.g., Gini, MSE), but limited to random features.
4. **Aggregation**: For classification, use majority voting across trees; for regression, average predictions. Out-of-Bag (OOB) samples (not used in a tree's training) enable internal validation.
5. **Feature Importance**: Computed as the average decrease in impurity across trees when a feature is used for splitting, providing insights into relevance.

This process reduces the high variance of single trees, leading to more stable and accurate models.

#### Detailed Explanation of Splitting in Random Forests

Splitting mirrors decision trees but uses random feature subsets at each node for diversity. Core criteria (Gini, Entropy, MSE, MAE) are the same.

- **Feature Subset Size**: Typically $m = \sqrt{n}$ (classification) or $m = n/3$ (regression), where $n$ is total features. Sample $m$ features randomly per split and select the best.

  This decorrelates trees by limiting options, even for dominant features.

- **Impurity Calculation**: Compute criterion (e.g., Gini) only over the $m$ features; choose split minimizing weighted child impurity.

  **Example**: 10 features, $m \approx 3$. Sample {Age, Income, Education}. Test splits (e.g., Age > 30?), pick lowest $Gini_{split}$. Repeat per node.

- **Impact**: Less greedy splits reduce overfitting. For regression, MSE over fewer features adds robustness.

- **OOB Error**: ~37% data OOB per tree; aggregate for generalization estimate without validation set.

Individual trees may overfit, but ensemble averages errors for better performance.

#### Example (Classification with Random Forest)

For the Iris dataset, build 100 trees:

- Each tree on a bootstrap sample, using random 2 features per split (√4=2).
- Tree 1 might split on Petal Length first (Gini reduction high).
- Tree 2 on Sepal Width due to random selection.
- Prediction for a new sample: 95 trees vote Versicolor → Final: Versicolor.

For regression (house prices):

- 50 trees, each averaging prices in leaves based on random feature subsets (e.g., Size, Location).
- Final prediction: Average of 50 tree outputs, e.g., $280k.

#### Random Forest Structure Graph (Simplified Ensemble)

```mermaid
graph TD
    A["Dataset"] --> B["Bootstrap Sample 1"]
    A --> C["Bootstrap Sample 2"]
    A --> D["Bootstrap Sample 3"]
    B --> E["Tree 1<br/>Random Features at Splits"]
    C --> F["Tree 2<br/>Different Random Features"]
    D --> G["Tree 3"]
    E --> H["Predictions from Tree 1"]
    F --> I["Predictions from Tree 2"]
    G --> J["Predictions from Tree 3"]
    H --> K["Aggregate:<br/>Majority Vote or Average"]
    I --> K
    J --> K
    K --> L["Final Prediction"]

    style A fill:#2563eb20,stroke:#2563eb,stroke-width:2px
    style B fill:#7c3aed20,stroke:#7c3aed
    style C fill:#7c3aed20,stroke:#7c3aed
    style D fill:#7c3aed20,stroke:#7c3aed
    style E fill:#d9770620,stroke:#d97706
    style F fill:#d9770620,stroke:#d97706
    style G fill:#d9770620,stroke:#d97706
    style K fill:#10b98120,stroke:#10b981
    style L fill:#16a34a40,stroke:#16a34a
```

This illustrates how multiple trees, trained on varied data and features, contribute to a consensus prediction.

#### Random Forest Prediction Flow

```mermaid
flowchart TD
    INPUT["New Sample"] --> TREE1["Tree 1 Predict"]
    INPUT --> TREE2["Tree 2 Predict"]
    INPUT --> TREES["... Up to N Trees"]
    TREE1 --> AGG["Aggregate Outputs<br/>Vote (Classif) / Avg (Reg)"]
    TREE2 --> AGG
    TREES --> AGG
    AGG --> OUTPUT["Final Prediction"]
    
    style INPUT fill:#f3f4f620,stroke:#6b7280
    style TREE1 fill:#d9770620,stroke:#d97706
    style TREE2 fill:#d9770620,stroke:#d97706
    style TREES fill:#d9770620,stroke:#d97706
    style AGG fill:#7c3aed20,stroke:#7c3aed
    style OUTPUT fill:#16a34a40,stroke:#16a34a
```

### Key Components of Random Forests

- **Bootstrap Aggregating (Bagging)**: Trains trees on random samples to reduce variance.
- **Random Feature Selection**: Ensures tree diversity by limiting split options.
- **Voting/Averaging**: Combines predictions for final output.
- **Out-of-Bag (OOB) Error**: Internal estimate of model performance.
- **Feature Importance**: Mean impurity decrease per feature across trees.

### Pruning and Regularization in Random Forests

Individual trees can use the same pruning as decision trees, but the ensemble inherently regularizes via averaging. Parameters like max_depth apply per tree, while n_estimators controls ensemble size.

### Algorithm Pipeline (Random Forest)

```mermaid
flowchart TD
    A["Load Dataset"] --> B["Generate B Bootstrap Samples"]
    B --> C["For Each Sample:<br/>Build Tree with Random Features"]
    C --> D["Collect Tree Predictions"]
    D --> E{"Aggregate:<br/>Vote (Classif)/Avg (Reg)"}
    E --> F["Final Prediction"]
    
    style A fill:#2563eb20,stroke:#2563eb,stroke-width:2px
    style B fill:#7c3aed20,stroke:#7c3aed,stroke-width:2px
    style C fill:#d9770620,stroke:#d97706,stroke-width:2px
    style D fill:#10b98120,stroke:#10b981,stroke-width:2px
    style E fill:#7c3aed20,stroke:#7c3aed,stroke-width:2px
    style F fill:#16a34a40,stroke:#16a34a,stroke-width:2px
```

### Parameters and Tuning (Random Forest)

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **N Estimators** | Number of trees | More trees improve stability but increase time |
| **Max Features** | Features per split | 'sqrt' for classification, 'n_features/3' for regression |
| **Max Depth** | Depth per tree | Controls individual tree complexity |
| **Bootstrap** | Use sampling with replacement | True for bagging benefits |
| **Min Samples Split/Leaf** | As in decision trees, per tree | Prevents overfitting in individual trees |

Tuning Strategies:

1. **Grid Search**: Test combinations of n_estimators, max_features, max_depth.
2. **N Estimators**: Start with 100, increase until OOB error stabilizes.
3. **Max Features**: Tune based on problem dimensionality.
4. **Cross-Validation**: Use with OOB for efficient evaluation.

### Advantages Over Single Decision Trees

- **Reduced Overfitting**: Averaging smooths individual tree errors.
- **Improved Accuracy**: Often state-of-the-art on tabular data.
- **Feature Importance**: Global ranking across the ensemble.
- **Robustness**: Handles noisy data and outliers better.
- **Parallelizable**: Trees trained independently.

---

## <a name="summary"></a>Summary

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Recursive Partitioning** | Split data hierarchically for pure subsets in trees; extended to ensembles in RF |
| **Impurity Measures** | Guide splits (Gini, Entropy, MSE) to maximize gain; randomized in RF for diversity |
| **Ensemble Averaging** | RF reduces variance by combining multiple trees via bagging and feature randomness |
| **Interpretability** | Trees show explicit decisions; RF provides feature importances for insights |

### Algorithm Parameters

| Parameter | Tuning Guideline |
|-----------|------------------|
| **Max Depth** | Limit to 5-20; use CV to find optimal for both trees and per-tree in RF |
| **N Estimators (RF)** | 100-500; more for better stability, monitor OOB error |
| **Min Samples Leaf** | 1-10 to control overfitting in leaves |
| **Max Features** | sqrt(total) for classification; adjust via grid search |

### Best Practices

- **Preprocess Data**: Encode categoricals; no scaling needed, but handle missing values (impute or use built-in handling).
- **Cross-Validate**: Use k-fold CV for hyperparameter tuning; leverage OOB in RF for quick estimates.
- **Prune Trees**: Apply cost-complexity pruning; in RF, rely on ensemble regularization.
- **Feature Engineering**: Select relevant features; use RF importances for selection.
- **Evaluate Models**: For trees, check depth and purity; for RF, monitor OOB vs. test error to detect overfitting.
- **Handle Imbalance**: Use class weights in both; RF benefits from balanced bootstraps.
- **Scalability Tips**: Limit n_estimators for large data; parallelize tree building.

### When to Use

- **Decision Trees**: For interpretable models on small/medium datasets; quick prototyping; when explainability is key (e.g., medical decisions).
- **Random Forests**: High-dimensional or noisy data; when accuracy > interpretability; baseline for tabular ML (e.g., finance, customer analytics).
- **Both**: Non-linear problems; mixed feature types; avoid when data is very large (consider sampling) or sequential (use RNNs/LSTMs).
- **Imbalanced Classes**: RF with undersampling or weights.
- **Quick Insights**: Trees for rules; RF for feature rankings.

### Performance Considerations

- **Training Time**: Single trees: O(n log n); RF: O(n_estimators * n log n), but parallelizable.
- **Prediction Speed**: Trees: O(depth); RF: O(n_estimators * depth)—faster with fewer trees.
- **Memory Usage**: Trees: O(nodes); RF: O(n_estimators * nodes)—store only necessary for prediction.
- **Scalability**: RF handles thousands of features; use subsampling for millions of samples.
- **Bias-Variance**: Trees high variance (overfit); RF low variance via averaging.

### Advanced Techniques

- **Gradient Boosting**: Sequential trees (e.g., XGBoost, LightGBM) for higher accuracy over RF.
- **Extra Trees**: RF variant with fully random splits for faster training.
- **Feature Selection**: Iterate with RF importances to reduce dimensionality.
- **Hybrid Models**: Stack RF with neural networks or use in pipelines for end-to-end ML.
- **Extensions**: Isolation Forests for anomaly detection; RF regressors for time series with lags.

Decision Trees provide a foundational, interpretable approach to ML, while Random Forests enhance this with ensemble power for robust, high-performance predictions. Together, they form versatile tools for real-world applications, from diagnostics to forecasting, emphasizing the value of simplicity and aggregation in model design.

## Additional Resources

### Videos

- [Decision Tree Classification Clearly Explained!](https://www.youtube.com/watch?v=ZVR2Way4nwQ)
- [Random Forest Algorithm Clearly Explained!](https://www.youtube.com/watch?v=v6VJ2RO66Ag)
