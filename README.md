# Tree-Ensembles-and-Random-Forests---Lab

# Salary Prediction Using Tree Ensembles and Random Forests

## Project Overview
This project explores the implementation, evaluation, and structural inner workings of tree-based ensemble models using `scikit-learn`. Using personal demographic attributes extracted from the Census Bureau database, models are constructed to predict whether an individual's annual salary exceeds $50k (`>50k` vs `<=50k`). 

The core objective is to analyze performance variations between single structural estimators and variance-reducing bootstrap aggregations.


## Technical Objectives
* **Categorical Feature Processing:** Construct binary matrices from multi-class categorical indicators via dummy variable structures.
* **Baseline Modeling:** Establish structural decision limits with a singular baseline `DecisionTreeClassifier`.
* **Bootstrap Aggregation (Bagging):** Implement a `BaggingClassifier` to reduce prediction variance via parallel model voting.
* **Subspace Feature Sampling:** Deploy a `RandomForestClassifier` to analyze how randomized feature restrictions decouple individual tree correlation.
* **Ensemble Interpretation:** Isolate, visualize, and contrast the `feature_importances_` across separate tree instances inside the forest structure.


## Machine Learning Pipeline

### 1. Feature Preprocessing
* **Feature Scope:** Processed demographics including `Age`, `Education`, `Occupation`, `Relationship`, `Race`, and `Sex`.
* **Structural Matrix Conversion:** Handled high-cardinality categorical variables using dummy variable transformations via Pandas (`pd.get_dummies()`).
* **Validation Split:** Segregated features and target arrays using a 75/25 stratified train/test configuration to secure stable metrics.

### 2. Model Performance Summary
Ensemble mechanics systematically outpace the single estimator baseline by distributing variance boundaries

### 3. Key Discovery: Subspace Feature Sampling
By isolating individual estimators inside the `RandomForestClassifier` (`forest_2.estimators_`), we visualize the core mechanism behind random forests:

* **Tree 1** might heavily split data boundaries along features like `Age` and `Education_Bachelors`.
* **Tree 2** might optimize paths along entirely distinct columns, such as `Relationship_Husband` or `Occupation_Exec-managerial`.

Because each tree is forced to look at a small, randomized slice of features (`max_features`), they build fundamentally different decision logic. When combined, their individual errors cancel out, leaving a robust, stable ensemble prediction.