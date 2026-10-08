# Phase 3 Teach-back: Machine Learning Model Development

## Supervised machine learning

Supervised machine learning uses labeled examples to learn a relationship between inputs and outputs.

In this project, each molecule is an example. The descriptor values are the inputs, and the experimental aqueous solubility is the output. The supervised model learns patterns from the training examples and then uses those patterns to predict LogS for new molecules.

The workflow is:

**Molecular descriptors → supervised regression model → predicted LogS**

## Regression versus classification

A classification model predicts a category or class. Examples include whether a molecule is active or inactive, or whether a compound belongs to a particular chemical series.

A regression model predicts a continuous numerical value. Here, the value is the aqueous solubility measured as LogS. Regression is appropriate because solubility can take many numerical values rather than a small set of discrete classes.

## Features and target

The feature matrix is usually called `X`, while the target vector is called `y`.

- `X` contains the input variables: MolWt, LogP, TPSA, HBD, HBA, and RotBonds.
- `y` contains the experimentally measured LogS values.

The model learns a function:

$$
\hat{y} = f(X)
$$

where $\hat{y}$ is the predicted LogS value.

The existing ESOL-predicted LogS column is not used as an input. Using it would provide a model-generated value rather than information derived from the molecule itself and could create data leakage.

## Training versus testing data

The dataset is divided into separate training and testing sets.

- The **training set** is used to fit the model parameters.
- The **testing set** is kept separate and used only for an initial evaluation.

This separation helps assess how well the model might generalize to molecules it has not seen during training.

The split uses `test_size=0.20` and `random_state=42`. The first value means that 20% of the molecules are reserved for testing. The second value makes the split reproducible, so the same data allocation is obtained when the script is run again.

## Why we use a baseline model

A baseline model provides a simple reference point. It does not necessarily represent the best available model, but it helps answer a useful question:

> Does the main model perform better than a simple prediction strategy?

The baseline used in Phase 3 is `DummyRegressor(strategy="mean")`. It predicts the mean LogS value from the training set for every molecule in the test set. It is easy to understand and provides a numerical benchmark.

If the Random Forest model cannot outperform the baseline, the result suggests that the selected descriptors, model configuration, or dataset split may need further investigation. This is an initial evaluation, not a conclusion about the final model quality.

## How Random Forest regression works

A Random Forest is an ensemble of many decision trees. Each tree is trained on a bootstrap sample of the training data and considers a random subset of features when choosing a split.

The forest combines the predictions from all trees. For regression, the final prediction is commonly the average of the tree predictions.

Decision trees divide the feature space into regions using rules such as:

- Is molecular weight less than or greater than a threshold?
- Is LogP greater than a selected value?

The ensemble reduces the risk that a single tree will memorize noise in the training data. Random forests are generally useful because they handle nonlinear relationships and can work well with tabular chemical descriptors.

## Decision trees and ensemble learning

A decision tree is a hierarchical series of yes-or-no questions. Each question creates a branch, and the tree eventually assigns a prediction to each final region.

An ensemble combines predictions from multiple models. Random Forest uses many trees, where each tree contributes one prediction. Averaging these predictions can improve stability compared with using a single tree.

The term "random" refers to two sources of randomness:

1. Each tree receives a random bootstrap sample of the data.
2. Each split considers only a random subset of features.

## Overfitting and generalization

A model that perfectly fits the training data may still perform poorly on new molecules. This is called overfitting.

The goal is to obtain good generalization: accurate predictions on molecules that were not used to train the model.

The held-out test set provides an initial estimate of generalization. A low training error accompanied by a much higher test error can indicate overfitting. The current Phase 3 evaluation is initial and limited because it uses one data split and does not include cross-validation, uncertainty analysis, or independent external validation.

## What `random_state=42` means

`random_state` controls the random number generator used by the model. Setting it to `42` makes the split and Random Forest training reproducible.

Using the same random seed produces the same data split and model initialization, which is valuable for debugging, comparing experiments, and documenting results.

It does not guarantee that the model is scientifically optimal. It only makes the process reproducible.

## MAE, RMSE, and R²

### Mean Absolute Error (MAE)

MAE is the average absolute difference between the predicted and experimental LogS values.

For example, if the actual values are $[-1.0, -2.0, -3.0]$ and the predictions are $[-1.5, -2.0, -2.0]$, the absolute errors are $[0.5, 0, 1.0]$. The MAE is:

$$
\text{MAE} = \frac{0.5 + 0 + 1.0}{3} = 0.5
$$

MAE is easy to interpret because it is expressed in the same units as LogS.

### Root Mean Squared Error (RMSE)

RMSE squares the prediction errors, averages them, and then takes the square root.

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

RMSE gives more weight to large errors than MAE. A larger RMSE indicates that some predictions are substantially different from the experimental values.

### R² score

R² represents the proportion of variation in the target that is explained by the model.

An R² of 1.0 indicates a perfect fit. An R² of 0.0 means that the model performs as well as predicting the mean LogS value. A negative R² means that the model performs worse than the mean baseline.

R² is useful for comparing models, but it should not be interpreted alone. A model can have a reasonable R² while still having chemically important prediction errors, especially near the extremes of solubility.

## Why experimental LogS is the target

The target is the experimentally measured aqueous solubility. This is the quantity we want to predict. It is directly connected to the known chemical property rather than an earlier estimate.

The existing ESOL-predicted LogS values are useful as historical modeling outputs, but they should not be used as the Phase 3 target because that would make the target dependent on another model. Using experimental measurements keeps the learning problem grounded in observed chemistry.

## Why feature order matters when loading a model

A trained scikit-learn model expects its input features in a defined order. The model learns from the columns as they were arranged during training.

If the order is changed when using the model, the model may interpret the wrong descriptor values as the wrong features. This can produce incorrect predictions even when the same descriptors are present.

The saved feature-order file preserves the original order:

```text
MolWt, LogP, TPSA, HBD, HBA, RotBonds
```

The saved model and feature columns should be loaded together to maintain the correct feature alignment.

## Medicinal chemistry and SAR context

Structure–activity relationship analysis seeks relationships between molecular structure and a biological or physicochemical outcome.

In this project, molecular descriptors are compact numerical summaries of structure. A model can learn relationships between descriptors and measured LogS, but it should not be treated as a complete description of molecular behavior.

The descriptors capture selected properties such as molecular size, lipophilicity, polarity, hydrogen bonding, and flexibility. They do not include all structural information or all factors that influence solubility.

This Phase 3 model is therefore an initial statistical approximation that can support SAR analysis. Future evaluation will examine whether the model generalizes to new molecules and whether its errors correlate with chemical regions or property ranges.

## Interview questions and answers

### 1. What is the difference between features and target data?

Features are the molecular descriptors used as model inputs. The target is the experimental LogS value that the model tries to predict.

### 2. Why do we keep the testing data separate from model training?

The test set provides an independent estimate of how the model performs on molecules it has not seen during training. This supports a more realistic assessment of generalization.

### 3. Why is a baseline model useful?

A baseline establishes a simple reference prediction. It helps show whether the more complex model has learned something meaningful rather than simply predicting an average value.

### 4. How does a Random Forest make predictions?

A Random Forest combines predictions from many decision trees. Each tree is trained on a sample of the data, and the final prediction is typically the average of the ensemble's predictions.

### 5. Why does feature order need to be preserved?

The model learned relationships between specific columns and their corresponding descriptor values. Reordering the columns can make the model receive the wrong information.

## Two-minute explanation of Phase 3

"Phase 3 adds supervised machine learning to the descriptor dataset created in Phase 2. We separate the six calculated descriptors into the feature matrix and the experimentally measured LogS value into the target vector. The dataset is divided into training and testing sets using a reproducible random split.

We first train a baseline model that predicts the mean LogS value from the training set. This provides a reference point for evaluating the main model. We then train a Random Forest regressor with 200 trees. The Random Forest uses many decision trees and averages their predictions to capture nonlinear relationships between molecular structure and aqueous solubility.

The held-out testing set is used to calculate MAE, RMSE, and R². MAE measures average prediction error, RMSE gives greater weight to large errors, and R² describes how much variance in LogS is explained by the model. The model is saved together with its feature-column order so that future predictions use the descriptors in the correct sequence.

This Phase 3 implementation is an initial model-development step. It does not yet represent comprehensive validation, and Phase 4 will address broader evaluation, uncertainty, and model comparison."
