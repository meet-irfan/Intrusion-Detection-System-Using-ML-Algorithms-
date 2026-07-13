# Intrusion-Detection-System-Using-ML-Algorithms-
An Explainable Hybrid Machine Learning Framework for Network Intrusion Detection Using Feature Selection and Ensemble Learning
Step-by-Step Implementation Process
Phase 1: Problem Definition
Objective

Develop a machine learning model to detect network intrusions using network traffic data and compare multiple algorithms while making their predictions explainable.

Research Question

Which machine learning algorithm performs best for intrusion detection?
Does feature selection improve performance?
Can SHAP explain why the model classifies a connection as an attack?
Phase 2: Dataset Selection

Use one dataset only for the first paper.

I recommend:

CIC-IDS2017

More widely used
Balanced attack scenarios
Easier to understand
Suitable for Scopus-quality research

The dataset contains network flow records such as:

Feature	Description
Flow Duration	Duration of the connection
Total Fwd Packets	Forward packets
Total Backward Packets	Backward packets
Flow Bytes/s	Bytes per second
Packet Length Mean	Average packet size
Flow IAT Mean	Average inter-arrival time
Destination Port	Destination port
Protocol	TCP/UDP
Label	Normal or Attack

Approximately 80+ features are available.

Phase 3: Data Preprocessing

This is one of the most important stages.

Step 3.1 Load Dataset
import pandas as pd

df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.csv")
Step 3.2 Explore Data

Check:

Shape
df.shape

Columns

df.columns

Data types

df.info()

Missing values

df.isnull().sum()

Duplicate rows

df.duplicated().sum()

Class distribution

df['Label'].value_counts()

This helps you understand whether the dataset is imbalanced.

Step 3.3 Remove Unnecessary Columns

Sometimes columns such as IDs or timestamps are not useful.

Example:

Flow ID
Source IP
Destination IP
Timestamp

These may introduce noise.

Step 3.4 Handle Missing Values

If few values are missing:

drop rows

Otherwise:

fill with median
Step 3.5 Remove Infinite Values

Many CIC datasets contain

Infinity
-Infinity

Replace them.

import numpy as np

df.replace([np.inf,-np.inf],np.nan)
Step 3.6 Handle Missing Again
df.dropna(inplace=True)
Step 3.7 Encode Labels

Normal

↓

0

Attack

↓

1

or multiclass

DoS
PortScan
Bot
Heartbleed

using LabelEncoder.

Step 3.8 Feature Scaling

Required for

SVM
KNN
Logistic Regression

Not necessary for

Random Forest
XGBoost
LightGBM

Still, create a scaled version for fair comparison.

Phase 4: Exploratory Data Analysis (EDA)

Produce plots.

Histogram

Feature distribution

Boxplot

Outliers

Heatmap

Correlation

Countplot

Attack distribution

Pairplot

Relationships between features

Phase 5: Feature Engineering

This is where your paper becomes more interesting.

Instead of using all 80 features:

Select the most informative ones.

Method 1

Correlation

Remove highly correlated features.

Method 2

Random Forest Feature Importance

Rank features.

Method 3

Mutual Information

Measure information gain.

Method 4

Recursive Feature Elimination (RFE)

Automatically remove weak features.

Method 5 (Best)

Hybrid Feature Selection

Combine

Correlation filtering
Mutual Information
Random Forest importance

This becomes one of your novel contributions.

Phase 6: Train-Test Split

Example

80% Training

20% Testing

or

70%

30%

Use stratified sampling because of class imbalance.

Phase 7: Model Training

Train one model at a time.

Model 1

Random Forest

Model 2

XGBoost

Model 3

LightGBM

Model 4

Support Vector Machine

Tune hyperparameters using GridSearchCV or RandomizedSearchCV.

Phase 8: Model Evaluation

For each model calculate:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Training Time
Prediction Time

Create comparison tables and plots.

Phase 9: Explainable AI

Use SHAP on the best-performing model.

Generate:

SHAP Summary Plot
SHAP Bar Plot
SHAP Waterfall Plot
SHAP Force Plot

Discuss which features contribute most to attack detection.

Phase 10: Ensemble Model

After evaluating individual models, create an ensemble.

Possible approaches:

Voting Classifier (soft or hard voting)
Stacking Classifier
Blending

Compare the ensemble against the individual models to see if it improves performance.

Phase 11: Statistical Validation

To strengthen the paper:

Perform k-fold cross-validation (e.g., 5-fold or 10-fold).
Report mean and standard deviation of key metrics.
If comparing models formally, consider statistical significance tests (e.g., paired t-test or Wilcoxon signed-rank test) where appropriate.
Phase 12: Results and Discussion

Summarize:

Which model performed best?
Did hybrid feature selection improve performance?
Did the ensemble outperform individual models?
Which features were most influential according to SHAP?
What limitations remain?
Suggested implementation timeline
Download and inspect the dataset.
Clean and preprocess the data.
Perform EDA.
Build the hybrid feature selection pipeline.
Train the four baseline models.
Tune hyperparameters.
Build and evaluate the ensemble.
Apply SHAP to explain the best model.
Prepare tables, figures, and write the methodology and results sections.

This workflow will produce a solid implementation that can support a high-quality machine learning research paper.
