

# Notebook 5: SHAP Explainability
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data and best model
X_train = np.load('X_train_selected.npy')
X_test = np.load('X_test_selected.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load the best performing model (e.g., XGBoost)
model = joblib.load('tuned_xgboost_model.pkl')

print(f"Test data shape: {X_test.shape}")

# 5.1 Create SHAP explainer
print("\n=== Creating SHAP Explainer ===")

# For tree-based models (XGBoost, Random Forest, LightGBM)
if hasattr(model, 'get_booster'):
    # XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
elif hasattr(model, 'booster_'):
    # LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
elif hasattr(model, 'estimators_'):
    # Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
else:
    # Fallback to KernelExplainer for SVM
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
    shap_values = explainer.shap_values(X_test[:100])

print("SHAP values computed successfully!")

# 5.2 Summary Plot
print("\n=== Summary Plot ===")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot - Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 Bar Plot (Mean SHAP values)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean |SHAP|)')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.4 Bee Swarm Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False, plot_size=(12, 8))
plt.title('SHAP Bee Swarm Plot')
plt.tight_layout()
plt.savefig('shap_bee_swarm.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.5 Individual Feature Dependence Plots
# Select top features
shap_values_abs = np.abs(shap_values).mean(0)
top_features_idx = np.argsort(shap_values_abs)[-6:][::-1]

# Create dependence plots for top features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature_idx in enumerate(top_features_idx):
    shap.dependence_plot(
        feature_idx, shap_values, X_test,
        ax=axes[idx], show=False
    )
    axes[idx].set_title(f'Feature {feature_idx} Dependence Plot')

plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.6 Waterfall Plot for a single prediction
plt.figure(figsize=(14, 8))
# Use first test sample for demonstration
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                    base_values=explainer.expected_value,
                                    data=X_test[0]),
                   show=False)
plt.title('SHAP Waterfall Plot - Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.7 Force Plot (for a single prediction)
plt.figure(figsize=(20, 4))
shap.initjs()

# Create force plot
force_plot = shap.force_plot(explainer.expected_value, shap_values[0],
                             X_test[0], matplotlib=True, show=False)
plt.title('SHAP Force Plot - Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.8 Decision Plot
plt.figure(figsize=(12, 8))
# Show first 10 samples
shap.decision_plot(explainer.expected_value, shap_values[:10],
                   X_test[:10], show=False)
plt.title('SHAP Decision Plot - First 10 Samples')
plt.tight_layout()
plt.savefig('shap_decision_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.9 Feature importance summary table
feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
feature_importance_shap = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_values_abs
})
feature_importance_shap = feature_importance_shap.sort_values('SHAP_Value', ascending=False)

print("\n=== Top 10 Features by SHAP Importance ===")
print(feature_importance_shap.head(10))

# 5.10 Save SHAP results
feature_importance# Notebook 5: SHAP Explainability
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data and best model
X_train = np.load('X_train_selected.npy')
X_test = np.load('X_test_selected.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load the best performing model (e.g., XGBoost)
model = joblib.load('tuned_xgboost_model.pkl')

print(f"Test data shape: {X_test.shape}")

# 5.1 Create SHAP explainer
print("\n=== Creating SHAP Explainer ===")

# For tree-based models (XGBoost, Random Forest, LightGBM)
if hasattr(model, 'get_booster'):
    # XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
elif hasattr(model, 'booster_'):
    # LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
elif hasattr(model, 'estimators_'):
    # Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
else:
    # Fallback to KernelExplainer for SVM
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
    shap_values = explainer.shap_values(X_test[:100])

print("SHAP values computed successfully!")

# 5.2 Summary Plot
print("\n=== Summary Plot ===")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot - Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 Bar Plot (Mean SHAP values)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean |SHAP|)')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.4 Bee Swarm Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False, plot_size=(12, 8))
plt.title('SHAP Bee Swarm Plot')
plt.tight_layout()
plt.savefig('shap_bee_swarm.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.5 Individual Feature Dependence Plots
# Select top features
shap_values_abs = np.abs(shap_values).mean(0)
top_features_idx = np.argsort(shap_values_abs)[-6:][::-1]

# Create dependence plots for top features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature_idx in enumerate(top_features_idx):
    shap.dependence_plot(
        feature_idx, shap_values, X_test,
        ax=axes[idx], show=False
    )
    axes[idx].set_title(f'Feature {feature_idx} Dependence Plot')

plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.6 Waterfall Plot for a single prediction
plt.figure(figsize=(14, 8))
# Use first test sample for demonstration
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                    base_values=explainer.expected_value,
                                    data=X_test[0]),
                   show=False)
plt.title('SHAP Waterfall Plot - Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.7 Force Plot (for a single prediction)
plt.figure(figsize=(20, 4))
shap.initjs()

# Create force plot
force_plot = shap.force_plot(explainer.expected_value, shap_values[0],
                             X_test[0], matplotlib=True, show=False)
plt.title('SHAP Force Plot - Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.8 Decision Plot
plt.figure(figsize=(12, 8))
# Show first 10 samples
shap.decision_plot(explainer.expected_value, shap_values[:10],
                   X_test[:10], show=False)
plt.title('SHAP Decision Plot - First 10 Samples')
plt.tight_layout()
plt.savefig('shap_decision_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.9 Feature importance summary table
feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
feature_importance_shap = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_values_abs
})
feature_importance_shap = feature_importance_shap.sort_values('SHAP_Value', ascending=False)

print("\n=== Top 10 Features by SHAP Importance ===")
print(feature_importance_shap.head(10))

# 5.10 Save SHAP results
feature_importance
