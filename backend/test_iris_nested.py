import os
import pandas as pd
import pickle
from datetime import datetime
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from trackmate import TrackMateSDK

# Define the "trackdata" directory path and ensure it exists
trackdata_base_dir = "../trackdata"
os.makedirs(trackdata_base_dir, exist_ok=True)

# Initialize the SDK
sdk = TrackMateSDK()

# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
iris_df['target'] = y

# Split into train and test sets
train_df, test_df = train_test_split(iris_df, test_size=0.3, random_state=42, stratify=iris_df["target"])
X_train = train_df[iris.feature_names]
y_train = train_df["target"]
X_test = test_df[iris.feature_names]
y_test = test_df["target"]

# Experiment Name
EXPERIMENT_NAME = "IRIS_dataset_classification_with_nested_runs"

# Create an experiment or get an existing one
experiment = sdk.create_experiment(name=EXPERIMENT_NAME, description="Iris dataset classification with RandomForest and nested runs")
if "experiment_id" not in experiment:
    raise ValueError(f"Failed to create or retrieve experiment: {experiment}")
experiment_id = experiment["experiment_id"]

# Start the main run (parent run)
main_run = sdk.start_run(experiment_id=experiment_id, run_name="main_run")
if "run_id" not in main_run:
    raise ValueError(f"Failed to start the main run: {main_run}")
main_run_id = main_run["run_id"]

# Define the "trackdata" directory for the main run and ensure it exists
main_run_dir = os.path.join(trackdata_base_dir, f"main_run_{main_run_id}")
os.makedirs(main_run_dir, exist_ok=True)

# Log parameters for the main run
sdk.log_param(run_id=main_run_id, key="experiment", value="main_run")

# Nested Run 1: RandomForest with max_depth = 5
nested_run_1 = sdk.start_run(experiment_id=experiment_id, run_name="nested_run_1", nested=True)
if "run_id" not in nested_run_1:
    raise ValueError(f"Failed to start nested run 1: {nested_run_1}")
nested_run_1_id = nested_run_1["run_id"]

# Create directory for Nested Run 1 within the main run directory
nested_run_1_dir = os.path.join(main_run_dir, f"nested_run_{nested_run_1_id}")
os.makedirs(nested_run_1_dir, exist_ok=True)

# Log parameters for Nested Run 1
sdk.log_param(run_id=nested_run_1_id, key="max_depth", value="5")
sdk.log_param(run_id=nested_run_1_id, key="n_estimators", value="100")
sdk.log_param(run_id=nested_run_1_id, key="random_state", value="0")

# Train RandomForest with max_depth = 5
clf_1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf_1.fit(X_train, y_train)
iris_predict_y_1 = clf_1.predict(X_test)

# Log metrics for Nested Run 1
roc_auc_score_1 = roc_auc_score(y_test, clf_1.predict_proba(X_test), multi_class='ovr')
sdk.log_metric(run_id=nested_run_1_id, key="test_roc_auc_score", value=roc_auc_score_1)
accuracy_1 = accuracy_score(y_test, iris_predict_y_1)
sdk.log_metric(run_id=nested_run_1_id, key="test_accuracy_score", value=accuracy_1)

# Save the model artifact locally in the nested run directory
model_1_path = os.path.join(nested_run_1_dir, "iris_rf_model_max_depth_5.pkl")
with open(model_1_path, "wb") as model_file:
    pickle.dump(clf_1, model_file)

# Log the model as an artifact
sdk.log_artifact(run_id=nested_run_1_id, file_path=model_1_path)

# Nested Run 2: RandomForest with max_depth = 10
nested_run_2 = sdk.start_run(experiment_id=experiment_id, run_name="nested_run_2", nested=True)
if "run_id" not in nested_run_2:
    raise ValueError(f"Failed to start nested run 2: {nested_run_2}")
nested_run_2_id = nested_run_2["run_id"]

# Create directory for Nested Run 2 within the main run directory
nested_run_2_dir = os.path.join(main_run_dir, f"nested_run_{nested_run_2_id}")
os.makedirs(nested_run_2_dir, exist_ok=True)

# Log parameters for Nested Run 2
sdk.log_param(run_id=nested_run_2_id, key="max_depth", value="10")
sdk.log_param(run_id=nested_run_2_id, key="n_estimators", value="100")
sdk.log_param(run_id=nested_run_2_id, key="random_state", value="0")

# Train RandomForest with max_depth = 10
clf_2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf_2.fit(X_train, y_train)
iris_predict_y_2 = clf_2.predict(X_test)

# Log metrics for Nested Run 2
roc_auc_score_2 = roc_auc_score(y_test, clf_2.predict_proba(X_test), multi_class='ovr')
sdk.log_metric(run_id=nested_run_2_id, key="test_roc_auc_score", value=roc_auc_score_2)
accuracy_2 = accuracy_score(y_test, iris_predict_y_2)
sdk.log_metric(run_id=nested_run_2_id, key="test_accuracy_score", value=accuracy_2)

# Save the model artifact locally in the nested run directory
model_2_path = os.path.join(nested_run_2_dir, "iris_rf_model_max_depth_10.pkl")
with open(model_2_path, "wb") as model_file:
    pickle.dump(clf_2, model_file)

# Log the model as an artifact
sdk.log_artifact(run_id=nested_run_2_id, file_path=model_2_path)

# Finish the main run
print(f"Main Run ID: {main_run_id}")
print(f"Nested Run 1 ID: {nested_run_1_id}")
print(f"Nested Run 2 ID: {nested_run_2_id}")
