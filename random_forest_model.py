# Import all necessary libraries.
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from sklearn.impute import SimpleImputer
from collections import Counter

# ----------------------------------------------
# 1. Load and Chronologically Sort the Data
# ----------------------------------------------

# Load and read the dataset.
df = pd.read_csv("PaulSkenes_data.csv")

# Find and gather columns that define the chronological order of pitches.
sort_cols = [

    "game_date",
    "game_pk",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",

]

# Sort the data in real time.
df = df.sort_values(sort_cols).reset_index(drop=True)

# ----------------------------------------------
# 2. Create Next-Pitch Targets (type + zone)
# ----------------------------------------------

# Shift the pitch type up by one row so each current pitch row has the next pitch type.
df["next_pitch_type"] = (df.groupby(["game_pk", "pitcher"])["pitch_type"].shift(-1))

# Similarly, shift the zone up by one row.
df["next_zone"] = (df.groupby(["game_pk", "at_bat_number"])["zone"].shift(-1))

# Only keep the rows where both pitch type and zone are known.
df_model = df.dropna(subset=["next_pitch_type", "next_zone"]).copy()

# Rename the current pitch type as the previous pitch type to use as a feature.
df_model = df_model.rename(columns={"pitch_type": "prev_pitch_type"})

# ----------------------------------------------
# 3. Runner State Feature Engineering
# ----------------------------------------------

# Convert runner existence (NaN or player ID) into simple 0/1 flags.
for base_col, new_col in [

    ("on_1b", "runner_on_1b"),
    ("on_2b", "runner_on_2b"),
    ("on_3b", "runner_on_3b"),

]:
    df_model[new_col] = df_model[base_col].notna().astype(int)

# ----------------------------------------------
# 4. Build Feature Set
# ----------------------------------------------

# Define contextual features like game state, batter/pitcher attributes, and previous pitch outcomes.
context_features = [

    "balls", "strikes", "outs_when_up",
    "inning", "inning_topbot",
    "bat_score", "fld_score",
    "home_score", "away_score",
    "stand", "p_throws",
    "game_type",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "prev_pitch_type",
    "type",
    "zone",

]

# Physical/mechanical measurements of the previous pitch.
ball_flight_cols = [

    "release_speed", "effective_speed",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "break_angle_deprecated", "break_length_deprecated",
    "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in",
    "spin_rate_deprecated", "release_spin", "spin_axis",
    "release_extension",

]

# Only keep physical columns that exist in this dataset.
ball_flight_cols = [c for c in ball_flight_cols if c in df_model.columns]

# Create new columns prefixed with "prev_" for each ball-flight metric
ball_flight_features = []
for col in ball_flight_cols:
    new_col = f"prev_{col}"
    df_model[new_col] = df_model[col]
    ball_flight_features.append(new_col)

# Gather all final input features.
feature_cols = context_features + ball_flight_features

# Output the list of input features used.
print("\nUsing features:")
for c in feature_cols:
    print("  -", c)

# ----------------------------------------------
# 5. Prepare X, y_type, y_zone
# ----------------------------------------------

# Define predictor matrix X and target labels y_type and y_zone.
X = df_model[feature_cols].copy()
y_type = df_model["next_pitch_type"].copy()
y_zone = df_model["next_zone"].copy()

# ----------------------------------------------
# 6. Chronological Train/Test Split (80/20)
# ----------------------------------------------

# Establish an 80/20 train/test split.
split_idx = int(0.8 * len(X))

# Split so that training data are only PAST pitches
X_train = X.iloc[:split_idx].copy()
y_type_train = y_type.iloc[:split_idx].copy()
y_zone_train = y_zone.iloc[:split_idx].copy()

# Split so that test data are only FUTURE pitches.
X_test = X.iloc[split_idx:].copy()
y_type_test = y_type.iloc[split_idx:].copy()
y_zone_test = y_zone.iloc[split_idx:].copy()

# Output the train and test split sizes.
print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# ----------------------------------------------
# 7. Encode predictors (fit on train, apply to test)
# ----------------------------------------------

# Split columns by data types.
num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# Drop numeric columns that are missing.
all_nan_cols = [c for c in num_cols if X_train[c].isna().all()]
if all_nan_cols:

    print("Dropping all-NaN numeric columns:", all_nan_cols)
    X_train = X_train.drop(columns=all_nan_cols)
    X_test = X_test.drop(columns=all_nan_cols)
    num_cols = [c for c in num_cols if c not in all_nan_cols]

# Impute numeric features using medians from the training data.
num_imputer = SimpleImputer(strategy="median")
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Encode categorical strings as integer codes.
encoders = {}
for col in cat_cols:

    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# --------------------------------------------------------------
# 8. Encode Target Labels for Type + Zone Predictions
# --------------------------------------------------------------

# Create separate label encoders.
y_type_le = LabelEncoder()
y_zone_le = LabelEncoder()

# Fit the pitch type encoders on training data and convert them to integer class IDs.
y_type_train_enc = y_type_le.fit_transform(y_type_train.astype(str))

# Use the mapping obtained above to transform the pitch type testing labels, no refitting done.
y_type_test_enc  = y_type_le.transform(y_type_test.astype(str))

# Fir the pitch zone encoders on training data and convert them to integer class IDs.
y_zone_train_enc = y_zone_le.fit_transform(y_zone_train.astype(str))

# Use the mapping obtained above to transform the pitch zone testing labels, no refitting done.
y_zone_test_enc  = y_zone_le.transform(y_zone_test.astype(str))

# Show distribution to understand class imbalance
print("\nLabel distribution (overall, type):")
print(y_type.value_counts(normalize=True))

print("\nMajority-class baseline (always guess most common type):")
test_counts = Counter(y_type_test_enc)
maj_baseline = max(test_counts.values()) / len(y_type_test_enc)
print(maj_baseline)

# ----------------------------------------------
# 9. Train Random Forest models
# ----------------------------------------------

# Establish the Random Forest Classifier for pitch types using balanced class weighting.
rf_type = RandomForestClassifier(

    n_estimators=400,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,

)

# Fit the Random Forest Classifier.
rf_type.fit(X_train, y_type_train_enc)

# Establish the Random Forest Classifier similarly to the pitch type one, but for zones.
rf_zone = RandomForestClassifier(

    n_estimators=400,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,

)

# Fir the Random Forest Classifier
rf_zone.fit(X_train, y_zone_train_enc)

# ----------------------------------------------
# 10. Predictions
# ----------------------------------------------

# Make the predictions for pitch types and zones.
y_type_pred = rf_type.predict(X_test)
y_zone_pred = rf_zone.predict(X_test)

# ----------------------------------------------
# 11. Evaluation: pitch TYPE
# ----------------------------------------------

# Output the accuracy, balanced accuracy, and a classification report for predicted pitch types.
print("\n=== Next-Pitch TYPE Accuracy ===")
print(accuracy_score(y_type_test_enc, y_type_pred))

print("\nBalanced accuracy (type):")
print(balanced_accuracy_score(y_type_test_enc, y_type_pred))

print("\n=== Next-Pitch TYPE Classification Report ===")
print(classification_report(

    y_type_test_enc,
    y_type_pred,
    target_names=y_type_le.classes_,
    zero_division=0,

    )
)

# Create and display a confusion matrix for pitch types.
cm_type = confusion_matrix(y_type_test_enc, y_type_pred)
disp_type = ConfusionMatrixDisplay(

    confusion_matrix=cm_type,
    display_labels=y_type_le.classes_,

)

fig, ax = plt.subplots(figsize=(8, 6))
disp_type.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Next-Pitch Type Confusion Matrix")
plt.tight_layout()
plt.show()

# Get feature importances from the pitch type model.
importances = rf_type.feature_importances_

# Sort the features from most important to least important.
idx = np.argsort(importances)[::-1]

# Create and display the feature importance graph.
plt.figure(figsize=(12, 6))
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)), [feature_cols[i] for i in idx], rotation=90)
plt.title("Feature Importances for Next-Pitch TYPE Prediction")
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 12. Evaluation: ZONE
# ----------------------------------------------

# Output the accuracy, balanced accuracy, and classification report for pitch zones.
print("\n=== Next-ZONE Accuracy ===")
print(accuracy_score(y_zone_test_enc, y_zone_pred))

print("\nBalanced accuracy (zone):")
print(balanced_accuracy_score(y_zone_test_enc, y_zone_pred))

print("\n=== Next-ZONE Classification Report ===")
print(classification_report(

    y_zone_test_enc,
    y_zone_pred,
    target_names=y_zone_le.classes_,
    zero_division=0,
    )
)

# Create and display the confusion matrix for pitch zones.
cm_zone = confusion_matrix(y_zone_test_enc, y_zone_pred)
disp_zone = ConfusionMatrixDisplay(

    confusion_matrix=cm_zone,
    display_labels=y_zone_le.classes_,

)

fig, ax = plt.subplots(figsize=(8, 6))
disp_zone.plot(ax=ax, cmap="Greens", xticks_rotation=45)
plt.title("Next-Pitch Zone Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 13. Feature Importances: ZONE model
# ----------------------------------------------

# Get feature importances from the pitch zone model.
zone_importances = rf_zone.feature_importances_

# Sort the features from most to least important
idx_zone = np.argsort(zone_importances)[::-1]

# Create and display the feature importance graph.
plt.figure(figsize=(12, 6))
plt.bar(range(len(idx_zone)), zone_importances[idx_zone])
plt.xticks(range(len(idx_zone)), [feature_cols[i] for i in idx_zone], rotation=90)
plt.title("Feature Importances for Next-Pitch ZONE Prediction")
plt.tight_layout()
plt.show()