import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
parkinsons = pd.read_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioData.csv")

# Shuffling and splitting the data
parkinsons = parkinsons.sample(frac=1, random_state=42).copy()
X = parkinsons.drop(["name", "status"], axis=1)
y = parkinsons["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=51)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=51, n_jobs=-1)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                           cv=StratifiedKFold(n_splits=5), scoring='recall', n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the final model with best hyperparameters
final_rf_model = grid_search.best_estimator_
final_rf_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
final_predictions = final_rf_model.predict(X_test_scaled)

# Evaluate the final model
accuracy = accuracy_score(y_test, final_predictions)
conf_matrix = confusion_matrix(y_test, final_predictions)
classification_report_str = classification_report(y_test, final_predictions)

print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report_str)

# Save the final model and scaler for deployment
joblib.dump(final_rf_model, "Audio_model.joblib")
joblib.dump(scaler, "scaler.joblib")
