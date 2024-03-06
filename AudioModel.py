import joblib
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

# Function to extract audio features using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Extract Chroma features
    mel = librosa.feature.melspectrogram(y=y, sr=sr)  # Extract Mel features
    features = np.concatenate([mfccs.mean(axis=1), chroma.mean(axis=1), mel.mean(axis=1)])
    return features

# Function to create a dataset from audio files with aggressive data augmentation
def create_augmented_dataset(data_folder, label):
    dataset = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_folder, file_name)
            y, sr = librosa.load(file_path, duration=3)
            features = extract_features(file_path)
            dataset.append(features.tolist() + [label])

            # Aggressive data augmentation
            for i in range(5):
                pitch_shifted_features = librosa.effects.pitch_shift(features[:-1], sr=sr, n_steps=np.random.uniform(-2, 2))
                augmented_data = np.concatenate([pitch_shifted_features, [label]])
                dataset.append(augmented_data.tolist())

    return dataset

# Load Healthy and Parkinson's datasets with aggressive data augmentation
augmented_healthy_data = create_augmented_dataset("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/HC_AH", 0)
augmented_parkinsons_data = create_augmented_dataset("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/PD_AH", 1)

# Combine the datasets
combined_data = augmented_healthy_data + augmented_parkinsons_data

# Create a DataFrame
df_columns = ['MFCC' + str(i) for i in range(1, 14)] + ['Chroma' + str(i) for i in range(1, 13)] + ['Mel' + str(i) for i in range(1, 129)] + ['Label']
df = pd.DataFrame(combined_data, columns=df_columns)

# Replace NaN values with zeros
df = df.fillna(0)

# Save the DataFrame to CSV
df.to_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv", index=False)

# Load the CSV data
data = pd.read_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv")

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
selector = SelectFromModel(RandomForestClassifier(random_state=42), threshold=0.01)
selector.fit(X_train_scaled, y_train)

X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Hyperparameter Tuning for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    # Add more parameters to tune
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_selected, y_train)

best_params_rf = grid_search_rf.best_params_
model_rf = grid_search_rf.best_estimator_

# Hyperparameter Tuning for GradientBoosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    # Add more parameters to tune
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train_selected, y_train)

best_params_gb = grid_search_gb.best_params_
model_gb = grid_search_gb.best_estimator_

# Support Vector Machine (SVM)
model_svm = SVC(probability=True, random_state=42)
model_svm.fit(X_train_selected, y_train)

# Neural Network (Multi-layer Perceptron) with increased complexity and iterations
model_nn = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=1000, random_state=42)
model_nn.fit(X_train_selected, y_train)

# Ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('rf', model_rf), ('gb', model_gb), ('svm', model_svm), ('nn', model_nn)], voting='soft')
ensemble_model.fit(X_train_selected, y_train)

# Evaluate the ensemble model
y_pred_ensemble = ensemble_model.predict(X_test_selected)

# Calculate confusion matrix, recall, and F1 score
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
recall_ensemble = np.diag(conf_matrix_ensemble) / np.sum(conf_matrix_ensemble, axis=1)
f1_score_ensemble = 2 * (recall_ensemble * np.mean(recall_ensemble)) / (recall_ensemble + np.mean(recall_ensemble))

print("Confusion Matrix:")
print(conf_matrix_ensemble)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))
print("\nRecall (per class):", recall_ensemble)
print("Average Recall:", np.mean(recall_ensemble))
print("F1 Score (per class):", f1_score_ensemble)
print("Average F1 Score:", np.mean(f1_score_ensemble))
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))

joblib.dump(ensemble_model, "Audio_model.joblib")
joblib.dump(scaler, "Audio_scaler.joblib")
joblib.dump(selector, "Audio_selector.joblib")

