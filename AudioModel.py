import joblib
import os
import tensorflow as tf
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Function to extract audio features using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    n_fft = 512  
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)  # Extract MFCC features with reduced n_fft
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)  # Extract Chroma features with reduced n_fft
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)  # Extract Mel features with reduced n_fft
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
# print(df.columns)
# print(df)
# Save the DataFrame to CSV
# df.to_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv", index=False)

# Load the CSV data
# data = pd.read_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv")

# Split the data into features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Data Augmentation
# No additional data augmentation is needed as it's already applied during dataset creation.

# Balancing the Dataset
oversampler = RandomOverSampler(random_state=42)
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
selector = SelectFromModel(GradientBoostingClassifier(random_state=42), threshold=0.01)
selector.fit(X_train_scaled, y_train)

X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Hyperparameter Tuning for GradientBoosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train_selected, y_train)

best_params_gb = grid_search_gb.best_params_
model_gb = grid_search_gb.best_estimator_

# Evaluate the Gradient Boosting model
y_pred_gb = model_gb.predict(X_test_selected)

# Calculate confusion matrix, recall, and F1 score
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
recall_gb = np.diag(conf_matrix_gb) / np.sum(conf_matrix_gb, axis=1)
f1_score_gb = 2 * (recall_gb * np.mean(recall_gb)) / (recall_gb + np.mean(recall_gb))

print("Confusion Matrix:")
print(conf_matrix_gb)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))
print("\nRecall (per class):", recall_gb)
print("Average Recall:", np.mean(recall_gb))
print("F1 Score (per class):", f1_score_gb)
print("Average F1 Score:", np.mean(f1_score_gb))
print("Gradient Boosting Model Accuracy:", accuracy_score(y_test, y_pred_gb))

if isinstance(model_gb, GradientBoostingClassifier):
    for i, tree in enumerate(model_gb.estimators_):
        plt.figure(figsize=(10, 10))
        plot_tree(tree[0], filled=True, feature_names=X.columns)
        plt.title(f'Decision Tree {i+1}')
        plt.show()

# Summary of the best parameters found during hyperparameter tuning
print("Best Parameters:")
print(best_params_gb)

# Summary of the trained Gradient Boosting model
print("\nGradient Boosting Model Summary:")
print(model_gb)


# joblib.dump(model_gb, "Audio_model_gb.joblib")
# joblib.dump(scaler, "Audio_scaler_gb.joblib")
# joblib.dump(selector, "Audio_selector_gb.joblib")
