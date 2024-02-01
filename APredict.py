import os
import librosa
import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

# Load the saved model
model_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/Audio_model.joblib"
ensemble_model = joblib.load(model_filename)

# Load the feature selector
selector_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/selector.joblib"
selector = joblib.load(selector_filename)

# Load the training data
data_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv"
data = pd.read_csv(data_filename)

# Split the data into features and labels
X = data.iloc[:, :-1]
y_train = data.iloc[:, -1]

# Function to extract audio features using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Extract Chroma features
    mel = librosa.feature.melspectrogram(y=y, sr=sr)  # Extract Mel features
    features = np.concatenate([mfccs.mean(axis=1), chroma.mean(axis=1), mel.mean(axis=1)])
    return features

# Function to scale and select features
def preprocess_features(features, selector):
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))

    # Select features using the loaded selector
    selected_features = selector.transform(features_scaled)

    return selected_features

# Function to predict using the ensemble model
def predict_with_feedback(model, features, selector, y_train):
    # Preprocess features
    selected_features = preprocess_features(features, selector)

    # Print unique classes in y before fitting
    print("Unique classes in y:", np.unique(y_train))

    # Predict using the provided model
    raw_prediction = model.predict(selected_features)[0]
    
    # Convert raw prediction to a more human-readable format
    prediction = "Parkinson's" if raw_prediction == 1 else "Healthy"

    # Get user feedback (1 for correct, 0 for incorrect)
    feedback = int(input(f"Is the prediction ({prediction}) correct? Enter 1 for correct, 0 for incorrect: "))

    # If feedback is provided, update the model
    if feedback in [0, 1]:
        features_with_label = np.concatenate([selected_features.flatten(), [feedback]])

        # Check if there are still at least two unique classes in y_train after feedback
        unique_classes = np.unique(features_with_label[-1])
        if len(unique_classes) >= 2:
            model.fit(features_with_label[:-1].reshape(1, -1), [feedback])
        else:
            print("Not enough unique classes to update the model. Retrain the model with a larger dataset.")


# Example of using the saved model for prediction and feedback
# new_data_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Healthy.wav"
new_data_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Parkinsons.wav"
new_data = extract_features(new_data_path)
predict_with_feedback(ensemble_model, new_data, selector, y_train)

# To load the model for future predictions
loaded_model = joblib.load(model_filename)
prediction_on_loaded_model = loaded_model.predict(new_data.reshape(1, -1))[0]
print(f"Prediction using loaded model: {prediction_on_loaded_model}")
