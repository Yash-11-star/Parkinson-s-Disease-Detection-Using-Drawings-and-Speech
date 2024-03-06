import os
import librosa
import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Load the saved model
model_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/Audio_model.joblib"
ensemble_model = joblib.load(model_filename)

# Load the feature selector
selector_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/Audio_selector.joblib"
selector = joblib.load(selector_filename)

# Load the scaler
scaler_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/Audio_scaler.joblib"
scaler = joblib.load(scaler_filename)

# Load the training data
data_filename = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/AudioData.csv"
data = pd.read_csv(data_filename)

# Split the data into features and labels
X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]

# Function to extract audio features using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Extract Chroma features
    mel = librosa.feature.melspectrogram(y=y, sr=sr)  # Extract Mel features
    features = np.concatenate([mfccs.mean(axis=1), chroma.mean(axis=1), mel.mean(axis=1)])
    return features

# Function to preprocess features
def preprocess_features(features, scaler, selector):
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Select features using the loaded selector
    selected_features = selector.transform(features_scaled)

    return selected_features

# Function to predict using the ensemble model
def predict(model, features, scaler, selector):
    # Preprocess features
    selected_features = preprocess_features(features, scaler, selector)

    # Predict using the provided model
    raw_prediction = model.predict(selected_features)[0]
    prediction = "Parkinson's" if raw_prediction == 1 else "Healthy"

    return prediction

# Example of using the saved model for prediction
new_data_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Parkinsons.wav"
# new_data_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Healthy.wav"
new_data = extract_features(new_data_path)
prediction = predict(ensemble_model, new_data, scaler, selector)
print(f"Prediction: {prediction}")
