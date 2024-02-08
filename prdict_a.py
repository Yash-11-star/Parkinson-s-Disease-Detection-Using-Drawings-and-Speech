import joblib
import librosa
import numpy as np

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to predict Parkinson's disease from a new audio file
def predict_parkinsons(audio_file, model):
    features = extract_features(audio_file)
    prediction = model.predict([features])[0]
    return prediction

# Example usage for prediction
audio_file_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/HC_AH/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav"  # Update with the path to your test audio file
loaded_model = joblib.load('Audio_model.joblib')  # Load the saved model
prediction = predict_parkinsons(audio_file_path, loaded_model)

if prediction == 1:
    print("The person may have Parkinson's disease.")
else:
    print("The pe rson is likely healthy.")

