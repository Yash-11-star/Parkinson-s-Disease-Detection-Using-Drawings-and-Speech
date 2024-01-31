
from joblib import dump
dump(clf, 'parkinsons_classifier.joblib')

# Function to predict Parkinson's disease from a new audio file
def predict_parkinsons(audio_file, model):
    features = extract_features(audio_file)
    prediction = model.predict([features])[0]
    return prediction

# Example usage for prediction
audio_file_path = 'path/to/your/test/audio.wav'  # Update with the path to your test audio file
loaded_model = joblib.load('parkinsons_classifier.joblib')  # Load the saved model
prediction = predict_parkinsons(audio_file_path, loaded_model)

if prediction == 1:
    print("The person may have Parkinson's disease.")
else:
    print("The person is likely healthy.")
