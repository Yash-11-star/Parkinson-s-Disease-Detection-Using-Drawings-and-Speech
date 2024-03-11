import numpy as np
import joblib
import librosa
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, message="n_fft=.*")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*", module='sklearn.base')

# Load all the trained models
model_dt = joblib.load('Audio_decision_tree_model.joblib')
model_rf = joblib.load('Audio_random_forest_model.joblib')
model_xgb = joblib.load('Audio_xgboost_model.joblib')
model_svm = joblib.load('Audio_svm_model.joblib')
model_knn = joblib.load('Audio_knn_model.joblib')
model_nb = joblib.load('Audio_naive_bayes_model.joblib')
model_bagging = joblib.load('Audio_bagging_model.joblib')
model_adaboost = joblib.load('Audio_adaboost_model.joblib')
model_gradient_boosting = joblib.load('Audio_gradient_boosting_model.joblib')
ensemble_model_predict = joblib.load('Audio_ensemble_model.joblib')

# Function to preprocess new audio data
def preprocess_audio(audio_data, target_sr=22050):
    audio, sr = librosa.load(audio_data, sr=target_sr)  

    features = []
    fo = np.min(audio)
    fhi = np.max(audio)
    flo = np.min(np.abs(np.fft.fft(audio)))
    features.extend([fo, fhi, flo])
    
    jitter = librosa.effects.harmonic(audio)
    jitter_perc = np.mean(librosa.feature.rms(y=jitter))
    jitter_abs = np.mean(librosa.feature.rms(y=jitter, frame_length=1, hop_length=1))
    rap = np.mean(librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512))
    ddp = rap * 2
    ppq = np.mean(librosa.feature.tempogram(y=audio, sr=sr))
    features.extend([jitter_perc, jitter_abs, rap, ppq, ddp])
    
    shimmer = np.mean(np.abs(np.diff(audio))) / np.mean(np.abs(audio))
    shimmer_db = 10 * np.log10(shimmer)
    apq3 = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(audio)), sr=sr, p=3))
    apq5 = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(audio)), sr=sr, p=5))
    apq = np.mean(librosa.feature.spectral_flatness(y=audio))
    dda = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features.extend([shimmer, shimmer_db, apq3, apq5, apq, dda])
    
    nhr = np.mean(librosa.feature.zero_crossing_rate(audio)**2)
    hnr = np.mean(librosa.effects.harmonic(audio) / librosa.effects.percussive(audio))
    rpde = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=50.0))
    dfa = np.mean(librosa.feature.tonnetz(y=audio, sr=sr))
    features.extend([nhr, hnr, rpde, dfa])
    
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spread1 = np.mean(centroid)
    spread2 = np.mean(bandwidth)
    d2 = spread1 / spread2
    ppe = np.mean(librosa.feature.mfcc(y=audio, sr=sr))
    features.extend([spread1, spread2, d2,ppe])
   
    return np.array(features).reshape(1, -1)


# Function to predict using all models
def predict_all_models(audio_data):
    preprocessed_audio_data = preprocess_audio(audio_data)
    predictions = {}
    
    # Make predictions using each model
    predictions['Decision_Tree'] = model_dt.predict(preprocessed_audio_data)
    predictions['Random_Forest'] = model_rf.predict(preprocessed_audio_data)
    predictions['XGBoost'] = model_xgb.predict(preprocessed_audio_data)
    predictions['SVM'] = model_svm.predict(preprocessed_audio_data)
    predictions['KNN'] = model_knn.predict(preprocessed_audio_data)
    predictions['Naive_Bayes'] = model_nb.predict(preprocessed_audio_data)
    predictions['Bagging'] = model_bagging.predict(preprocessed_audio_data)
    predictions['AdaBoost'] = model_adaboost.predict(preprocessed_audio_data)
    predictions['Gradient_Boosting'] = model_gradient_boosting.predict(preprocessed_audio_data)
    
    # Count the number of votes for each class prediction
    vote_counts = {'Parkinson': 0, 'Healthy': 0}
    for model_name, prediction in predictions.items():
        if prediction == 1:
            vote_counts['Parkinson'] += 1
        else:
            vote_counts['Healthy'] += 1
    
    # Find the class with the maximum number of votes
    final_prediction = max(vote_counts, key=vote_counts.get)
    
    return predictions, final_prediction


# Sample audio data
# new_audio_data = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Parkinsons.wav"
new_audio_data = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Healthy.wav"

# Predict using all models
all_predictions, final_predict = predict_all_models(new_audio_data)

# Display predictions
for model_name, prediction in all_predictions.items():
    if model_name :
        print(f"{model_name} Prediction: {'Parkinson' if prediction == 1 else 'Healthy'}")
    
print(f"Final Prediction: {final_predict}")