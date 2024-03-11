import os
import joblib
import librosa
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier
import graphviz

# EDA and Feature Extraction

# healthy_dir = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/HC_AH"
# parkinsons_dir = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/PD_AH"

# def plot_waveform(audio_file, title):
#     data, sample_rate = librosa.load(audio_file)
#     plt.figure(figsize=(14, 5))
#     plt.plot(data)
#     plt.title(title + " Waveform")
#     plt.xlabel('Sample')
#     plt.ylabel('Amplitude')
#     plt.show()

# healthy_files = os.listdir(healthy_dir)
# for file in healthy_files:
#     audio_file = os.path.join(healthy_dir, file)
#     plot_waveform(audio_file, "Healthy")

# parkinsons_files = os.listdir(parkinsons_dir)
# for file in parkinsons_files:
#     audio_file = os.path.join(parkinsons_dir, file)
#     plot_waveform(audio_file, "Parkinson's")


# def extract_features(file_path,target_sr=22050,n_fft=512):
#     audio, sr = librosa.load(file_path, sr=target_sr)  

#     features = []
#     fo = np.min(audio)
#     fhi = np.max(audio)
#     flo = np.min(np.abs(np.fft.fft(audio)))
#     features.extend([fo, fhi, flo])
    
#     jitter = librosa.effects.harmonic(audio)
#     jitter_perc = np.mean(librosa.feature.rms(y=jitter))
#     jitter_abs = np.mean(librosa.feature.rms(y=jitter, frame_length=1, hop_length=1))
#     rap = np.mean(librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512))
#     ddp = rap * 2
#     ppq = np.mean(librosa.feature.tempogram(y=audio, sr=sr))
#     features.extend([jitter_perc, jitter_abs, rap, ppq, ddp])
    
#     shimmer = np.mean(np.abs(np.diff(audio))) / np.mean(np.abs(audio))
#     shimmer_db = 10 * np.log10(shimmer)
#     apq3 = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(audio)), sr=sr, p=3))
#     apq5 = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(audio)), sr=sr, p=5))
#     apq = np.mean(librosa.feature.spectral_flatness(y=audio))
#     dda = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
#     features.extend([shimmer, shimmer_db, apq3, apq5, apq, dda])
    
#     nhr = np.mean(librosa.feature.zero_crossing_rate(audio)**2)
#     hnr = np.mean(librosa.effects.harmonic(audio) / librosa.effects.percussive(audio))
#     rpde = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=50.0))
#     dfa = np.mean(librosa.feature.tonnetz(y=audio, sr=sr))
#     features.extend([nhr, hnr, rpde, dfa])
    
#     centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
#     bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
#     spread1 = np.mean(centroid)
#     spread2 = np.mean(bandwidth)
#     d2 = spread1 / spread2
#     ppe = np.mean(librosa.feature.mfcc(y=audio, sr=sr))
#     features.extend([spread1, spread2, d2,ppe])
   
    
#     return features

# def create_augmented_dataset(data_folder, label):
#     dataset = []
#     for file_name in os.listdir(data_folder):
#         if file_name.endswith('.wav'):
#             file_path = os.path.join(data_folder, file_name)
#             features = extract_features(file_path)  # Extract features using the provided function
#             dataset.append(features + [label])

#             # Aggressive data augmentation
#             y, sr = librosa.load(file_path, duration=3)
#             for i in range(5):
#                 pitch_shifted_features = extract_features(file_path)  # Extract features from the pitch-shifted audio
#                 dataset.append(pitch_shifted_features + [label])
    
#     return dataset

# # Create augmented datasets
# augmented_healthy_data = create_augmented_dataset(healthy_dir , 0)
# augmented_parkinsons_data = create_augmented_dataset(parkinsons_dir , 1)

# combined_data = augmented_healthy_data + augmented_parkinsons_data
# num_features = 7  # Number of features returned by extract_features function
# df_columns = ['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP','MDVP:Shimmer', 'MDVP:Shimmer_dB', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'] + ['Label']
# df = pd.DataFrame(combined_data, columns=df_columns)

# output_file_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Audio.csv"
# df.to_csv(output_file_path, index=False)

output_file_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Audio.csv"
df= pd.read_csv(output_file_path)

# Model Training
parkinsons = df.sample(frac=1, random_state=42).copy()


X = parkinsons.drop(["Label"], axis=1) 
y = parkinsons["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

parkinsons = X_train.copy()


scaler = StandardScaler()
scaler.fit(parkinsons)
parkinsons = scaler.transform(parkinsons)

joblib.dump(scaler, "Audio_scaler_joblib")

y_train.value_counts(normalize=True)

parkinsons = df

smote = SMOTE(random_state=51)
X = parkinsons.drop("Label", axis=1) 
y = parkinsons["Label"]
X_train, y_train = smote.fit_resample(X, y)

y_train.value_counts(normalize=True)

X_test = scaler.transform(X_test)

# Model Training
# Random Forest
model_rf = RandomForestClassifier(random_state=51, n_jobs=-1)
model_rf.fit(X_train, y_train)
joblib.dump(model_rf, "Audio_random_forest_model.joblib")

# XGBoost
model_xgb = XGBClassifier(random_state=51)
model_xgb.fit(X_train, y_train)
joblib.dump(model_xgb, "Audio_xgboost_model.joblib")

# SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)
joblib.dump(model_svm, "Audio_svm_model.joblib")

# KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
joblib.dump(model_knn, "Audio_knn_model.joblib")

# Naive Bayes
model_nb = gnb()
model_nb.fit(X_train, y_train)
joblib.dump(model_nb, "Audio_naive_bayes_model.joblib")

# Bagging
model_bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50)
model_bagging.fit(X_train, y_train)
joblib.dump(model_bagging, "Audio_bagging_model.joblib")

# AdaBoost
model_adaboost = AdaBoostClassifier(n_estimators=50)
model_adaboost.fit(X_train, y_train)
joblib.dump(model_adaboost, "Audio_adaboost_model.joblib")

# Gradient Boosting
model_gradient_boosting = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05)
model_gradient_boosting.fit(X_train, y_train)
joblib.dump(model_gradient_boosting, "Audio_gradient_boosting_model.joblib")

# Decision Tree 
model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=100, min_samples_leaf=5)
model_dt.fit(X_train, y_train)
joblib.dump(model_dt, "Audio_decision_tree_model.joblib")

# Model Evaluation

# Decision Tree
predictions_dt = model_dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, predictions_dt) * 100:.2f}%")
print(f"Decision Tree F1 Score: {f1_score(y_test, predictions_dt) * 100:.2f}%")
print(f"Decision Tree Recall: {recall_score(y_test, predictions_dt) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_dt), annot=True, cbar=False)

# Random Forest
predictions_rf = model_rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, predictions_rf) * 100:.2f}%")
print(f"Random Forest F1 Score: {f1_score(y_test, predictions_rf) * 100:.2f}%")
print(f"Random Forest Recall: {recall_score(y_test, predictions_rf) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_rf), annot=True, cbar=False)

# XGBoost
predictions_xgb = model_xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, predictions_xgb) * 100:.2f}%")
print(f"XGBoost F1 Score: {f1_score(y_test, predictions_xgb) * 100:.2f}%")
print(f"XGBoost Recall: {recall_score(y_test, predictions_xgb) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_xgb), annot=True, cbar=False)

# SVM
predictions_svm = model_svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, predictions_svm) * 100:.2f}%")
print(f"SVM F1 Score: {f1_score(y_test, predictions_svm) * 100:.2f}%")
print(f"SVM Recall: {recall_score(y_test, predictions_svm) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_svm), annot=True, cbar=False)

# KNN
predictions_knn = model_knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, predictions_knn) * 100:.2f}%")
print(f"KNN F1 Score: {f1_score(y_test, predictions_knn) * 100:.2f}%")
print(f"KNN Recall: {recall_score(y_test, predictions_knn) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_knn), annot=True, cbar=False)

# Naive Bayes
predictions_nb = model_nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, predictions_nb) * 100:.2f}%")
print(f"Naive Bayes F1 Score: {f1_score(y_test, predictions_nb) * 100:.2f}%")
print(f"Naive Bayes Recall: {recall_score(y_test, predictions_nb) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_nb), annot=True, cbar=False)

# Bagging
predictions_bagging = model_bagging.predict(X_test)
print(f"Bagging Accuracy: {accuracy_score(y_test, predictions_bagging) * 100:.2f}%")
print(f"Bagging F1 Score: {f1_score(y_test, predictions_bagging) * 100:.2f}%")
print(f"Bagging Recall: {recall_score(y_test, predictions_bagging) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_bagging), annot=True, cbar=False)

# AdaBoost
predictions_adaboost = model_adaboost.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, predictions_adaboost) * 100:.2f}%")
print(f"AdaBoost F1 Score: {f1_score(y_test, predictions_adaboost) * 100:.2f}%")
print(f"AdaBoost Recall: {recall_score(y_test, predictions_adaboost) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_adaboost), annot=True, cbar=False)

# Gradient Boosting
predictions_gradient_boosting = model_gradient_boosting.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, predictions_gradient_boosting) * 100:.2f}%")
print(f"Gradient Boosting F1 Score: {f1_score(y_test, predictions_gradient_boosting) * 100:.2f}%")
print(f"Gradient Boosting Recall: {recall_score(y_test, predictions_gradient_boosting) * 100:.2f}%")
sns.heatmap(confusion_matrix(y_test, predictions_gradient_boosting), annot=True, cbar=False)

# Ensemble Model
def ensemble_model_predict(X):
    pred_rf = model_rf.predict(X)
    pred_xgb = model_xgb.predict(X)
    pred_svm = model_svm.predict(X)
    pred_knn = model_knn.predict(X)
    pred_nb = model_nb.predict(X)
    pred_bagging = model_bagging.predict(X)
    pred_adaboost = model_adaboost.predict(X)
    pred_gradient_boosting = model_gradient_boosting.predict(X)
    pred_dt = model_dt.predict(X)
    
    ensemble_pred = np.round((pred_rf + pred_xgb + pred_svm + pred_knn + pred_nb + pred_bagging + pred_adaboost + pred_gradient_boosting + pred_dt) / 9)
    print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred) * 100:.2f}%")
    print(f"Ensembel Boosting F1 Score: {f1_score(y_test, ensemble_pred) * 100:.2f}%")
    print(f"Ensemble Recall: {recall_score(y_test, ensemble_pred) * 100:.2f}%")
    sns.heatmap(confusion_matrix(y_test, ensemble_pred), annot=True, cbar=False)
    return ensemble_pred

joblib.dump(ensemble_model_predict, "Audio_ensemble_model.joblib")
