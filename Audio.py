import os
import librosa
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Function to extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Specify the path to your dataset

path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioData.csv"
df = pd.read_csv(path)
# Extract features and labels
X = df.drop(["name", "status"], axis=1) 
y = df["status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
