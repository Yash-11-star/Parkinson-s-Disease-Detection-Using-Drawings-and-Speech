import os
import librosa
import numpy as np
import pandas as pd


def extend_audio_duration(y, target_duration, sr):
    audio_duration = librosa.get_duration(y=y, sr=sr)  # Get duration of the loaded audio
    
    # If the audio duration is already longer than the target duration, return the original audio
    if audio_duration >= target_duration:
        return y
    
    # Calculate the number of repetitions needed to match or exceed the target duration
    repetitions = int(target_duration / audio_duration)
    remainder = target_duration % audio_duration
    
    # Concatenate the audio with itself to match or exceed the target duration
    extended_audio = y
    for _ in range(repetitions):
        extended_audio = np.concatenate((extended_audio, y), axis=0)
    
    # Append the remainder if needed
    if remainder > 0:
        extended_audio = np.concatenate((extended_audio, y[:int(remainder * sr)]), axis=0)
    
    return extended_audio

def extract_features(file_path, duration=5, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)  # Resample audio to target sample rate
    y = extend_audio_duration(y, duration, sr)  # Extend audio duration
    
    # Extract features
    features = []
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
    features.extend(mfccs.flatten())
    
    # Calculate Fhi, Flo
    fhi = np.max(y)
    flo = np.min(y)
    features.extend([fhi, flo])
    
    # Calculate Jitter_Perc, Jitter_Abs, RAP, DDP
    jitter = librosa.effects.harmonic(y)
    jitter_perc = np.mean(librosa.feature.rms(y=jitter))
    jitter_abs = np.mean(librosa.feature.rms(y=jitter, frame_length=1, hop_length=1))
    rap = np.mean(librosa.onset.onset_strength(y=y, sr=sr, hop_length=512))
    ddp = rap * 2
    features.extend([jitter_perc, jitter_abs, rap, ddp])
    
    # Calculate PPQ
    ppq = np.mean(librosa.feature.tempogram(y=y, sr=sr))
    features.append(ppq)
    
    # Calculate Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA
    shimmer = librosa.feature.spectral_centroid(y=y, sr=sr)
    shimmer_db = librosa.amplitude_to_db(shimmer)
    apq3 = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(y)), sr=sr))
    apq5 = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50.0))  # Adjust fmin to reduce the frequency range
    apq = np.mean(librosa.feature.spectral_flatness(y=y))
    dda = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features.extend([shimmer, shimmer_db, apq3, apq5, apq, dda])
    
    # Calculate NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE
    nhr = np.mean(librosa.feature.zero_crossing_rate(y))
    hnr = np.mean(librosa.feature.spectral_bandwidth(S=np.abs(librosa.stft(y)), sr=sr))
    rpde = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50.0))  # Adjust fmin to reduce the frequency range
    dfa = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
    
    # Check if poly_features returns enough elements
    poly_features = librosa.feature.poly_features(y=y, sr=sr)
    if poly_features.shape[0] >= 3:
        spread1 = np.mean(poly_features[0])
        spread2 = np.mean(poly_features[1])
        d2 = np.mean(poly_features[2])
        features.extend([spread1, spread2, d2])
    else:
        features.extend([0, 0, 0])  
    
    ppe = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    features.extend([nhr, hnr, rpde, dfa, ppe])
    
    return features


# Function to create a dataset from audio files with aggressive data augmentation
def create_augmented_dataset(data_folder, label):
    dataset = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_folder, file_name)
            features = extract_features(file_path)  # Extract features using the provided function
            dataset.append(features + [label])

            # Aggressive data augmentation
            y, sr = librosa.load(file_path, duration=3)
            for i in range(5):
                pitch_shifted_features = extract_features(file_path)  # Extract features from the pitch-shifted audio
                dataset.append(pitch_shifted_features + [label])
    
    return dataset

# Load Healthy and Parkinson's datasets with aggressive data augmentation
augmented_healthy_data = create_augmented_dataset("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/HC_AH", 0)
augmented_parkinsons_data = create_augmented_dataset("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/PD_AH", 1)

# Combine the datasets
combined_data = augmented_healthy_data + augmented_parkinsons_data

# Create a DataFrame
df_columns = ['MFCC' + str(i) for i in range(1, 3)] + ['Fhi', 'Flo'] + ['Jitter_Perc', 'Jitter_Abs', 'RAP', 'DDP', 'PPQ'] + ['Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA'] + ['NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE'] + ['Label']
df = pd.DataFrame(combined_data, columns=df_columns)

# Replace NaN values with zeros
df = df.fillna(0)

# Save the DataFrame to CSV
df.to_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/parkinsons.csv", index=False)

# Load the CSV data
data = pd.read_csv("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/parkinsons.csv")

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(data)
