import os
import joblib
import librosa
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns 
from scipy.io import wavfile as wav
from imblearn.over_sampling import SMOTE
from IPython.display import Image
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,  recall_score

# EDA

healthy_dir = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/HC_AH"
parkinsons_dir = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/PD_AH"

def plot_waveform(audio_file, title):
    data, sample_rate = librosa.load(audio_file)
    plt.figure(figsize=(14, 5))
    plt.plot(data)
    plt.title(title + " Waveform")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
    

healthy_files = os.listdir(healthy_dir)
for file in healthy_files:
    audio_file = os.path.join(healthy_dir, file)
    plot_waveform(audio_file, "Healthy")


parkinsons_files = os.listdir(parkinsons_dir)
for file in parkinsons_files:
    audio_file = os.path.join(parkinsons_dir, file)
    plot_waveform(audio_file, "Parkinson's")


# Feature Extraction

def extract_features(file_path,target_sr=22050,n_fft=512):
    audio, sr = librosa.load(file_path, sr=target_sr)  

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
   
    
    return features

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

augmented_healthy_data = create_augmented_dataset(healthy_dir , 0)
augmented_parkinsons_data = create_augmented_dataset(parkinsons_dir , 1)

combined_data = augmented_healthy_data + augmented_parkinsons_data
num_features = 7  # Number of features returned by extract_features function
df_columns = ['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP','MDVP:Shimmer', 'MDVP:Shimmer_dB', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'] + ['Label']
df = pd.DataFrame(combined_data, columns=df_columns)

output_file_path = "/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/AudioDataset/Audio.csv"
df.to_csv(output_file_path, index=False)

###################################################

parkinsons = df.sample(frac=1, random_state=42).copy()

X = parkinsons.drop(["Label"], axis=1) 
y = parkinsons["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

parkinsons = X_train.copy()

parkinsons.iloc[0]

# Extracting features
features = [feature for feature in parkinsons.columns]

parkinsons.isna().sum()

# initialize and scale values
scaler = StandardScaler()
scaler.fit(parkinsons[features])
parkinsons[features] = scaler.transform(parkinsons[features])

joblib.dump(scaler, "scaler_joblib")

########################################################################

y_train.value_counts(normalize=True)

parkinsons = pd.concat([parkinsons, y_train], axis=1)

# Model

smote = SMOTE(random_state=51)
X = parkinsons.drop("Label", axis=1) 
y = parkinsons["Label"]
X_train, y_train = smote.fit_resample(X, y)

#  Ratio of No Diabetes to Diabetes
y_train.value_counts(normalize=True)

X_test[features] = scaler.transform(X_test[features]) # scaling features

# Model
model = RandomForestClassifier(random_state=51, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);

# Model
xgb = XGBClassifier(random_state=51)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);
# TN   FP
# FN*   TP - Recall

# Model
svm = SVC()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);
# TN   FP
# FN*   TP - Recall

################################

importance_df = pd.DataFrame({
    "Feature" : features,
    "Importance" : model.feature_importances_}).sort_values("Importance", ascending=False)

plt.figure(figsize=[10,6])
plt.title("Most Important Features")
sns.barplot(data=importance_df.head(10), y="Feature", x="Importance");

joblib.dump(model, "FinnalAudiomodel_joblib")

################################

# Decision  Tree Model 

df.describe().transpose()
fig, ax = plt.subplots(1,3,figsize=(16,10)) 
sns.boxplot(y='spread1',data=df, ax=ax[0],orient='v') 
sns.boxplot(y='spread2',data=df, ax=ax[1],orient='v')
sns.boxplot(y='PPE',data=df,ax=ax[2],orient='v')

fig, ax = plt.subplots(1,3,figsize=(16,8)) 
sns.distplot(df['MDVP:Flo'],ax=ax[0]) 
sns.distplot(df['MDVP:Fo'],ax=ax[1]) 
sns.distplot(df['MDVP:Fhi'],ax=ax[2])

fig, ax = plt.subplots(1,2,figsize=(16,8)) 
sns.distplot(df['NHR'],ax=ax[0]) 
sns.distplot(df['HNR'],ax=ax[1])

fig, ax = plt.subplots(2,3,figsize=(16,8)) 
sns.distplot(df['MDVP:Shimmer'],ax=ax[0,0]) 
sns.distplot(df['MDVP:Shimmer_dB'],ax=ax[0,1]) 
sns.distplot(df['Shimmer:APQ3'],ax=ax[0,2]) 
sns.distplot(df['Shimmer:APQ5'],ax=ax[1,0]) 
sns.distplot(df['MDVP:APQ'],ax=ax[1,1]) 
sns.distplot(df['Shimmer:DDA'],ax=ax[1,2])

sns.distplot( df[df.Label == 0]['spread1'], color = 'r')
sns.distplot( df[df.Label == 1]['spread1'], color = 'g')

fig, ax = plt.subplots(1,2,figsize=(16,8))
sns.boxplot(x='Label',y='NHR',data=df,ax=ax[0])
sns.boxplot(x='Label',y='HNR',data=df,ax=ax[1])

fig, ax = plt.subplots(1,2,figsize=(16,8))
sns.boxplot(x='Label',y='MDVP:Flo',data=df,palette="Set1",ax=ax[0])
sns.boxplot(x='Label',y='MDVP:Fo',data=df,palette="Set1",ax=ax[1])

# For categorical predictors
cols = ["MDVP:Jitter(%)","MDVP:Jitter(%)","MDVP:RAP","MDVP:PPQ","Jitter:DDP"]
fig, axs = plt.subplots(ncols = 5,figsize=(16,8))
fig.tight_layout()
for i in range(0,len(cols)):
    sns.boxplot(x='Label',y=cols[i],data=df, ax = axs[i])

corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.5})
plt.figure(figsize=(20,9))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

X = df.drop("Label",axis=1)
Y = df["Label"]

# Splitting Data into 70% Training data and 30% Testing Data:
X_train, X_test, y_train,  y_test = train_test_split(X, Y,train_size=0.7, test_size=0.3, random_state=42)
print(len(X_train)),print(len(X_test))

dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=100,min_samples_leaf=5)
dt_model.fit(X_train, y_train)

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=6,
            max_features=None, max_leaf_nodes=None,
            min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=100, splitter='best')

dt_model.score(X_test , y_test)
y_pred = dt_model.predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cbar=False);
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

train_char_label = ['No', 'Yes']
pd_tree_regularized = open('pd_tree_regularized.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file= pd_tree_regularized , feature_names = list(X_train), class_names = list(train_char_label))
pd_tree_regularized.close()
print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))

feature_names = df.columns[:-1].tolist()
target_names = [str(label) for label in df['Label'].unique().tolist()]

dot_data = export_graphviz(dt_model, out_file=None, 
                     feature_names=feature_names,  
                     class_names=target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  

# Render the DOT file to a PNG image
graph = graphviz.Source(dot_data)
graph.render("Parkinsons_decision_tree", format="png")
Image("/Users/yashtembhurnikar/Programming/Pccoe Final Year/Parkinson's Detection/Parkinsons_decision_tree.png")

# KNN Model 

k_model = KNeighborsClassifier(n_neighbors=5)
k_model.fit(X_train, y_train)
k_model.score(X_test,y_test)

y_pred = k_model.predict(X_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in KNN: {}'.format(count_misclassified))

# Random Forest Model

rfcl = RandomForestClassifier(n_estimators = 50)
rfcl = rfcl.fit(X_train, y_train)
y_pred = rfcl.predict(X_test)
rfcl.score(X_test , y_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Random Forest: {}'.format(count_misclassified))

feature_imp = pd.Series(rfcl.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Bagging Model

bgcl = BaggingClassifier(base_estimator=dt_model, n_estimators=50, max_samples=.7)
bgcl = bgcl.fit(X_train, y_train)
y_pred = bgcl.predict(X_test)
bgcl.score(X_test , y_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Bagging: {}'.format(count_misclassified))

# AdaBoosting Model

abcl = AdaBoostClassifier( n_estimators= 50)
abcl = abcl.fit(X_train,y_train)
y_pred = abcl.predict(X_test)
abcl.score(X_test , y_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Ada Boosting: {}'.format(count_misclassified))

# Gradient Boosting

gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.05)
gbcl = gbcl.fit(X_train,y_train)
y_pred = gbcl.predict(X_test)
gbcl.score(X_test , y_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Gradient Boosting: {}'.format(count_misclassified))

