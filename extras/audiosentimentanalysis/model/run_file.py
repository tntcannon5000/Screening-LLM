import librosa
import os, pickle
import numpy as np
import tensorflow as tf

# Get audios
test_file_path = 'input/chunk-02.wav'
emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

# declaration of variables
X,sr = librosa.load(test_file_path, sr = None)
stft = np.abs(librosa.stft(X))
f2 = open('model/feature.pkl','rb')
feature_all = pickle.load(f2)
f3 = open('model/label.pkl','rb')
labels = pickle.load(f3)



# load model
with open('model/mlp_model_relu_adadelta.json', 'r') as json_file:   
    json_config = json_file.read()        
model = tf.keras.models.model_from_json(json_config)  
model.load_weights('model/mlp_relu_adadelta_model.weights.h5')
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T,axis=0)
#mel = np.mean(librosa.feature.melspectrogram(S=stft, sr=sr).T,axis=0)

contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

feature_all = np.vstack([feature_all,features])

x_chunk = np.array(features)

x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])

# Predict the features sentiment
y_chunk_model1 = model.predict(x_chunk)
index = np.argmax(y_chunk_model1)
print('Emotion:',emotions[index])