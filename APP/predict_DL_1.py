import tensorflow as tf
import librosa
import numpy as np

def predict(audio_data):
    def load_model():
        model = tf.keras.models.load_model(r"/content/drive/MyDrive/projet_machine/ML/Deep_L/RNN/RNN_1.keras")
        return model

    SAMPLE_RATE = 22050
    TRACK_DURATION = 30  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    samples_per_segment = int(SAMPLES_PER_TRACK / 10)
    class_names = ['CHAABI', 'CHARKI', 'GNAWA', 'RAP', 'RAI', 'TAKTOKA', 'TACHLHIT']

    model = load_model()

    signal, sample_rate = librosa.load(audio_data, sr=SAMPLE_RATE)
    start = samples_per_segment * 4
    finish = start + samples_per_segment

    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mfcc)
    score = tf.nn.softmax(prediction[0])
    predicted_class = class_names[np.argmax(score)]
    confidence_percentage = 100 * np.max(score)

    return predicted_class, confidence_percentage


