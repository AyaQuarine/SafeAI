import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 20


def extract_features(audio_path):
    """
    Extract acoustic features from one audio recording.
    Returns a fixed-length NumPy array.
    """

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    # MFCC: 20 coefficients over time
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Energy
    rms = librosa.feature.rms(y=y)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    # Combine features
    features = []

    for feature in [mfcc, rms, zcr, centroid]:
        features.extend(np.mean(feature, axis=1))
        features.extend(np.std(feature, axis=1))

    return np.array(features, dtype=np.float32)
