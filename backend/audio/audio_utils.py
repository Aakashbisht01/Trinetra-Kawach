import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import os

# This global variable will store the loaded model to avoid reloading it every time.
YAMNET_MODEL = None

def _load_yamnet_model():
    """
    Loads the YAMNet model from TensorFlow Hub. This is done only once.
    """
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        try:
            print("Loading YAMNet model from TensorFlow Hub...")
            YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
            print("YAMNet model loaded successfully.")
        except Exception as e:
            print(f"Error loading YAMNet model: {e}")
            print("Please ensure you have an internet connection.")
            YAMNET_MODEL = None
    return YAMNET_MODEL

def extract_embedding(audio_file: str) -> np.ndarray:
    """
    Extracts a single, averaged 1024-dimensional embedding from an audio file.
    """
    yamnet_model = _load_yamnet_model()
    if yamnet_model is None:
        return None

    try:
        # Load the audio file, ensuring it's mono and at a 16kHz sample rate.
        wav_data, _ = librosa.load(audio_file, sr=16000, mono=True)
        waveform = tf.constant(wav_data, dtype=tf.float32)

        # Get the scores and embeddings from the model.
        scores, embeddings, _ = yamnet_model(waveform)
        
        # Average the embeddings over the time axis to get a single feature vector.
        averaged_embedding = np.mean(embeddings.numpy(), axis=0)
        
        return averaged_embedding
        
    except FileNotFoundError:
        print(f"Error: The file '{audio_file}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing the audio file '{audio_file}': {e}")
        return None
        
def extract_raw_embeddings(audio_file: str) -> np.ndarray:
    """
    Extracts the full sequence of 1024-dimensional embeddings from an audio file.
    """
    yamnet_model = _load_yamnet_model()
    if yamnet_model is None:
        return None
        
    try:
        wav_data, _ = librosa.load(audio_file, sr=16000, mono=True)
        waveform = tf.constant(wav_data, dtype=tf.float32)
        
        # Get the full sequence of embeddings from the YAMNet model.
        scores, embeddings, _ = yamnet_model(waveform)
        
        return embeddings.numpy()
        
    except FileNotFoundError:
        print(f"Error: The file '{audio_file}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing the audio file '{audio_file}': {e}")
        return None
        
# ======================================================================
# Main Block for Quick Testing
# This code will only run if the script is executed directly.
# ======================================================================

if __name__ == "__main__":
    # Example usage:
    # IMPORTANT: Replace 'test_audio.wav' with the path to a real audio file.
    # You need to create or provide this file for the test to work.
    TEST_FILE = "path/to/your/test_audio.wav"

    # Create a dummy audio file for testing if it doesn't exist.
    if not os.path.exists(TEST_FILE):
        print(f"'{TEST_FILE}' not found. Creating a dummy silent audio file for testing.")
        # Create a silent 2-second audio clip
        sr = 16000
        duration = 2
        dummy_audio = np.zeros(sr * duration)
        import soundfile as sf
        os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
        sf.write(TEST_FILE, dummy_audio, sr)
        print("Dummy file created. You may want to replace it with a real audio file.")

    print(f"\n--- Testing feature extraction with '{TEST_FILE}' ---")
    
    averaged_emb = extract_embedding(TEST_FILE)
    if averaged_emb is not None:
        print(f"Shape of averaged embedding: {averaged_emb.shape}")
        
    print("\n----------------------------------------------------")
    
    raw_emb = extract_raw_embeddings(TEST_FILE)
    if raw_emb is not None:
        print(f"Shape of raw embeddings: {raw_emb.shape}")
