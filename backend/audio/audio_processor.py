# audio_processor.py

import logging
import numpy as np
import sounddevice as sd
import time
import os
import joblib
import json
import soundfile as sf
from datetime import datetime

# --- Import your actual project modules ---
from audio.audio_utils import extract_embedding

# Configuration
SAMPLE_RATE = 16000  # Audio sampling rate in Hz, must match YAMNet
CHUNK_DURATION = 1   # Duration of each audio chunk in seconds
BUFFER_DURATION = 3  # Duration of the audio buffer in seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Thresholds
ALERT_THRESHOLD = 0.75
NOISE_THRESHOLD = 0.40
SILENCE_THRESHOLD = 0.05

# File paths for the trained model and labels
MODEL_OUTPUT_DIR = 'models'
MODEL_FILENAME = 'keyword_detector.pkl'
LABELS_FILENAME = 'keyword_labels.json'

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model():
    """
    Loads the trained classifier and its corresponding labels from disk.
    """
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    labels_path = os.path.join(MODEL_OUTPUT_DIR, LABELS_FILENAME)

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        logging.error("Trained model or labels file not found. Please train the model first.")
        return None, None
    
    try:
        model = joblib.load(model_path)
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        logging.info("Audio model and labels loaded successfully.")
        return model, labels
    except Exception as e:
        logging.error(f"Error loading audio model files: {e}")
        return None, None

def process_chunk(audio_data, model, labels, callback=None):
    """
    Processes a single chunk of audio data and sends results via a callback.
    """
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms < SILENCE_THRESHOLD:
        logging.info("Prediction: 'silence'")
        if callback:
            callback("silence", rms)
        return

    temp_dir = 'temp_audio'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'temp_chunk.wav')

    try:
        sf.write(temp_file, audio_data, SAMPLE_RATE)
        embedding = extract_embedding(temp_file)

        if embedding is None:
            logging.error("Embedding extraction failed.")
            return

        embedding = np.expand_dims(embedding, axis=0)
        probabilities = model.predict_proba(embedding)[0]
        
        most_confident_idx = np.argmax(probabilities)
        most_confident_label = labels[most_confident_idx]
        most_confident_score = probabilities[most_confident_idx]

        # --- Use the callback to send data back to main.py ---
        if callback:
            callback(most_confident_label, most_confident_score)

        # Logging logic remains the same
        if most_confident_score > ALERT_THRESHOLD and most_confident_label != "noise":
            logging.warning(f"ALERT: Suspicious sound! Class: '{most_confident_label}', Confidence: {most_confident_score:.2f}")
        elif most_confident_score < NOISE_THRESHOLD:
            logging.info(f"Prediction: 'background noise', Confidence: {most_confident_score:.2f}")
        else:
            logging.info(f"Prediction: '{most_confident_label}', Confidence: {most_confident_score:.2f}")

    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main(callback=None):
    """
    Main orchestration function for real-time audio processing.
    Accepts a callback function to send results to.
    """
    logging.info("Starting audio processor...")

    classifier, labels = load_trained_model()
    if classifier is None:
        return

    audio_buffer = np.zeros(BUFFER_SIZE, dtype='float32')

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE
        ) as stream:
            logging.info("Audio stream opened. Listening for suspicious sounds...")
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    logging.warning("Audio buffer overflowed!")
                
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                audio_buffer[-CHUNK_SIZE:] = audio_chunk.flatten()

                # Pass the callback to the processing function
                process_chunk(audio_buffer, classifier, labels, callback)
                time.sleep(CHUNK_DURATION)

    except sd.PortAudioError as e:
        logging.critical(f"PortAudio error: {e}. Please ensure a microphone is connected.")
    except KeyboardInterrupt:
        logging.info("Exiting audio processor.")
    except Exception as e:
        logging.critical(f"A fatal error occurred in audio processor: {e}")

if __name__ == "__main__":
    main()
