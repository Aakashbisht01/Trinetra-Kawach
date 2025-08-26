import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import json


DATA_PATH = os.path.join('data', 'audio_samples')


MODEL_OUTPUT_DIR = 'models'
MODEL_FILENAME = 'keyword_detector.pkl'
LABELS_FILENAME = 'keyword_labels.json'


YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
YAMNET_SAMPLE_RATE = 16000



class AudioModelTrainer:
    

    def __init__(self):
        
        
        self.yamnet_model = self._load_yamnet_model()
        if self.yamnet_model is None:
            raise RuntimeError("Couldn't connect to YAMNet. Please check your internet connection.")
        

    def _load_yamnet_model(self):
        
        try:
            return hub.load(YAMNET_MODEL_URL)
        except Exception as e:
            print(f"Error loading YAMNet model: {e}")
            return None

    def _get_file_paths_and_labels(self):
        

        print(f"\nLet's go on a data hunt!  Looking for audio samples in '{DATA_PATH}'...")
        filepaths = []
        labels = []
        
        categories = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        if not categories:
            print(f"Error: I couldn't find any category folders in '{DATA_PATH}'.")
            print("Please create folders like 'help', 'noise', etc., and put your audio files inside.")
            return None, None, None

       
        label_map = {category: i for i, category in enumerate(categories)}
        
        for category in categories:
            category_path = os.path.join(DATA_PATH, category)
            for filename in os.listdir(category_path):
                if filename.endswith(('.wav', '.mp3')):
                    filepaths.append(os.path.join(category_path, filename))
                    labels.append(label_map[category])
        
        print(f" Found {len(filepaths)} audio files across {len(categories)} categories: {', '.join(categories)}")
        return filepaths, np.array(labels), categories

    def _extract_embeddings(self, filepaths):
       

        print("\nAsking YAMNet to listen to each file and write down its notes... ")
        all_embeddings = []
        
        for file_path in tqdm(filepaths, desc="Analyzing sounds"):
            try:
                
                wav_data, sr = librosa.load(file_path, sr=YAMNET_SAMPLE_RATE, mono=True)
                
                
                _, embeddings, _ = self.yamnet_model(wav_data)
                
                
                mean_embedding = np.mean(embeddings.numpy(), axis=0)
                all_embeddings.append(mean_embedding)
            except Exception as e:
                print(f"\nWarning: Had trouble reading '{os.path.basename(file_path)}'. Skipping. Error: {e}")
                all_embeddings.append(np.zeros(1024)) 

        return np.array(all_embeddings)

    def train_classifier(self, embeddings, labels):
        
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Using {len(X_train)} samples for training and {len(X_test)} for the final exam.")
        
        
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOur model scored {accuracy * 100:.2f}% on the test.")
        
        return classifier

    def save_model(self, model, labels):
        
        
        print("\nPacking the trained brain into a file... ")
        
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        
        model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
        labels_path = os.path.join(MODEL_OUTPUT_DIR, LABELS_FILENAME)
        
        joblib.dump(model, model_path)
        
        with open(labels_path, 'w') as f:
            json.dump(labels, f)
            
        print(f" Model saved to: '{model_path}'")
        print(f" Labels saved to: '{labels_path}'")
        

    def run_training_pipeline(self):
        
        
        
        filepaths, labels, categories = self._get_file_paths_and_labels()
        if filepaths is None:
            print("\n--- Mission Aborted: No data found. ---")
            return
            
        embeddings = self._extract_embeddings(filepaths)
        
        trained_model = self.train_classifier(embeddings, labels)
        
        self.save_model(trained_model, categories)
        
        


if __name__ == "__main__":
    
    
    trainer = AudioModelTrainer()
    trainer.run_training_pipeline()
