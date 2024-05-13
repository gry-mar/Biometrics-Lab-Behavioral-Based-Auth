from scipy.spatial.distance import cosine
import pandas as pd 
from speechbrain.inference.speaker import EncoderClassifier
from scipy.io import wavfile
import noisereduce as nr
import json
import torch
import numpy as np
import os


PATH_TO_DB='../data/users_all.csv'

def identify(audio_path: str, threshold=0.6):
    """User identification based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        str| None: closest user or None if none of users is close enough
    """
    df = pd.read_csv(PATH_TO_DB)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))

    user_names = list(embedding_dict.keys())
    rate, data = wavfile.read(audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    emb = classifier.encode_batch(torch.tensor(reduced_noise).float()).reshape(-1, 1).view(-1)
    closest_user = None
    min_distance = -1
    
    for user_name, stored_embedding in embedding_dict.items():
        similarity = 1 - cosine(emb, stored_embedding)
        if similarity > min_distance:
            min_distance = similarity
            closest_user = user_name
    if min_distance >= threshold:
        return closest_user
    else:
        return None
    


def identify(audio_path: str, user_name:str, threshold=0.6):
    """User authentication based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        bool: True if auth correctly or false 
    """
    name = user_name.replace(" ", "_")
    df = pd.read_csv(PATH_TO_DB)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))

    user_names = list(embedding_dict.keys())
    rate, data = wavfile.read(audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    emb = classifier.encode_batch(torch.tensor(reduced_noise).float()).reshape(-1, 1).view(-1)
    closest_user = None
    min_distance = -1
    
    for user_name, stored_embedding in embedding_dict.items():
        similarity = 1 - cosine(emb, stored_embedding)
        if similarity > min_distance:
            min_distance = similarity
            closest_user = user_name
    if min_distance >= threshold and name == closest_user:
        return True
    else:
        return False
    


def load_and_encode(directory:str):
    embeddings = {}
    labels = []
    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person, "profile")
        embs_profile = []
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        print(person)
        for filename in os.listdir(person_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(person_dir, filename)
                rate, data = wavfile.read(filepath)
                reduced_noise = nr.reduce_noise(y=data, sr=rate)
                emb = classifier.encode_batch(torch.tensor(reduced_noise).float())
                embs_profile.append(emb.squeeze().detach().numpy()) 
                labels.append(person)
        embeddings[person]=np.mean(embs_profile, axis=0)
        data_for_df = [{'user_name': key, 'embedding': json.dumps(value.tolist())} for key, value in embeddings.items()]
        df = pd.DataFrame(data_for_df)
        df.to_csv(PATH_TO_DB, index=False)    
    return np.array(embeddings), labels