from scipy.spatial.distance import cosine
import pandas as pd 
from speechbrain.inference.speaker import EncoderClassifier
from scipy.io import wavfile
import noisereduce as nr
import json
import torch
import numpy as np
import os
import librosa
from io import BytesIO
import soundfile as sf


PATH_TO_DB='data/users_all.csv'

def identify(audio_path: str, threshold=0.6, path_to_db=PATH_TO_DB):
    """User identification based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        str| None: closest user or None if none of users is close enough
    """
    df = pd.read_csv(path_to_db)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))
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
    


def authorize(audio_path: str, user_name:str, threshold=0.6, path_to_db=PATH_TO_DB):
    """User authentication based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        bool: True if auth correctly or false 
    """
    name = user_name.replace(" ", "_")
    df = pd.read_csv(path_to_db)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))

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
    


def authorize_from_bytes(file: bytes, user_name:str, threshold=0.6, path_to_db=PATH_TO_DB):
    """User authentication based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        bool: True if auth correctly or false 
    """
    name = user_name.replace(" ", "_")
    df = pd.read_csv(path_to_db)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))

    
    data, rate = librosa.load(BytesIO(file), sr=None)
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
    

def identify_from_bytes(file: bytes, threshold=0.6, path_to_db=PATH_TO_DB):
    """User identification based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        str| None: closest user or None if none of users is close enough
    """
    df = pd.read_csv(path_to_db)
    embedding_dict = dict(zip(df['user_name'], df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))))
    data, rate = librosa.load(BytesIO(file), sr=None)
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
    


def load_and_encode(directory:str, path_to_db=PATH_TO_DB):
    """Loads users and audio from voc folder and saves users with embeddings to csv database

    Args:
        directory (str): location of wav files
        path_to_db (str, optional): Path that csv will be saved to. Defaults to PATH_TO_DB.

    """
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
        df.to_csv(path_to_db, index=False)    




def add_to_db(audio_files, user_name, path_to_db=PATH_TO_DB):
    """Adds user to csv db
    """
    user_name = user_name.replace(" ", "_")
    df = pd.read_csv(path_to_db)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embs_profile = []
    for file in audio_files:
        data, rate = librosa.load(BytesIO(file.read()), sr=None)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        emb = classifier.encode_batch(torch.tensor(reduced_noise).float()).reshape(-1, 1).view(-1)
        embs_profile.append(emb.squeeze().detach().numpy()) 
    new_row = pd.DataFrame([{'user_name': user_name, 'embedding':  json.dumps(np.mean(embs_profile, axis=0).tolist())}])
    print(new_row)
    df = pd.concat([df, new_row])
    df.to_csv(path_to_db, index=False)


