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


PATH_TO_DB = "data/users_all.csv"
df = pd.read_csv(PATH_TO_DB)
embedding_dict = dict(
    zip(
        df["user_name"],
        df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32)),
    )
)
df["embedding"] = df["embedding"].apply(
    lambda x: np.array(json.loads(x), dtype=np.float32)
)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"}
)
wave_files_cache = {}


def identify(
    audio_path: str,
    threshold=0.6,
    path_to_db=PATH_TO_DB,
    denoise=True,
    transform_fn=None,
):
    """User identification based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        str| None: closest user or None if none of users is close enough
    """
    global df
    global classifier
    global wave_files_cache

    if audio_path in wave_files_cache:
        rate, data = wave_files_cache[audio_path]
    else:
        rate, data = wavfile.read(audio_path)
        wave_files_cache[audio_path] = (rate, data)

    if transform_fn:
        data, rate = transform_fn(data, rate)

    if denoise:
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
    else:
        reduced_noise = data
    emb = (
        (
            classifier.encode_batch(
                torch.tensor(reduced_noise, device=torch.device("cuda"))
            )
            .reshape(-1, 1)
            .view(-1)
        )
        .cpu()
        .numpy()
    )

    df["cosine_distance"] = df["embedding"].apply(lambda x: cosine(emb, x))
    df = df.sort_values(by="cosine_distance", ascending=True)
    df_closest_user = df.iloc[0]["user_name"]
    df_min_distance = df.iloc[0]["cosine_distance"]
    if df_min_distance <= threshold:
        return df_closest_user
    else:
        return None


def authorize(
    audio_path: str,
    user_name: str,
    threshold=0.6,
    path_to_db=PATH_TO_DB,
    denoise=True,
    transform_fn=None,
):
    """User authentication based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        bool: True if auth correctly or false
    """
    global df
    global classifier
    global wave_files_cache
    if audio_path in wave_files_cache:
        rate, data = wave_files_cache[audio_path]
    else:
        rate, data = wavfile.read(audio_path)
        wave_files_cache[audio_path] = (rate, data)

    if transform_fn:
        data, rate = transform_fn(data, rate)

    if denoise:
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
    else:
        reduced_noise = data
    emb = (
        (
            classifier.encode_batch(
                torch.tensor(reduced_noise, device=torch.device("cuda"))
            )
            .reshape(-1, 1)
            .view(-1)
        )
        .cpu()
        .numpy()
    )

    cosine_distance = cosine(
        emb, df.loc[df["user_name"] == user_name, "embedding"].values[0]
    )
    if cosine_distance <= threshold:
        return True
    else:
        return False


def authorize_from_bytes(
    file: bytes, user_name: str, threshold=0.6, path_to_db=PATH_TO_DB
):
    """User authentication based on cosine similarity.

    Args:
        audio_path (str): string path to the audio file
        threshold (float, optional): Threshold thad defines the minimum cosine similarity value that should be reached to identify user. Defaults to 0.6.

    Returns:
        bool: True if auth correctly or false
    """
    name = user_name.replace(" ", "_")
    df = pd.read_csv(path_to_db)
    embedding_dict = dict(
        zip(
            df["user_name"],
            df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32)),
        )
    )

    data, rate = librosa.load(BytesIO(file), sr=None)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    emb = (
        classifier.encode_batch(torch.tensor(reduced_noise).float())
        .reshape(-1, 1)
        .view(-1)
    )
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
    embedding_dict = dict(
        zip(
            df["user_name"],
            df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32)),
        )
    )
    data, rate = librosa.load(BytesIO(file), sr=None)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    emb = (
        classifier.encode_batch(torch.tensor(reduced_noise).float())
        .reshape(-1, 1)
        .view(-1)
    )
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


def load_and_encode(directory: str, path_to_db=PATH_TO_DB):
    embeddings = {}
    labels = []
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person, "profile")
        embs_profile = []

        print(person)
        for filename in os.listdir(person_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(person_dir, filename)
                rate, data = wavfile.read(filepath)
                reduced_noise = nr.reduce_noise(y=data, sr=rate)
                emb = classifier.encode_batch(torch.tensor(reduced_noise).float())
                embs_profile.append(emb.squeeze().detach().numpy())
                labels.append(person)
        embeddings[person] = np.mean(embs_profile, axis=0)
        data_for_df = [
            {"user_name": key, "embedding": json.dumps(value.tolist())}
            for key, value in embeddings.items()
        ]
        df = pd.DataFrame(data_for_df)
        df.to_csv(path_to_db, index=False)
    return np.array(embeddings), labels


def save_as_wav(audio_files, user_name):
    """
    Saves a list of audio files as .wav in a specified directory structure.

    Parameters:
        audio_files (list of tuples): List of tuples, each containing (audio_data, sample_rate)
        user_name (str): The user's name to customize the folder path
    """
    user_name = user_name.replace(" ", "_")
    base_dir = f"data/vox1/{user_name}/profile"

    os.makedirs(base_dir, exist_ok=True)

    # Loop through the list of audio files and their sample rates
    for i, (audio_data, sample_rate) in enumerate(audio_files):
        # Define the file path
        file_path = os.path.join(base_dir, f"{user_name}_{i}.wav")

        # Save the audio file as a .wav file
        sf.write(file_path, audio_data, sample_rate)
        print(f"Saved {file_path}")
