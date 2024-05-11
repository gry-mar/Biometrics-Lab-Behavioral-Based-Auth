import streamlit as st
from st_audiorec import st_audiorec
import os


st.title('🎶AA🎶 - the best Audio Auth')

# Sound input section
st.subheader("Upload or Record Audio")
audio_file = st.file_uploader("Choose an audio file or record one:", type=['wav', 'mp3', 'ogg'])
# audio_data = st_audio_recorder(rec='Record new audio', play='Play', download='Download', save='Save', upload='Upload')

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

if audio_file is not None:
    # Save the file to the server temporarily
    with open(os.path.join("tempDir",audio_file.name),"wb") as f:
        f.write(audio_file.getbuffer())

    if st.button('Confirm Audio'):
        st.success('Audio confirmed!')

    st.audio(audio_file)

user_input = st.text_input("Enter some text")


if st.button('Click me!'):
    st.write(f"You entered: {user_input}")