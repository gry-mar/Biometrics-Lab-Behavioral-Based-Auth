import streamlit as st
from st_audiorec import st_audiorec
import os


st.title('🎶AA🎶 - the best Audio Auth')


st.subheader("Record file or choose from disk")

if "audio_option" not in st.session_state:
    st.session_state.audio_option = "Load"
    
st.radio(
        label="Choose",
        options=["Record", "Load"],
        key="audio_option",
    )

if st.session_state.audio_option == "Record":
    audio_file = st_audiorec()
else: 
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
st.subheader("Select auth mode")
# if wav_audio_data is not None:
#         st.audio(wav_audio_data, format='audio/wav')

tab1, tab2= st.tabs(["Check identity", "Authorize as an user"])

with tab1:
    
    if audio_file is not None:
        # Save the file to the server temporarily
        with open(os.path.join("tempDir",audio_file.name),"wb") as f:
            f.write(audio_file.getbuffer())

        if st.button('Confirm Audio'):
            st.success('Audio confirmed!')

        # st.audio(audio_file)

    if st.button('Check'):
        pass

with tab2:

    # wav_audio_data = st_audiorec()

    # if wav_audio_data is not None:
    #     st.audio(wav_audio_data, format='audio/wav')

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