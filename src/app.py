# import streamlit as st
from st_audiorec import st_audiorec
# import os
import streamlit as st

def save_file(file, name):
    if file is not None:
        with open(f'./{name}', "wb") as f:
            f.write(file.getbuffer())
        st.success('File saved')

option = st.radio("Choose your operation:", ('Add User', 'Authorization', 'Identification'))

if option == 'Add User':
    st.header("Add a New User")
    # File uploader allows multiple files
    uploaded_files = st.file_uploader("Choose audio files", accept_multiple_files=True, type=['wav', 'mp3'])
    user_name = st.text_input("Enter the user's name:")
    if st.button("Save User and Files"):
        for uploaded_file in uploaded_files:
            save_file(uploaded_file, f"{user_name}_{uploaded_file.name}")

elif option == 'Authorization':
    st.header("User Authorization")
    # File uploader for a single file
    uploaded_file = st.file_uploader("Upload your authorization audio file", accept_multiple_files=False, type=['wav', 'mp3'])
    who_are_you = st.text_input("Who are you?")
    
    record_option = st.radio("Or would you like to record audio?", ('No', 'Yes'))
    if record_option == 'Yes':
        uploaded_file = st_audiorec()

    if st.button("Authorize"):
        if uploaded_file is not None:
            save_file(uploaded_file, f"{who_are_you}_{uploaded_file.name}")
        st.write(f"Authorization attempt by {who_are_you}")

elif option == "Identification":
    st.header("User Identification")
    # File uploader for a single file
    uploaded_file = st.file_uploader("Upload your authorization audio file", accept_multiple_files=False, type=['wav', 'mp3'])
    
    record_option = st.radio("Or would you like to record audio?", ('No', 'Yes'))
    if record_option == 'Yes':
        uploaded_file = st_audiorec()

    if st.button("Identify"):
        st.write(f"Iedntified as ...")
