
from st_audiorec import st_audiorec
import streamlit as st
from auth.utils import *


option = st.radio("Choose your operation:", ('Add User', 'Authorization', 'Identification'))

if option == 'Add User':
    st.header("Add a New User")
    # File uploader allows multiple files
    uploaded_files = st.file_uploader("Choose audio files", accept_multiple_files=True, type=['wav', 'mp3'])
    print(uploaded_files.count)
    user_name = st.text_input("Enter the user's name:")
    if st.button("Save User and Files"):
        add_to_db(uploaded_files, user_name)
        st.success("User added")


elif option == 'Authorization':
    st.header("User Authorization")
    # File uploader for a single file
    uploaded_file = st.file_uploader("Upload your authorization audio file", accept_multiple_files=False, type=['wav', 'mp3'])
    user_name = st.text_input("Who are you?")
    
    record_option = st.radio("Or would you like to record audio?", ('No', 'Yes'))
    if record_option == 'Yes':
        uploaded_file = st_audiorec()

    if st.button("Authorize"):
        if uploaded_file is not None:
            if record_option == 'No':
                uploaded_file = uploaded_file.read()
            ret = authorize_from_bytes(uploaded_file, user_name)
        st.write(f"Authorization attempt by {user_name} with result {ret}")

elif option == "Identification":
    st.header("User Identification")
    # File uploader for a single file
    uploaded_file = st.file_uploader("Upload your authorization audio file", accept_multiple_files=False, type=['wav', 'mp3'])
    
    record_option = st.radio("Or would you like to record audio?", ('No', 'Yes'))
    if record_option == 'Yes':
        uploaded_file = st_audiorec()

    if st.button("Identify"):
        if uploaded_file is not None:
            if record_option == 'No':
                uploaded_file = uploaded_file.read()
            out_person = identify_from_bytes(uploaded_file)
            if out_person == None:
                st.write("Unable to identify, such person not in database")
            else:
                st.write(f"Identified as {out_person}")
