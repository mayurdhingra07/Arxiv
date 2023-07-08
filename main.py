import os
from collections import deque
import streamlit as st
from embedchain import App

# Set the OpenAI API key
@st.cache(allow_output_mutation=True)
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

# Set the title in the middle of the page
st.title("Arxiv Bot")

# Sidebar form for API key
with st.sidebar.form(key='api_key_form'):
    api_key = st.text_input('Enter your OpenAI API key', value="", type="password")
    submitted = st.form_submit_button('Enter')
    if submitted:
        set_api_key(api_key)

# If the API key is entered, display the chat interface
if os.getenv("OPENAI_API_KEY"):
    # Create the Arxiv chatbot
    arxiv_chat_bot = App()

    # Create the chat interface
    st.markdown("**Chat Interface**")
    user_input = st.text_input("Type your question here...")

    # Get or initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque([], maxlen=5)

    if st.button("Send"):
        response = arxiv_chat_bot.chat(user_input)
        st.session_state.chat_history.appendleft((user_input, response))

    # Display the chat history
    for user_msg, bot_msg in reversed(st.session_state.chat_history):
        st.markdown(f"> **User**: {user_msg}\n> **Bot**: {bot_msg}")
else:
    st.write("Please enter your OpenAI API key to continue.")
