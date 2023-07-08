import os
from collections import deque
import streamlit as st
from embedchain import App

os.environ["EMBEDSTORE_API_KEY"] = 'V7x2LVMb4N'
from embedstore.rag.retrievers import EmbedStoreRetriever

# Initialize the retriever
arxiv_retriever = EmbedStoreRetriever(dataset_id = "arxiv_01", num_docs=3)

# Set the OpenAI API key
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
        contexts = arxiv_retriever.query(user_input)
        # Extract the summary text from the contexts
        contexts_text = [context['summary'] for context in contexts]
        extended_input = user_input + '  ' + '  '.join(contexts_text)
        response = arxiv_chat_bot.chat(extended_input)
        st.session_state.chat_history.appendleft((user_input, response))

    # Display the chat history
    for user_msg, bot_msg in reversed(st.session_state.chat_history):
        st.markdown(f"> **User**: {user_msg}\n> **Bot**: {bot_msg}")
else:
    st.write("Please enter your OpenAI API key to continue.")
