import os
from collections import deque
import streamlit as st
from embedchain import App

os.environ["EMBEDSTORE_API_KEY"] = "V7x2LVMb4N"
from embedstore.rag.retrievers import EmbedStoreRetriever

# Initialize the retriever
arxiv_retriever = EmbedStoreRetriever(dataset_id = "arxiv_01", num_docs=3)

@st.cache(allow_output_mutation=True)
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

st.title("Arxiv Bot")

with st.sidebar.form(key='api_key_form'):
    api_key = st.text_input('Enter your OpenAI API key', value="", type="password")
    submitted = st.form_submit_button('Enter')
    if submitted:
        set_api_key(api_key)

if os.getenv("OPENAI_API_KEY"):
    arxiv_chat_bot = App()
    st.markdown("**Chat Interface**")
    user_input = st.text_input("Type your question here...")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque([], maxlen=5)

    if st.button("Send"):
        contexts = arxiv_retriever.query(user_input)
        contexts_text = [context['summary'] for context in contexts]
        extended_input = user_input + ' '.join(contexts_text)
        response = arxiv_chat_bot.chat(extended_input[:3000]) # Truncate input to respect model's token limit
        st.session_state.chat_history.appendleft((user_input, response))

    for i, (user_msg, bot_msg) in enumerate(reversed(st.session_state.chat_history)):
        col1, col2 = st.columns(2) # Changed from st.beta_columns(2)
        with col1:
            st.markdown(f"> **User**: {user_msg}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"> **Bot**: {bot_msg}", unsafe_allow_html=True)
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")
else:
    st.write("Please enter your OpenAI API key to continue.")
