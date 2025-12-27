import streamlit as st
from chatbot import qa

st.set_page_config(page_title="Sahil Khan AI", page_icon="🤖")
st.title("🤖 Sahil Khan – AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask me anything about Sahil Khan...")

if query:
    response = qa.run(query)
    st.session_state.history.append(("You", query))
    st.session_state.history.append(("Sahil Khan", response))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)