import streamlit as st
from chatbot import qa
import time

st.set_page_config(page_title="Sahil Khan AI", page_icon="🤖")
st.title("🤖 Sahil Khan – AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history FIRST
for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.write(msg)

query = st.chat_input("Ask me anything about Sahil Khan...")

if query:
    # 1. Save user message
    st.session_state.history.append(("You", query))

    # 2. Show thinking placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 **Thinking...**")

        response = qa.run(query)
        time.sleep(0.4)

        placeholder.markdown(response)

    # 3. Save assistant message ONLY ONCE
    st.session_state.history.append(("Sahil Khan", response))
