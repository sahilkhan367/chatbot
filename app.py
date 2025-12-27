import streamlit as st
from chatbot import qa, llm
import time
from object_detetction import detect_objects
import cv2
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title="Sahil Khan AI piepline test", page_icon="🤖")
st.title("🤖 Sahil Khan – AI Assistant and code updated from github")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history FIRST
for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.write(msg)



query = st.chat_input("Ask me anything about Sahil Khan...")

uploaded_image = st.file_uploader(
    "Upload an image (optional)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

#---------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------

auto_query = None

if uploaded_image:
    # ✅ Convert Streamlit file → OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Failed to decode image")
    else:
        objects = detect_objects(img)
        # st.success(f"Detected objects: {objects}")
        auto_query = "explain about" + ", ".join(objects)

final_query = query or auto_query

if uploaded_image and not query:
    mode = "image"
elif query:
    mode = "rag"


if final_query:
    # 1. Save user message
    st.session_state.history.append(("You", final_query))

    # 2. Show thinking placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 **Thinking...**")

        if mode == "image":
            response = llm.invoke(auto_query)
        else:
            response = qa.run(query)
        time.sleep(0.4)

        placeholder.markdown(response)

    # 3. Save assistant message ONLY ONCE
    st.session_state.history.append(("Sahil Khan", response))
