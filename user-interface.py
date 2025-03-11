import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER'] = 'false'

import streamlit as st
from datetime import datetime
from chat import ask_rag

# Title
st.title("💬 Capstone RAG")

# Initialize session state for storing messages
if "messages" not in st.session_state:
  st.session_state.messages = []

# Create a scrollable container for messages
messages_container = st.container(height=400)  # Adjust height as needed
with messages_container:
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.caption(message["timestamp"])
      st.markdown(f"{message['content']}")
      if "related_files" in message and len(message["related_files"]) > 0:
        st.write("Related files:")
        for file in message["related_files"]:
          st.markdown(f"- [{file}]({file})", unsafe_allow_html=True)

  # Add this to auto-scroll to bottom
  st.markdown(
    """
    <script>
    window.onload = function() {
        var container = document.querySelector('[data-testid="stVerticalBlock"]');
        container.scrollTop = container.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
  )

# Function to handle message sending
def send_message():
  if st.session_state.user_input:
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add user message to chat with timestamp
    st.session_state.messages.append({
      "role": "user",
      "content": st.session_state.user_input,
      "timestamp": timestamp
    })
    
    # Get bot response using RAG
    bot_response, related_files = ask_rag(st.session_state.user_input)
    
    # Add bot response to chat
    st.session_state.messages.append({
      "role": "assistant",
      "content": bot_response,
      "related_files": related_files,
      "timestamp": timestamp
    })
    
    # Clear input box
    st.session_state.user_input = ""

# Input box & send button in same row
col1, col2 = st.columns([4, 1])
with col1:
  user_input = st.text_input(
    "Type your message:", 
    key="user_input", 
    label_visibility="collapsed",
    on_change=send_message  # On pressing Enter
  )
with col2:
  # Use on_click to call the send_message function
  send_button = st.button("Send", on_click=send_message)
