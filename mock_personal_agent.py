import streamlit as st
import time

# Mock API endpoint (replace with actual API endpoint)
API_ENDPOINT = "http://localhost:8000/api/v1/anonymized-support"

# Sample demo conversation file (replace with actual file path)
DEMO_CONVERSATION_FILE = "demo_chatbot_conversation.txt"

# Load the demo conversation from the text file
def load_demo_conversation():
    with open(DEMO_CONVERSATION_FILE, "r") as file:
        return file.readlines()

# Simulate API request to fetch anonymized data
def fetch_anonymized_data(session_id: str):
    # Mock API request with different data for each session ID
    if session_id == "session_1":
        return {
            "anonymized_goals": ["Practice mindfulness daily", "Journal for 10 minutes"],
            "anonymized_coping_strategies": ["Deep breathing exercises", "Take a walk"],
            "actionable_insights": "Based on your progress, here are some suggestions to help you stay on track."
        }
    elif session_id == "session_2":
        return {
            "anonymized_goals": ["Exercise for 30 minutes", "Limit screen time before bed"],
            "anonymized_coping_strategies": ["Progressive muscle relaxation", "Listen to calming music"],
            "actionable_insights": "It looks like you're making progress. Here are some additional strategies to try."
        }
    else:
        return None

# Initialize session state for conversation and API data
if "messages_1" not in st.session_state:
    st.session_state.messages_1 = []

if "messages_2" not in st.session_state:
    st.session_state.messages_2 = []

if "api_data_1" not in st.session_state:
    st.session_state.api_data_1 = None

if "api_data_2" not in st.session_state:
    st.session_state.api_data_2 = None

# Load the demo conversation
demo_conversation = load_demo_conversation()

# Streamlit app layout
st.title("Personal AI Chatbot")
st.write("Welcome to your personal mental health companion. How can I help you today?")

# Create two columns for side-by-side layout
col1, col2 = st.columns(2)

time.sleep(2)

# Column 1: Session 1
with col1:
    st.subheader("Session 1")
    if st.session_state.api_data_1 is None:
        # Fetch anonymized data from the API for session 1
        st.session_state.api_data_1 = fetch_anonymized_data("session_1")

    if st.session_state.api_data_1:
        # Use anonymized data to drive the conversation
        anonymized_goals = st.session_state.api_data_1.get("anonymized_goals", [])
        anonymized_coping_strategies = st.session_state.api_data_1.get("anonymized_coping_strategies", [])
        actionable_insights = st.session_state.api_data_1.get("actionable_insights", "")

        # Display received mock data
        st.write("**Received Mock Data for Session 1:**")
        st.write(f"- Goals: {', '.join(anonymized_goals)}")
        st.write(f"- Coping Strategies: {', '.join(anonymized_coping_strategies)}")
        st.write(f"- Actionable Insights: {actionable_insights}")

        # Simulate conversation based on demo file
        for line in demo_conversation:
            role, content = line.split(":", 1)
            role = role.strip()
            content = content.strip()

            # Replace placeholders with anonymized data
            if "goals" in content.lower():
                content = f"Your current goals are: {', '.join(anonymized_goals)}. How can I help you work towards them?"
            elif "coping strategies" in content.lower():
                content = f"Here are some coping strategies you can try: {', '.join(anonymized_coping_strategies)}. Would you like to explore one of these?"
            elif "actionable insights" in content.lower():
                content = actionable_insights

            # Add message to session state
            st.session_state.messages_1.append({"role": role, "content": content})

            # Display message in chat
            with st.chat_message(role):
                st.write(content)

            # Simulate chatbot processing time
            time.sleep(1)

        # Contextually relevant ending for Session 1
        st.session_state.messages_1.append({"role": "user", "content": "I've tried journaling, but it hasn't been helping much."})
        with st.chat_message("user"):
            st.write("I've tried journaling, but it hasn't been helping much.")

        chatbot_response = f"Journaling is a great start. Here are some additional strategies you can try: {', '.join(anonymized_coping_strategies)}. Would you like to explore one of these?"
        st.session_state.messages_1.append({"role": "assistant", "content": chatbot_response})
        with st.chat_message("assistant"):
            st.write(chatbot_response)

# Column 2: Session 2
with col2:
    st.subheader("Session 2")
    if st.session_state.api_data_2 is None:
        # Fetch anonymized data from the API for session 2
        st.session_state.api_data_2 = fetch_anonymized_data("session_2")

    if st.session_state.api_data_2:
        # Use anonymized data to drive the conversation
        anonymized_goals = st.session_state.api_data_2.get("anonymized_goals", [])
        anonymized_coping_strategies = st.session_state.api_data_2.get("anonymized_coping_strategies", [])
        actionable_insights = st.session_state.api_data_2.get("actionable_insights", "")

        # Display received mock data
        st.write("**Received Mock Data for Session 2:**")
        st.write(f"- Goals: {', '.join(anonymized_goals)}")
        st.write(f"- Coping Strategies: {', '.join(anonymized_coping_strategies)}")
        st.write(f"- Actionable Insights: {actionable_insights}")

        # Simulate conversation based on demo file
        for line in demo_conversation:
            role, content = line.split(":", 1)
            role = role.strip()
            content = content.strip()

            # Replace placeholders with anonymized data
            if "goals" in content.lower():
                content = f"Your current goals are: {', '.join(anonymized_goals)}. How can I help you work towards them?"
            elif "coping strategies" in content.lower():
                content = f"Here are some coping strategies you can try: {', '.join(anonymized_coping_strategies)}. Would you like to explore one of these?"
            elif "actionable insights" in content.lower():
                content = actionable_insights

            # Add message to session state
            st.session_state.messages_2.append({"role": role, "content": content})

            # Display message in chat
            with st.chat_message(role):
                st.write(content)

            # Simulate chatbot processing time
            time.sleep(1)

        # Contextually relevant ending for Session 2
        st.session_state.messages_2.append({"role": "user", "content": "I've tried limiting screen time, but it hasn't been helping much."})
        with st.chat_message("user"):
            st.write("I've tried limiting screen time, but it hasn't been helping much.")

        chatbot_response = f"Limiting screen time is a good effort. Here are some additional strategies you can try: {', '.join(anonymized_coping_strategies)}. Would you like to explore one of these?"
        st.session_state.messages_2.append({"role": "assistant", "content": chatbot_response})
        with st.chat_message("assistant"):
            st.write(chatbot_response)