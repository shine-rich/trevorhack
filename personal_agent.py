import streamlit as st
import openai
import random
import os
import uuid
import requests

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

# Mock API endpoint (replace with actual API endpoint)
API_ENDPOINT = "http://localhost:8000/api/v1/anonymized-support"

# Simulate API request to fetch anonymized data
def fetch_anonymized_data(session_id: str):
    # Mock API request (replace with actual API call)
    response = requests.post(
        API_ENDPOINT,
        json={"session_id": session_id},
        headers={"Authorization": "Bearer your-secure-oauth2-token"}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch data from the API.")
        return None

@st.cache_resource(show_spinner=False)
def build_agent():
    agent = ReActAgent.from_tools([], llm=OpenAI(model="gpt-4"), verbose=True)
    return agent

def get_modified_prompt(user_input) -> str:
    return f"""You are a helpful mental health assistant chatbot, helping to train a junior counselor by providing suggestions on responses to client chat inputs. What would you recommend that the consider could say if someone says or asks '{user_input}'? Keep your responses limited to 4-5 lines; do not ask if the client needs more resources. If the case is not high risk, check for resources to help inform your response. If you need to send an email to share therapist contacts, call that action.
    """

# Load chatbot responses from a text file
def load_responses(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            responses = [line.strip() for line in file if line.strip()]
        return responses
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []

# Initialize session state for conversation history and API data
if "api_data" not in st.session_state:
    st.session_state.api_data = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate a unique session ID

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'openai_apikey' not in st.session_state:
    st.session_state.openai_apikey = st.secrets.openai_key

if st.session_state.openai_apikey:
    openai.api_key = st.session_state.openai_apikey

# Simulate chatbot interaction
if st.session_state.api_data is None:
    # Fetch anonymized data from the API
    st.session_state.api_data = fetch_anonymized_data(st.session_state.session_id)

# Title and description
st.title("🤖 Personal Agent")
st.write("Welcome! Share your thoughts, and I'll respond with supportive advice.")

if st.session_state.api_data:
    # Use anonymized data to drive the conversation
    anonymized_goals = st.session_state.api_data.get("anonymized_goals", [])
    anonymized_coping_strategies = st.session_state.api_data.get("anonymized_coping_strategies", [])
    actionable_insights = st.session_state.api_data.get("actionable_insights", "")

    # Display received mock data
    st.write("**Received Mock Data for Session 1:**")
    st.write(f"- Goals: {', '.join(anonymized_goals)}")
    st.write(f"- Coping Strategies: {', '.join(anonymized_coping_strategies)}")
    st.write(f"- Actionable Insights: {actionable_insights}")

if not st.session_state.openai_apikey:
    # Path to the chatbot responses file
    responses_file = "chatbot_responses.txt"

    # Load responses
    chatbot_responses = load_responses(responses_file)

    if not chatbot_responses:
        st.error("No responses available. Please check the chatbot_responses.txt file.")
    else:
        # User input
        user_input = st.text_input("You:", placeholder="Type your message here...")

        # Send button
        if st.button("Send") and user_input.strip():
            # Add user message to conversation history
            st.session_state.conversation.append(("You", user_input))

            # Select a random response from the script
            bot_response = random.choice(chatbot_responses)
            st.session_state.conversation.append(("Chatbot", bot_response))
        
        # Display conversation history
        st.write("---")
        st.subheader("Conversation:")
        for speaker, message in st.session_state.conversation:
            st.write(f"**{speaker}:** {message}")

        # Reset button to clear the conversation
        if st.button("Reset Conversation"):
            st.session_state.conversation = []
else:
    agent = build_agent()

    # User input
    user_input = st.text_input("You:", placeholder="Type your message here...")

    # Send button
    if st.button("Send") and user_input.strip():
        # Add user message to conversation history
        st.session_state.conversation.append(("You", user_input))

        # Generate an AI response
        try: 
            bot_response = agent.chat(get_modified_prompt(user_input))
        except:
            bot_response = "LLM not available"
        suggested_reply = str(bot_response)
        suggested_reply = suggested_reply.split('"')[1] if '"' in suggested_reply else suggested_reply        
        st.session_state.conversation.append(("Chatbot", suggested_reply))

    # Display conversation history
    st.write("---")
    st.subheader("Conversation:")
    for speaker, message in st.session_state.conversation:
        st.write(f"**{speaker}:** {message}")

    # Reset button to clear the conversation
    if st.button("Reset Conversation"):
        st.session_state.conversation = []