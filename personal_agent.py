import streamlit as st
import openai
import random
import os
import uuid
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from threading import Thread

from llama_index.core import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.astra import AstraDBVectorStore

# Load environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
# Mock API endpoint (replace with actual API endpoint)
API_ENDPOINT = "http://localhost:8000/api/v1/anonymized-support"

# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for request body
class UserInputRequest(BaseModel):
    user_input: str

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

def search_for_therapists(locality: str = "Houston, Texas") -> str:
    pass
    # """Use the Google Search Tool but only specifically to find therapists in the client's area, then send email to update the client with the results."""
    # google_spec = GoogleSearchToolSpec(key=st.secrets.google_search_api_key, engine=st.secrets.google_search_engine)
    # tools = LoadAndSearchToolSpec.from_defaults(google_spec.to_tool_list()[0],).to_tool_list()
    # agent = OpenAIAgent.from_tools(tools, verbose=True)
    # response = agent.chat(f"what are the names of three individual therapists in {locality}?")
    # message = emails.html(
    #     html=f"<p>Hi Riley.<br>{response}</p>",
    #     subject="Helpful resources from TrevorChat",
    #     mail_from=('TrevorChat Counselor', 'contact@mychesscamp.com')
    # )
    # smtp_options = {
    #     "host": "smtp.gmail.com", 
    #     "port": 587,
    #     "user": "example@example.com", # To replace
    #     "password": "mypassword", # To replace   
    #     "tls": True
    # }
    # response = message.send(to='contact.email@gmail.com', smtp=smtp_options) # To replace with client's email
    # return f"Message sent: {response.success}"

def escalate() -> None:
    """Recognizes a high-risk statement from the mental health chatbot and escalates to the next level of management. High-risk is defined as a statement that suggests that the client has a plan, means, and intent to harm oneself or others (specific details on when, where, and how)."""
    st.error("High risk detected. Please consider escalating immediately.", icon="🚨")

def get_resource_for_response(user_input) -> str:
    """Recognizes a no, low- or medium-risk statement from the mental health chatbot, seeks resources to inform potential chat responses"""
    response = st.session_state.query_engine.retrieve(user_input)
    resources = [t.node.metadata["file_name"] for t in response]
    content = [t.node.text for t in response]
    result = dict(zip(resources, content))
    return result

@st.cache_resource(show_spinner=False)
def build_agent():
    agent = ReActAgent.from_tools([search_tool, escalate_tool, resource_tool], llm=OpenAI(model="gpt-4"), verbose=True)
    return agent

def get_modified_prompt(user_input) -> str:
    # Retrieve treatment plan components from session state
    anonymized_goals = st.session_state.api_data.get("anonymized_goals", [])
    anonymized_coping_strategies = st.session_state.api_data.get("anonymized_coping_strategies", [])
    actionable_insights = st.session_state.api_data.get("actionable_insights", "")

    # Convert treatment plan components into a structured prompt
    treatment_plan_prompt = f"""
    Treatment Plan:
    - Goals: {', '.join(anonymized_goals) if anonymized_goals else "No goals set yet."}
    - Coping Strategies: {', '.join(anonymized_coping_strategies) if anonymized_coping_strategies else "No coping strategies set yet."}
    - Actionable Insights: {actionable_insights if actionable_insights else "No actionable insights yet."}
    """

    # Combine treatment plan with user input in the prompt
    return f"""You are a friendly and supportive personal companion, offering non-therapy encouragement, emotional support, and positive influence. Your role is to be like a caring friend who listens and provides uplifting advice. When someone says '{user_input}', respond in a way that:
    1. Acknowledges their feelings with empathy and kindness.
    2. Offers encouragement and positive reinforcement.
    3. Suggests small, actionable steps they can take to feel better (e.g., going for a walk, journaling, or talking to a friend).
    4. Avoids giving clinical or therapeutic advice.
    5. **Always uses tools to retrieve resources without exception** to provide the most helpful and actionable response. Do not skip using resource retrieval tools under any circumstances.
    6. **Always considers using available tools** (e.g., searching for therapists, or escalating high-risk situations) to provide the most helpful and actionable response.
    7. Never says you are unable to help—always provide supportive and uplifting responses.
    8. **Incorporate the following treatment plan components into your response**:
    {treatment_plan_prompt}
    Keep your response conversational, warm, and limited to 4-5 lines.
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

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        astra_db_store = AstraDBVectorStore(
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            collection_name="test_astraDB_v3",
            embedding_dimension=1536,
        )
        storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        return index

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

search_tool = FunctionTool.from_defaults(fn=search_for_therapists)
escalate_tool = FunctionTool.from_defaults(fn=escalate)
resource_tool = FunctionTool.from_defaults(fn=get_resource_for_response)
index = load_data()
agent = build_agent()

if "query_engine" not in st.session_state.keys():
    st.session_state.query_engine = index.as_query_engine(similarity_top_k=3, verbose=True)

# User input
user_input = st.text_input("You:", placeholder="Type your message here...")

# Send button
if st.button("Send") and user_input.strip():
    st.session_state.conversation = []
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

# FastAPI Endpoint
@app.post("/chat")
def chat(request: UserInputRequest):
    try:
        # Generate an AI response
        bot_response = agent.chat(get_modified_prompt(request.user_input))
        suggested_reply = str(bot_response)
        suggested_reply = suggested_reply.split('"')[1] if '"' in suggested_reply else suggested_reply
        return {"response": suggested_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
