from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
import requests
from llama_index.llms.openai_like import OpenAILike

# Replace Streamlit session state with a global variable
api_data = None
session_id = str(uuid.uuid4())

# Pydantic model for request body
class UserInputRequest(BaseModel):
    user_input: str
    user_id: str  # Add user_id to the request model

def fetch_anonymized_data(session_id: str):
    response = requests.post(
        "http://localhost:8000/api/v1/anonymized-support",
        json={"session_id": session_id},
        headers={"Authorization": "Bearer your-secure-oauth2-token"}
    )
    return response.json() if response.status_code == 200 else None

def get_llm():
    return OpenAILike(
        api_base="http://127.0.0.1:11434/v1",
        model="gemma3:4b",
        api_key="sk-no-key-required",
        is_chat_model=True,
        stream=True,
        temperature=0.65,
        max_tokens=300,
        system_prompt="""Respond EXACTLY in this format:
        [DIRECT RESPONSE ONLY, NO INTRODUCTORY PHRASES]
        
        Rules:
        1. Never start with "Here's", "Okay", or similar
        2. No disclaimers or analysis references
        3. Maximum 2 sentences
        4. Use casual contractions (you're, don't)
        5. Respond like a close friend""",
        timeout=30
    )

def get_modified_prompt(user_input) -> str:
    # Retrieve treatment plan components from session state
    anonymized_goals = api_data.get("anonymized_goals", [])
    anonymized_coping_strategies = api_data.get("anonymized_coping_strategies", [])
    actionable_insights = api_data.get("actionable_insights", "")

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

# Response cleaner
def clean_response(text):
    removals = [
        "Okay, ", "Here's", "I'll", "Let me", "Based on",
        "natural response:", "empathetic response:",
        "framework:", "assessment:", "*", "\""
    ]
    for phrase in removals:
        text = text.replace(phrase, "")
    return text.strip().strip('"').strip(":").strip()

def process_form_field(llm, prompt):
    """Get cleaned response from LLM for form field population"""
    response = llm.complete(prompt).text
    return clean_response(response)

# Initialize FastAPI and LLM as before (omitting Streamlit-dependent functions)
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

llm = get_llm()

@app.post("/chat")
def chat(request: UserInputRequest):
    global api_data
    if api_data is None:
        api_data = fetch_anonymized_data(session_id)
    try:
        # Check if the user_id matches the allowed value
        if request.user_id != "tGcsMce6an1Hl4GtW0lQOQiOMyCFGD3v":
            raise HTTPException(status_code=403, detail="Invalid session ID")

        # Generate an AI response
        bot_response = process_form_field(
            llm, get_modified_prompt(request.user_input)
        )
        suggested_reply = str(bot_response)
        suggested_reply = suggested_reply.split('"')[1] if '"' in suggested_reply else suggested_reply
        return {"response": suggested_reply}
    except HTTPException as e:
        raise e  # Re-raise HTTPException for invalid session ID
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
