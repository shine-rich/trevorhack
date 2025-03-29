import streamlit as st
from streamlit_elements import elements, mui, html, dashboard, nivo
import numpy as np
import openai
import time
import helpers
import emails
import random
import os
from dotenv import load_dotenv

from llama_index.core import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.vector_stores.astra import AstraDBVectorStore

from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
# from llama_hub.tools.google_search.base import GoogleSearchToolSpec

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Client conversation idx initialization
client_script = open("data/library/demo_conversation_client.txt", "r").readlines()
if 'openai_apikey' not in st.session_state:
    st.session_state.openai_apikey = st.secrets.openai_key

if 'script_idx' not in st.session_state:
    st.session_state.script_idx = 1

if 'custom_chat_input' not in st.session_state:
    st.session_state.custom_chat_input = ''  # Initialize it with an empty string

if 'should_end_session' not in st.session_state:
    st.session_state.should_end_session = False

if 'should_suggest_reply_using_ai' not in st.session_state:
    st.session_state.should_suggest_reply_using_ai = False

if 'mood' not in st.session_state:
    st.session_state.mood = None

if 'suggested_reply' not in st.session_state:
    st.session_state.suggested_reply = ""

if 'longitudinal_data' not in st.session_state:
    st.session_state.longitudinal_data = []  # For storing historical data

if 'longitudinal_data_history' not in st.session_state:
    st.session_state.longitudinal_data_history = []

if 'should_populate_form_using_ai' not in st.session_state:
    st.session_state.should_populate_form_using_ai = False

if 'should_display_latest_treatment_plan' not in st.session_state:
    st.session_state.should_display_latest_treatment_plan = True

if "trigger_log" not in st.session_state:
    st.session_state.update({
        "trigger_log": [
            {"date": "05/20", "description": "Missed medication", "severity": 3},
            {"date": "05/22", "description": "Work deadline stress", "severity": 2}
        ],
        "coping_strategies": [
            {
                "name": "5-4-3-2-1 Grounding",
                "duration": "3 mins",
                "instructions": "Name 5 things you see, 4 you can touch...",
                "last_used": None,
                "helpful": None
            }
        ],
        "risk_level": "Medium"  # Dynamically updated from radar
    })

if "missed_reasons" not in st.session_state:
    st.session_state.update({
        "missed_reasons": {},
        "coping_strategies": [
            {
                "name": "5-Minute CBT",
                "duration": "5 mins",
                "instructions": "",
                "last_used": None,
                "helpful": None
            }
        ],
        "adherence_data": {
            "CBT": {"target": 5, "actual": 3},
            "Medication": {"target": 7, "actual": 4}
        }
    })

# In your session state initialization:
if "current_case" not in st.session_state:
    st.session_state.current_case = 0
    st.session_state.case_queue = [
        {
            "id": 101,
            "name": "Alex Chen",
            "priority": "High",
            "last_session": "2023-06-15",
            "adherence_alerts": ["Journaling", "Medication"],
            "suggestions": [...]  # Your auto-generated suggestions
        },
        # ...other cases
    ]

# Initialize session state for Case Form fields
if 'case_form_data' not in st.session_state:
    st.session_state.case_form_data = {
        "first_name": "",
        "reason_for_contact": "",
        "age": 0,
        "location": "",
        "gender_identity": "",
        "preferred_pronouns": "",
        "suicidal_ideation": "No",
        "risk_level": "Medium Risk",
        "brief_summary": "",
        "coping_strategies": "",
        "goals": "",
        "progress": "",
        "emergency_contact_name": "",
        "relationship": "",
        "phone_number": "",
        "previous_mental_health_history": "",
        "follow_up_actions": "",
        "next_session_date": ""
    }

st.set_page_config(page_title="Santo Chat, powered by LlamaIndex", page_icon="💬", layout="wide")

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

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0, system_prompt="You are an expert and sensitive mental health copilot assistant for a mental health counselor. Your job is to help the counselor by providing suggestions based on reference documents."))
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        return index

@st.cache_resource(show_spinner=False)
def build_agent():
    agent = ReActAgent.from_tools([search_tool, escalate_tool, resource_tool], llm=OpenAI(model="gpt-4"), verbose=True)
    return agent

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

# Longitudinal Database Integration
def save_to_longitudinal_database(data):
    st.session_state.longitudinal_data.append(data)
    # st.sidebar.json(data)

def search_for_therapists(locality: str = "Houston, Texas") -> str:
    pass
    # """Use the Google Search Tool but only specifically to find therapists in the client's area, then send email to update the client with the results."""
    # google_spec = GoogleSearchToolSpec(key=st.secrets.google_search_api_key, engine=st.secrets.google_search_engine)
    # tools = LoadAndSearchToolSpec.from_defaults(google_spec.to_tool_list()[0],).to_tool_list()
    # agent = OpenAIAgent.from_tools(tools, verbose=True)
    # response = agent.chat(f"what are the names of three individual therapists in {locality}?")
    # message = emails.html(
    #     html=f"<p>Hi Riley.<br>{response}</p>",
    #     subject="Helpful resources from Santo Chat",
    #     mail_from=('Santo Chat Counselor', 'contact@mychesscamp.com')
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

@st.cache_resource(show_spinner=False)
def get_counselor_resources(_response) -> list:
    output = ['cheatsheet_empathetic_language.txt', 'cheatsheet_maintaining_rapport.txt', 'cheatsheet_risk_assessment.txt']
    try:
        # uncomment the followings only when using vector store
        # raw_output = _response.sources[0].raw_output
        # output_dict = dict(raw_output)
        # output = [key for key in output_dict.keys()]
        pass
    except: # Hard-coded documents in case of agent failure
        return output
    return output

@st.cache_resource(show_spinner=False)
def client_summary() -> str:
    return helpers.CLIENT_SUMMARY

def get_modified_prompt(user_input) -> str:
    return f"""You are a friendly and supportive personal companion, offering non-therapy encouragement, emotional support, and positive influence. Your role is to be like a caring friend who listens and provides uplifting advice. When someone says '{user_input}', respond in a way that:
    1. Acknowledges their feelings with empathy and kindness.
    2. Offers encouragement and positive reinforcement.
    3. Suggests small, actionable steps they can take to feel better (e.g., going for a walk, journaling, or talking to a friend).
    4. Avoids giving clinical or therapeutic advice.
    5. **Always uses tools to retrieve resources without exception** to provide the most helpful and actionable response. Do not skip using resource retrieval tools under any circumstances.
    6. **Always considers using available tools** (e.g., searching for therapists, or escalating high-risk situations) to provide the most helpful and actionable response.
    7. Never says you are unable to help—always provide supportive and uplifting responses.
    """

def send_chat_message():
    if st.session_state.custom_chat_input:  # Check if the input is not empty
        st.session_state.messages.append({"role": "user", "content": st.session_state.custom_chat_input})
        st.session_state.custom_chat_input = ""
        st.session_state.suggested_reply = ""

def set_custom_chat_input():
    st.session_state.custom_chat_input = st.session_state.suggested_reply

def get_form_value_from_convo(convo, form_value) -> str:
    return f"""You are a helpful assistant filling out a form. Extract the person's {form_value} from the following converstation in to input into the form. {convo}"""

def get_int_value_from_convo(convo, form_value) -> str:
    return f"""You are a helpful assistant filling out a form. Extract the person's {form_value} from the following converstation in to input into the form. {convo}"""

def get_risk_value_from_convo(convo) -> str:
    return f"""You are a helpful assistant filling out a form. Reply 0 if the person does not seem at risk based on the conversation. {convo}"""

def mood_selectbox():
    column_options = ["Happy", "Neutral", "Stressed", "Overwhelmed"]
    st.session_state.mood = st.selectbox(
        "How are you feeling today?", 
        column_options,
        key="column_selectbox", 
        index=column_options.index(st.session_state.mood) if st.session_state.mood in column_options else 0
    )

def display_sentiment_analysis(student_message):
    # Hardcoded sentiment values for demo
    mock_sentiment = {
        "label": "POSITIVE",  # Could be NEGATIVE/NEUTRAL
        "score": 0.92  # Confidence score between 0-1
    }

    # Display as a metric card
    st.metric("Current Mood 🌟", 
            f"{mock_sentiment['label'].title()} ({mock_sentiment['score']:.2f})",
            delta="improving" if mock_sentiment['label'] == "POSITIVE" else "declining")

def display_risk_dashboard(student_message):
    # Hardcoded risk keywords and demo scoring
    risk_keywords = {
        "suicide": 3,
        "harm": 2,
        "self-harm": 3,
        "hopeless": 1,
        "worthless": 1
    }
    
    # Simulate risk detection
    detected_risks = {word: count for word, count in risk_keywords.items() 
                     if word in student_message.lower()}
    
    # Calculate mock risk score
    risk_score = sum(detected_risks.values()) 
    risk_level = "Low" if risk_score < 2 else "Medium" if risk_score < 4 else "High"
    
    # Create columns layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Risk level indicator with color coding
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; 
                    background-color: {"#ffcccc" if risk_level == "High" else 
                                      "#fff3cd" if risk_level == "Medium" else 
                                      "#d4edda"};
                    text-align: center;'>
            <h3 style='color: {"#721c24" if risk_level == "High" else 
                              "#856404" if risk_level == "Medium" else 
                              "#155724"};'>
                Risk Level: {risk_level}
            </h3>
            <p>Score: {risk_score}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Detected keywords display
        if detected_risks:
            st.write("**Detected Risk Indicators:**")
            for word, severity in detected_risks.items():
                st.write(f"- {word.capitalize()} (severity: {'⭐' * severity})")
        else:
            st.success("✅ No high-risk indicators detected")

def suggest_cbt_techniques(student_message):
    # Hardcoded technique database
    cbt_library = {
        "anxiety": [
            "🌬️ 4-7-8 Breathing: Inhale 4s, hold 7s, exhale 8s",
            "📝 Thought Challenging: 'What evidence supports this worry?'"
        ],
        "depression": [
            "🎯 Behavioral Activation: Schedule 1 enjoyable activity today",
            "🌈 Positive Affirmations: 'I am capable of overcoming challenges'"
        ],
        "anger": [
            "🕑 Time-Out Technique: Pause for 10 minutes before responding",
            "📊 Cost-Benefit Analysis: List pros/cons of angry reaction"
        ]
    }

    # Mock theme detection
    detected_themes = []
    if any(word in student_message.lower() for word in ["worry", "anxious"]):
        detected_themes.append("anxiety")
    if any(word in student_message.lower() for word in ["sad", "hopeless"]):
        detected_themes.append("depression")
    if any(word in student_message.lower() for word in ["angry", "frustrated"]):
        detected_themes.append("anger")

    # Display suggestions
    with st.container(border=True):
        st.subheader("🧠 Recommended CBT Techniques")
        
        if detected_themes:
            for theme in detected_themes[:2]:  # Show max 2 themes
                st.markdown(f"**{theme.title()} Interventions:**")
                for technique in cbt_library.get(theme, [])[:2]:  # Show 2 techniques per theme
                    st.write(f"- {technique}")
                st.divider()
        else:
            st.write("💡 General Wellness Suggestion:")
            st.write("- 🚶♂️ 5-Minute Mindful Walk: Focus on sensory experiences")

def display_conversation_themes(chat_history):
    # Hardcoded theme detection
    theme_keywords = {
        "Academic Stress": ["school", "exam", "homework"],
        "Family Dynamics": ["parent", "family", "mom", "dad"],
        "Social Anxiety": ["friend", "social", "crowd"],
        "Self-Esteem": ["worth", "confidence", "ugly"]
    }
    
    # Mock analysis
    detected_themes = []
    for theme, keywords in theme_keywords.items():
        if any(keyword in chat_history.lower() for keyword in keywords):
            detected_themes.append(theme)
    
    # Display in columns
    with st.expander("🔍 Conversation Themes Analysis", expanded=True):
        if detected_themes:
            cols = st.columns(2)
            for i, theme in enumerate(detected_themes[:4]):  # Max 4 themes
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style='padding:0.5rem; margin:0.5rem 0; 
                                border-radius:0.3rem; background:#f0f2f6;'>
                        <b>{theme}</b>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.write("🌟 Primary Theme: General Well-being Check-in")

def parse_and_update_treatment_plan(query, longitudinal_data):
    """
    Parse the update query and either update the latest treatment plan or add a new one.
    """
    try:
        # Save the current state to history before making changes
        st.session_state.longitudinal_data_history.append({
            "timestamp": time.time(),
            "data": [entry.copy() for entry in longitudinal_data]  # Deep copy of the data
        })

        # Check if the query is to add a new treatment plan
        if query.lower().startswith("add a new treatment plan"):
            # Extract progress and next steps from the query
            progress = None
            next_steps = None

            if "progress =" in query:
                progress = query.split("progress =")[1].split("and")[0].strip().strip("'")
            if "next step =" in query:
                next_steps = query.split("next step =")[1].strip().strip("'")

            # Create a new treatment plan entry
            new_entry = {
                "timestamp": time.time(),
                "progress": progress if progress else "",
                "next_steps": next_steps if next_steps else "",
                "goals": "",
                "coping_strategies": "",
                "feedback": "",
            }

            # Add the new entry to the longitudinal data
            longitudinal_data.append(new_entry)
            return new_entry

        else:
            # Update the latest treatment plan (existing logic)
            if not longitudinal_data:
                raise ValueError("No treatment plan found to update.")

            latest_entry = longitudinal_data[-1]

            # Parse the updates from the query
            updates = {}
            if "add progress =" in query:
                progress = query.split("add progress =")[1].split("and")[0].strip().strip("'")
                updates["progress"] = progress
            if "next step =" in query:
                next_step = query.split("next step =")[1].strip().strip("'")
                updates["next_steps"] = next_step

            # Apply the updates to the latest entry
            for key, value in updates.items():
                latest_entry[key] = value

            return latest_entry

    except Exception as e:
        raise ValueError(f"Failed to parse and update treatment plan: {str(e)}")

def undo_last_update():
    """
    Revert the treatment plan to the previous state.
    """
    if st.session_state.longitudinal_data_history:
        # Get the last saved state
        last_state = st.session_state.longitudinal_data_history.pop()
        st.session_state.longitudinal_data = last_state["data"]
        st.success("Last update reverted successfully!")
    else:
        st.warning("No changes to undo.")

def display_latest_treatment_plan(latest_entry, previous_entry=None):
    """
    Display the latest treatment plan with animations for changes.
    """
    st.subheader(f"Latest Treatment Plan ({time.strftime('%e %b %Y %H:%M:%S%p', time.localtime(latest_entry['timestamp']))})")

    # Define a function to highlight changes with animations
    def highlight_change(current_value, previous_value, field_name):
        if previous_entry and current_value != previous_value.get(field_name, ""):
            return f"<span class='updated-field'>{current_value}</span>"
        return current_value

    # Display each field with visual feedback
    if "goals" in latest_entry:
        st.write(f"**Goals:** {highlight_change(latest_entry['goals'], previous_entry, 'goals')}", unsafe_allow_html=True)
    if "coping_strategies" in latest_entry:
        st.write(f"**Coping Strategies:** {highlight_change(latest_entry['coping_strategies'], previous_entry, 'coping_strategies')}", unsafe_allow_html=True)
    if "progress" in latest_entry:
        st.write(f"**Progress:** {highlight_change(latest_entry['progress'], previous_entry, 'progress')}", unsafe_allow_html=True)
    if "feedback" in latest_entry:
        st.write(f"**Feedback:** {highlight_change(latest_entry['feedback'], previous_entry, 'feedback')}", unsafe_allow_html=True)
    if "next_steps" in latest_entry:
        st.write(f"**Next Steps:** {highlight_change(latest_entry['next_steps'], previous_entry, 'next_steps')}", unsafe_allow_html=True)
    st.markdown("---")

def mark_strategy_tried(index):
    st.session_state.coping_strategies[index]["last_used"] = datetime.now()
    st.rerun()

def flag_unhelpful_strategy(index):
    st.session_state.coping_strategies[index]["helpful"] = False
    st.rerun()

def generate_adjustments(case_data):
    suggestions = []
    
    # Rule 1: Missed activities
    for activity in case_data["missed_activities"]:
        if "Journaling" in activity:
            suggestions.append({
                "activity": "Journaling",
                "recommendation": "Switch to bullet-point format",
                "evidence": f"Missed {activity.split('(')[1]} in past week"
            })
    
    # Rule 2: Low adherence + high risk
    if case_data["adherence_rate"] < 0.7 and case_data["risk_level"] == "High":
        suggestions.append({
            "activity": "All",
            "recommendation": "Simplify plan to 2-3 core activities",
            "evidence": "High risk + low adherence"
        })
    
    return suggestions

if st.session_state.openai_apikey:
    search_tool = FunctionTool.from_defaults(fn=search_for_therapists)
    escalate_tool = FunctionTool.from_defaults(fn=escalate)
    resource_tool = FunctionTool.from_defaults(fn=get_resource_for_response)
    index = load_data()
    agent = build_agent()
    if "query_engine" not in st.session_state.keys():
        st.session_state.query_engine = index.as_query_engine(similarity_top_k=3, verbose=True)
    # st.sidebar.write("Longitudinal Data:")
    # st.sidebar.json(st.session_state.longitudinal_data)
    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Crisis Hotline Chat", "Case Form", "Treatment Plans & Feedback"])
    
    # Define valid options for Suicidal Ideation
    suicidal_ideation_options = ["No", "Passive", "Active with Plan", "Active without Plan"]
    risk_level_options = ["Not Suicidal", "Low Risk", "Medium Risk", "High Risk", "Imminent Risk"]

    # Crisis intervention room tab
    with tab1:
        col_a1, col_a2 = st.columns([0.5, 0.5], gap="small")
        with col_a1:
            if "chat_engine" not in st.session_state.keys():
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"role": "user", "content": "Hi, welcome to Santo Chat. Can you tell me a little bit about yourself?"}]
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    if st.session_state.script_idx < len(client_script):
                        response = client_script[st.session_state.script_idx][2:]
                        st.session_state.script_idx += 2
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)  # Add response to message history
                    elif not st.session_state.should_end_session:
                        st.info("Contact has left the chat")
                        st.session_state.should_end_session = True
                        with st.status("Autopopulating form...", expanded=False) as status:
                            st.write("Downloading chat history...")
                            chathistory = ""
                            for item in st.session_state.messages:
                                chathistory += item['content'] + '\n'
                            st.write("Analyzing chat...")
                            if st.session_state.should_populate_form_using_ai:
                                # Populate Case Form data in session state
                                st.session_state.case_form_data["first_name"] = agent.chat(get_form_value_from_convo(chathistory, "First Name")).response
                                st.session_state.case_form_data["reason_for_contact"] = agent.chat(get_form_value_from_convo(chathistory, "Reason for Contact")).response
                                try:
                                    st.session_state.case_form_data["age"] = int(agent.chat(get_int_value_from_convo(chathistory, "Age")).response)
                                except ValueError:
                                    st.session_state.case_form_data["age"] = 0  # Default value if conversion fails
                                st.session_state.case_form_data["location"] = agent.chat(get_form_value_from_convo(chathistory, "Location (City, State)")).response
                                st.session_state.case_form_data["gender_identity"] = agent.chat(get_form_value_from_convo(chathistory, "Gender Identity")).response
                                st.session_state.case_form_data["preferred_pronouns"] = agent.chat(get_form_value_from_convo(chathistory, "Preferred Pronouns")).response
                                suicidal_ideation_response = agent.chat(get_form_value_from_convo(chathistory, "Suicidal Ideation")).response
                                if suicidal_ideation_response in suicidal_ideation_options:
                                    st.session_state.case_form_data["suicidal_ideation"] = suicidal_ideation_response
                                else:
                                    st.session_state.case_form_data["suicidal_ideation"] = "No"  # Default value
                                risk_level_response = agent.chat(get_form_value_from_convo(chathistory, "Suicidal Ideation")).response
                                if risk_level_response in risk_level_options:
                                    st.session_state.case_form_data["risk_level"] = risk_level_response
                                else:
                                    st.session_state.case_form_data["risk_level"] = "Medium Risk"  # Default value
                                st.session_state.case_form_data["brief_summary"] = agent.chat(get_form_value_from_convo(chathistory, "Brief Summary/Narrative")).response
                                st.session_state.case_form_data["coping_strategies"] = agent.chat(get_form_value_from_convo(chathistory, "Current Coping Strategies")).response
                                st.session_state.case_form_data["goals"] = agent.chat(get_form_value_from_convo(chathistory, "Goals for This Session")).response
                                st.session_state.case_form_data["progress"] = agent.chat(get_form_value_from_convo(chathistory, "Progress Toward Previous Goals")).response
                                st.session_state.case_form_data["emergency_contact_name"] = agent.chat(get_form_value_from_convo(chathistory, "Emergency Contact Name")).response
                                st.session_state.case_form_data["relationship"] = agent.chat(get_form_value_from_convo(chathistory, "Relationship")).response
                                st.session_state.case_form_data["phone_number"] = agent.chat(get_form_value_from_convo(chathistory, "Phone Number")).response
                                st.session_state.case_form_data["previous_mental_health_history"] = agent.chat(get_form_value_from_convo(chathistory, "Previous Mental Health History")).response
                                st.session_state.case_form_data["follow_up_actions"] = agent.chat(get_form_value_from_convo(chathistory, "Follow-Up Actions")).response
                                st.session_state.case_form_data["next_session_date"] = agent.chat(get_form_value_from_convo(chathistory, "Next Session Date")).response
                            else:
                                # Populate Case Form data in session state with more realistic demo data
                                st.session_state.case_form_data["first_name"] = "Alex"
                                st.session_state.case_form_data["reason_for_contact"] = "Feeling overwhelmed with work and personal life balance."
                                st.session_state.case_form_data["age"] = 29
                                st.session_state.case_form_data["location"] = "San Francisco, CA"
                                st.session_state.case_form_data["gender_identity"] = "Non-binary"
                                st.session_state.case_form_data["preferred_pronouns"] = "They/Them"
                                st.session_state.case_form_data["suicidal_ideation"] = "Passive"
                                st.session_state.case_form_data["risk_level"] = "Medium Risk"
                                st.session_state.case_form_data["brief_summary"] = "Alex has been feeling overwhelmed due to increasing work pressure and personal responsibilities. They have expressed passive suicidal ideation but no specific plans or means."
                                st.session_state.case_form_data["coping_strategies"] = "Meditation, journaling, and occasional walks."
                                st.session_state.case_form_data["goals"] = "Develop better time management skills and find effective stress-relief techniques."
                                st.session_state.case_form_data["progress"] = "Alex has started practicing mindfulness exercises and is working on setting boundaries at work."
                                st.session_state.case_form_data["emergency_contact_name"] = "Jordan Smith"
                                st.session_state.case_form_data["relationship"] = "Friend"
                                st.session_state.case_form_data["phone_number"] = "555-1234"
                                st.session_state.case_form_data["previous_mental_health_history"] = "Previous episodes of anxiety managed with therapy."
                                st.session_state.case_form_data["follow_up_actions"] = "Schedule next session, provide resources on stress management, follow up with emergency contact if necessary."
                                st.session_state.case_form_data["next_session_date"] = "2025-11-15"
                            st.write("Populating form...")
                            save_to_longitudinal_database({
                                "timestamp": time.time(),
                                "name": st.session_state.case_form_data["first_name"],
                                "issue": st.session_state.case_form_data["reason_for_contact"],
                                "risk_level": st.session_state.case_form_data["risk_level"],
                                "coping_strategies": st.session_state.case_form_data["coping_strategies"],
                                "progress": "Moderate improvement noted."
                            })

        with col_a2:
            # Toggle for should_populate_form_using_ai
            st.subheader("Settings")
            st.session_state.should_populate_form_using_ai = st.toggle(
                "Populate Case Form Using AI", 
                value=st.session_state.get("should_populate_form_using_ai", False)
            )
            # Toggle for should_populate_form_using_ai
            st.session_state.should_suggest_reply_using_ai = st.toggle(
                "Suggest Replies Using AI", 
                value=st.session_state.get("should_suggest_reply_using_ai", False)
            )

            st.subheader("Live Insights 🔍")
            with st.expander("Case Overview", expanded=False):
                if len(st.session_state.messages) > 1:
                    with st.spinner("Loading..."):
                        st.write(client_summary())
            with st.expander("Emotional State & Risk Analysis", expanded=False):
                display_sentiment_analysis(st.session_state.messages[-1]["content"])  # Custom function
                display_risk_dashboard(st.session_state.messages[-1]["content"])
            
            st.subheader("Tools & Resources 🛠️")
            with st.expander("CBT Techniques", expanded=False):
                suggest_cbt_techniques(st.session_state.messages[-1]["content"])      # Custom function
            
            with st.expander("Quick Actions", expanded=False):
                counselor_note = st.text_input("Add Session Note")
                st.button("🚨 Escalate Case")
                st.button("📝 Save Note to Form")
            
            st.subheader("Session Themes 📌")
            display_conversation_themes(st.session_state.messages[-1]["content"])         # Custom function
            
            st.subheader("Suggested Reply")
            source_file_names = ['cheatsheet_empathetic_language.txt', 'cheatsheet_maintaining_rapport.txt', 'cheatsheet_risk_assessment.txt']
            if not st.session_state.suggested_reply:   # Check if the input is empty
                if st.session_state.should_suggest_reply_using_ai:
                    try: 
                        response = agent.chat(get_modified_prompt(st.session_state.messages[-1]["content"]))
                        st.session_state.suggested_reply = str(response)
                        source_file_names = get_counselor_resources(response)
                    except:
                        st.session_state.suggested_reply = client_script[st.session_state.script_idx-1][2:]
                else:
                    st.session_state.suggested_reply = client_script[st.session_state.script_idx-1][2:]
                st.session_state.suggested_reply = st.session_state.suggested_reply.split('"')[1] if '"' in st.session_state.suggested_reply else st.session_state.suggested_reply
            st.info(st.session_state.suggested_reply)
            if st.button("Use Suggested Reply", on_click=set_custom_chat_input):
                pass

            st.subheader("Sources")
            source_links = [
                "https://www.nimh.nih.gov/health/publications/stress/index.shtml",
                "https://www.apa.org/topics/resilience",
                "https://www.mind.org.uk/information-support/types-of-mental-health-problems/"
            ]
            source_titles = [
                "NIMH Coping with Stress",
                "APA Building Your Resilience",
                "Mind Coping with Mental Health Problems"
            ]
            i = 0
            sources_row = st.columns(len(source_links))
            for col in sources_row:
                with col.container(height=50):
                    st.markdown(f'<a href="{source_links[i]}" target="_blank">"{source_titles[i]}"</a>', unsafe_allow_html=True)
                i += 1

        with col_a1:  
            if not st.session_state.should_end_session:
                with st.form("chat_form"):
                    custom_chat_input = st.text_area("Your reply", key="custom_chat_input")
                    _, right_justified_button_col = st.columns([0.8, 0.15])   # Adjust the ratio as needed
                    with right_justified_button_col:
                        submit_button = st.form_submit_button("Send :incoming_envelope:", on_click=send_chat_message)

    # Case Form Tab
    with tab2:
        st.header("Case Form")
        col_b1, col_b2 = st.columns([0.5, 0.5], gap="small")
        with col_b1:
            st.session_state.case_form_data["first_name"] = st.text_input("First Name", value=st.session_state.case_form_data["first_name"])
            st.session_state.case_form_data["reason_for_contact"] = st.text_input("Reason for Contact", value=st.session_state.case_form_data["reason_for_contact"])
            st.session_state.case_form_data["age"] = st.number_input("Age", value=st.session_state.case_form_data["age"])
            st.session_state.case_form_data["location"] = st.text_input("Location (City, State)", value=st.session_state.case_form_data["location"])
            st.session_state.case_form_data["gender_identity"] = st.text_input("Gender Identity", value=st.session_state.case_form_data["gender_identity"])
            st.session_state.case_form_data["preferred_pronouns"] = st.text_input("Preferred Pronouns", value=st.session_state.case_form_data["preferred_pronouns"])
            st.session_state.case_form_data["suicidal_ideation"] = st.selectbox(
                "Suicidal Ideation",
                suicidal_ideation_options,
                index=suicidal_ideation_options.index(st.session_state.case_form_data["suicidal_ideation"])
            )
            st.session_state.case_form_data["risk_level"] = st.selectbox(
                "Risk Level",
                risk_level_options,
                index=risk_level_options.index(st.session_state.case_form_data["risk_level"])
            )
        with col_b2:
            st.session_state.case_form_data["brief_summary"] = st.text_area("Brief Summary/Narrative", value=st.session_state.case_form_data["brief_summary"])
            st.session_state.case_form_data["coping_strategies"] = st.text_area("Current Coping Strategies", value=st.session_state.case_form_data["coping_strategies"])
            st.session_state.case_form_data["goals"] = st.text_area("Goals for This Session", value=st.session_state.case_form_data["goals"])
            st.session_state.case_form_data["progress"] = st.text_area("Progress Toward Previous Goals", value=st.session_state.case_form_data["progress"])
            st.session_state.case_form_data["emergency_contact_name"] = st.text_input("Emergency Contact Name", value=st.session_state.case_form_data["emergency_contact_name"])
            st.session_state.case_form_data["relationship"] = st.text_input("Relationship", value=st.session_state.case_form_data["relationship"])
            st.session_state.case_form_data["phone_number"] = st.text_input("Phone Number", value=st.session_state.case_form_data["phone_number"])
            st.session_state.case_form_data["previous_mental_health_history"] = st.text_area("Previous Mental Health History", value=st.session_state.case_form_data["previous_mental_health_history"])
            st.session_state.case_form_data["follow_up_actions"] = st.text_area("Follow-Up Actions", value=st.session_state.case_form_data["follow_up_actions"])
            st.session_state.case_form_data["next_session_date"] = st.text_input("Next Session Date", value=st.session_state.case_form_data["next_session_date"])


    with tab3:
        st.header("Treatment Plan Assistant")
        
        # Initialize chat history and latest_entry if not exists
        if "treatment_chat" not in st.session_state:
            st.session_state.update({
                "treatment_chat": [
                    {"role": "assistant", "content": "Ask me to show: progress, tools, or CBT suggestions"}
                ],
                "client_data": {
                    "progress": "Client has reduced anxiety attacks from 5x/week to 2x/week",
                    "goals": "Practice grounding techniques daily, establish sleep routine",
                    "next_steps": "Schedule follow-up in 2 weeks",
                    "tools": ["Mood tracker", "Breathing exercise audio", "Sleep hygiene checklist"],
                    "cbt_techniques": [
                        "Thought record for cognitive distortions",
                        "Behavioral activation schedule",
                        "5-4-3-2-1 grounding technique"
                    ],
                    "risk_level": "Low"
                },
                "latest_entry": {
                    "progress": "medium progress",
                    "goals": "eat more apple",
                    "next_steps": "come back in 2 weeks",
                    "risk_level": "Medium",
                    "risk_notes": "Monitor for increased hopelessness"
                }
            })

        # Define a clean grid layout (x, y, width, height)
        layout = [
            # Row 1: Top-wide cards
            dashboard.Item("case_recommendations", 0, 0, 4, 4),  # Wider progress summary
            dashboard.Item("risk_alert", 4, 0, 4, 1),        # Risk alert (full width)

            # Row 1: Top-wide cards
            dashboard.Item("progress_summary", 0, 0, 4, 2),  # Wider progress summary
            dashboard.Item("progress_chart", 4, 0, 4, 2),     # Progress card (left)
            dashboard.Item("goal_timeline", 0, 2, 4, 2.5),     # Progress card (left)

            # Row 2: Middle components
            dashboard.Item("risk_chart", 3.5, 0, 4, 2),     # Progress card (left)
            dashboard.Item("trigger_log", 3.5, 2, 6, 2),   # Wider CBT suggestions
            dashboard.Item("crisis_protocols", 0, 2, 3.5, 2),   # Wider CBT suggestions
            dashboard.Item("coping_strategies", 0, 0, 3.5, 2),   # Wider CBT suggestions

            # Row 3: Bottom cards          
            dashboard.Item("missed_activity_reasons", 0, 0, 4, 2.5),     # Progress card (left)
            dashboard.Item("adherence_chart", 4, 0, 5, 2.5),     # Progress card (left)
            dashboard.Item("auto_adjustments", 4, 2, 5, 2.5),     # Progress card (left)
            dashboard.Item("next_steps", 0, 6, 4, 2.5),     # Progress card (left)
            
            dashboard.Item("cbt_suggestions", 0, 2, 4, 2),   # Wider CBT suggestions
        ]
        
        # Display chat history
        for i, message in enumerate(st.session_state.treatment_chat):
            with st.chat_message(message["role"]):
                if isinstance(message.get("content"), str):
                    # Simple text response
                    st.write(message["content"])
                elif isinstance(message.get("content"), dict):
                    # Structured response with MUI components
                    with elements(f"response_{i}"):  # Unique key for each response
                        with dashboard.Grid(layout):
                            if message["content"]["type"] == "check-in":
                                with mui.Card(key="case_recommendations", sx={"p": 2, "mb": 2, "borderLeft": "4px solid #2196F3"}):
                                    with mui.CardHeader(
                                        title="Next Case Recommendations",
                                        avatar=mui.icon.Assignment(color="primary"),
                                        action=mui.IconButton(mui.icon.Refresh)
                                    ):
                                        pass
                                    
                                    # Current case data (would come from your database)
                                    case = {
                                        "name": "Alex Chen",
                                        "age": 17,
                                        "goals": ["Reduce anxiety attacks", "Improve sleep routine"],
                                        "adherence_rate": 0.65,  # 65%
                                        "missed_activities": ["Journaling (3x)", "Medication (2x)"],
                                        "risk_level": "Medium"
                                    }
                                    
                                    with mui.CardContent():
                                        # --- Student Summary Row ---
                                        with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"mb": 2}):
                                            mui.Avatar("AC", sx={"bgcolor": "#1976D2"})
                                            with mui.Box():
                                                mui.Typography(case["name"], variant="h6")
                                                mui.Typography(
                                                    f"Age {case['age']} • {case['risk_level']} Risk • {int(case['adherence_rate']*100)}% Adherence",
                                                    variant="body2",
                                                    color="text.secondary"
                                                )
                                        
                                        # --- Recommended Adjustments ---
                                        mui.Divider(text="Suggested Adjustments", sx={"my": 2})
                                        
                                        adjustments = [
                                            {
                                                "activity": "Journaling",
                                                "current": "Daily written entries",
                                                "suggestion": "Switch to voice memos 3x/week",
                                                "reason": "Missed 3x last week due to time constraints"
                                            },
                                            {
                                                "activity": "CBT Exercises",
                                                "current": "20-minute sessions",
                                                "suggestion": "Try 5-minute 'micro-CBT' exercises",
                                                "reason": "Low energy reported on weekdays"
                                            }
                                        ]
                                        
                                        for adj in adjustments:
                                            with mui.Alert(
                                                severity="info",
                                                icon=mui.icon.LightbulbOutline(),
                                                sx={"mb": 1, "alignItems": "flex-start"}
                                            ):
                                                with mui.Stack(spacing=0.5):
                                                    mui.Typography(
                                                        f"{adj['activity']}: {adj['suggestion']}",
                                                        fontWeight="bold"
                                                    )
                                                    mui.Typography(
                                                        f"Instead of {adj['current']} • {adj['reason']}",
                                                        variant="body2"
                                                    )
                                                    with mui.Stack(direction="row", spacing=1, sx={"mt": 1}):
                                                        mui.Button(
                                                            "Apply This",
                                                            size="small",
                                                            variant="outlined",
                                                            startIcon=mui.icon.Check()
                                                        )
                                                        mui.Button(
                                                            "See Alternatives",
                                                            size="small",
                                                            startIcon=mui.icon.Search()
                                                        )
                                        
                                        # --- Quick Actions ---
                                        mui.Divider(text="Quick Actions", sx={"my": 2})
                                        with mui.Stack(direction="row", spacing=1):
                                            mui.Button(
                                                "View Full Case",
                                                variant="outlined",
                                                startIcon=mui.icon.FolderOpen()
                                            )
                                            mui.Button(
                                                "Start Session Notes",
                                                variant="contained",
                                                startIcon=mui.icon.Edit()
                                            )

                                # Risk Alert (Middle, full width)
                                with mui.Paper(
                                    key="risk_alert",
                                    sx={
                                        "mb": 2,
                                        "borderLeft": "4px solid #ff9800",
                                        "bgcolor": "#fff3e0",
                                        "p": 2,
                                        "height": "100%",
                                    }
                                ):
                                    mui.Typography("Risk Level: Medium", variant="h6")
                                    mui.Typography("Monitor for increased hopelessness per last session.")

                            elif message["content"]["type"] == "progress_summary":
                                with mui.Paper(key="progress_summary", elevation=3, sx={"p": 2, "my": 1, "borderLeft": "4px solid #1976d2"}):
                                    mui.Typography("Treatment Progress", variant="h6")
                                    mui.Divider()
                                    mui.Typography(f"Last Session: {message['content']['data']['progress']}")                                    
                                    mui.Stack(
                                        mui.Chip(label=f"Goals: {message['content']['data']['goals']}", color="primary"),
                                        mui.Chip(label=f"Next Steps: {message['content']['data']['next_steps']}", color="secondary"),
                                        spacing=1
                                    )
                                    mui.Button("VIEW FULL HISTORY", 
                                            variant="outlined", 
                                            sx={"mt": 1},
                                            onClick=lambda: st.session_state.treatment_chat.append(
                                                {"role": "assistant", "content": "Full history: [Would display expanded timeline here]"}
                                            ))

                                DATA = [
                                    { "progress": "Anxiety", "baseline": 93, "current": 61 },
                                    { "progress": "Depression", "baseline": 91, "current": 37 },
                                    { "progress": "Sleep", "baseline": 56, "current": 95 },
                                    { "progress": "Social", "baseline": 64, "current": 90 },
                                    { "progress": "Focus", "baseline": 119, "current": 94 },
                                ]

                                with mui.Card(key="progress_chart", sx={"p": 2, "my": 1, "bgcolor": "#FEFBF5"}):
                                    nivo.Radar(
                                        data=DATA,
                                        keys=[ "baseline", "current" ],
                                        indexBy="progress",
                                        valueFormat=">-.2f",
                                        margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                                        borderColor={ "from": "color" },
                                        gridLabelOffset=36,
                                        dotSize=10,
                                        dotColor={ "theme": "background" },
                                        dotBorderWidth=2,
                                        motionConfig="wobbly",
                                        legends=[
                                            {
                                                "anchor": "top-left",
                                                "direction": "column",
                                                "translateX": -50,
                                                "translateY": -40,
                                                "itemWidth": 80,
                                                "itemHeight": 20,
                                                "itemTextColor": "#999",
                                                "symbolSize": 12,
                                                "symbolShape": "circle",
                                                "effects": [
                                                    {
                                                        "on": "hover",
                                                        "style": {
                                                            "itemTextColor": "#000"
                                                        }
                                                    }
                                                ]
                                            }
                                        ],
                                        theme={
                                            "background": "#FEFBF5",
                                            "textColor": "#31333F",
                                            "tooltip": {
                                                "container": {
                                                    "background": "#FEFBF5",
                                                    "color": "#31333F",
                                                }
                                            }
                                        }
                                    )

                                with mui.Card(key="goal_timeline", sx={"p": 2, "mt": 2}):
                                    with mui.CardHeader(
                                        title="Goal Progress Timeline",
                                        avatar=mui.icon.Timeline()
                                    ):
                                        pass
                                    
                                    with mui.CardContent():
                                        with mui.List():
                                            milestones = [
                                                {"date": "Jun 5", "goal": "Start CBT", "status": "completed"},
                                                {"date": "Jun 12", "goal": "Reduce anxiety attacks", "status": "in-progress"},
                                                {"date": "Jun 20", "goal": "Establish sleep routine", "status": "pending"}
                                            ]
                                            
                                            for milestone in milestones:
                                                with mui.ListItem(key=milestone["date"], sx={"py": 1}):
                                                    mui.ListItemAvatar(
                                                        mui.Avatar(
                                                            mui.icon.Check() if milestone["status"] == "completed" else 
                                                            mui.icon.AccessTime() if milestone["status"] == "in-progress" else 
                                                            mui.icon.Event()
                                                        )
                                                    )
                                                    with mui.ListItemText(
                                                        primary=milestone["goal"],
                                                        secondary=milestone["date"],
                                                        sx={"color": "text.primary" if milestone["status"] == "in-progress" else "text.secondary"}
                                                    ):
                                                        pass
                                                    mui.Chip(
                                                        label=milestone["status"].replace("-", " ").title(),
                                                        size="small",
                                                        color={
                                                            "completed": "success",
                                                            "in-progress": "warning",
                                                            "pending": "default"
                                                        }[milestone["status"]]
                                                    )

                            elif message["content"]["type"] == "risk_analysis":
                                with mui.Card(key="trigger_log", sx={"p": 2, "mt": 2, "borderLeft": "4px solid #ff5722"}):
                                    with mui.CardHeader(title="Trigger Log", avatar=mui.icon.Warning(color="error")):
                                        mui.Typography("Recent risk triggers", variant="body2", color="text.secondary")
                                    
                                    with mui.CardContent():
                                        # Editable table
                                        with mui.TableContainer():
                                            with mui.Table(size="small"):
                                                with mui.TableHead():
                                                    with mui.TableRow():
                                                        mui.TableCell("Date")
                                                        mui.TableCell("Trigger")
                                                        mui.TableCell("Severity (1-5)")
                                                        mui.TableCell("")  # Actions column
                                                
                                                with mui.TableBody():
                                                    for i, trigger in enumerate(st.session_state.trigger_log):
                                                        with mui.TableRow(key=f"trigger_{i}"):
                                                            mui.TableCell(trigger["date"])
                                                            mui.TableCell(trigger["description"])
                                                            mui.TableCell(
                                                                mui.Rating(
                                                                    value=trigger["severity"],
                                                                    size="small"
                                                                )
                                                            )
                                                            mui.TableCell(
                                                                mui.IconButton(mui.icon.Delete, onClick=lambda idx=i: 
                                                                    st.session_state.trigger_log.pop(idx)
                                                                )
                                                            )
                                        
                                                            # Add new trigger
                                                            with mui.Stack(direction="row", spacing=2, sx={"mt": 2}):
                                                                trigger_input = mui.TextField(
                                                                    size="small",
                                                                    placeholder="New trigger (e.g. 'Argument with family')",
                                                                    sx={"flexGrow": 1}
                                                                )
                                                                mui.Button(
                                                                    "Add",
                                                                    variant="outlined",
                                                                    startIcon=mui.icon.Add(),
                                                                    onClick=lambda: st.session_state.trigger_log.append({
                                                                        "date": datetime.now().strftime("%m/%d"),
                                                                        "description": trigger_input.value,
                                                                        "severity": 3
                                                                    })
                                                                )

                                with mui.Card(key="crisis_protocols", sx={"p": 2, "mt": 2}):
                                    with mui.CardHeader(
                                        title="Crisis Protocols",
                                        avatar=mui.icon.Emergency(color="warning"),
                                        action=mui.IconButton(mui.icon.ExpandMore)
                                    ):
                                        pass
                                    
                                    with mui.CardContent():
                                        with mui.List(dense=True):
                                            # Dynamic based on risk level
                                            if st.session_state.risk_level == "High":
                                                protocols = [
                                                    ("🚨 Contact emergency services", "#ffebee"),
                                                    ("📞 Call designated responder: Jordan (555-1234)", "#fff3e0"),
                                                    ("🏠 Initiate safe space protocol", "#e8f5e9")
                                                ]
                                            else:
                                                protocols = [
                                                    ("📱 Schedule crisis check-in call", "#e3f2fd"),
                                                    ("✍️ Complete safety plan worksheet", "#f3e5f5"),
                                                    ("🌿 Use grounding techniques", "#e8f5e9")
                                                ]
                                            
                                            for protocol, bgcolor in protocols:
                                                with mui.ListItem(
                                                    sx={"borderLeft": f"4px solid {bgcolor}", "mb": 1}
                                                ):
                                                    mui.ListItemText(primary=protocol)
                                                    mui.Checkbox(edge="end")

                                with mui.Card(key="coping_strategies", sx={"p": 2, "mt": 2, "border": "1px solid #e0e0e0"}):
                                    with mui.CardHeader(
                                        title="Recommended Coping Strategies",
                                        subheader="Adapted to current risk profile",
                                        avatar=mui.icon.Psychology(color="primary")
                                    ):
                                        pass
                                    
                                    with mui.CardContent():
                                        mui.Chip(
                                            label="High Anxiety" if st.session_state.risk_level == "High" else "General",
                                            color="warning" if st.session_state.risk_level == "High" else "default",
                                            size="small"
                                        )
                                        for strategy in st.session_state.coping_strategies:
                                            with mui.Paper(sx={"p": 2, "mb": 2, "bgcolor": "#f5f5f5"}):
                                                mui.Typography(strategy["name"], fontWeight="bold")
                                                mui.Typography(strategy["instructions"], variant="body2")
                                                # Buttons here...
                                                            
                                                # Action buttons
                                                with mui.Stack(direction="row", spacing=1):
                                                    mui.Button(
                                                        "Mark as Tried",
                                                        variant="outlined",
                                                        size="small",
                                                        startIcon=mui.icon.Check(),
                                                        onClick=lambda i=i: mark_strategy_tried(i),
                                                        sx={"mr": 1}
                                                    )
                                                    mui.Button(
                                                        "Not Helpful",
                                                        variant="outlined",
                                                        size="small",
                                                        color="error",
                                                        startIcon=mui.icon.Close(),
                                                        onClick=lambda i=i: flag_unhelpful_strategy(i)
                                                    )

                                DATA = [
                                    { "risk": "Suicidal Ideation", "baseline": 83, "current": 51, "thresholds": 41 },
                                    { "risk": "Self-Harm Urges", "baseline": 61, "current": 47, "thresholds": 41 },
                                    { "risk": "Substance Use", "baseline": 76, "current": 65, "thresholds": 41 },
                                    { "risk": "Impulsivity", "baseline": 54, "current": 30, "thresholds": 41 },
                                    { "risk": "Social Isolation", "baseline": 49, "current": 24, "thresholds": 41 },
                                ]

                                with mui.Card(key="risk_chart", sx={"p": 2, "my": 1, "bgcolor": "#FEFBF5"}):
                                    nivo.Radar(
                                        data=DATA,
                                        keys=[ "baseline", "thresholds", "current" ],
                                        indexBy="risk",
                                        valueFormat=">-.2f",
                                        margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                                        borderColor={ "from": "color" },
                                        gridLabelOffset=36,
                                        dotSize=10,
                                        dotColor={ "theme": "background" },
                                        dotBorderWidth=2,
                                        motionConfig="wobbly",
                                        legends=[
                                            {
                                                "anchor": "top-left",
                                                "direction": "column",
                                                "translateX": -50,
                                                "translateY": -40,
                                                "itemWidth": 80,
                                                "itemHeight": 20,
                                                "itemTextColor": "#999",
                                                "symbolSize": 12,
                                                "symbolShape": "circle",
                                                "effects": [
                                                    {
                                                        "on": "hover",
                                                        "style": {
                                                            "itemTextColor": "#000"
                                                        }
                                                    }
                                                ]
                                            }
                                        ],
                                        theme={
                                            "background": "#FEFBF5",
                                            "textColor": "#31333F",
                                            "tooltip": {
                                                "container": {
                                                    "background": "#FEFBF5",
                                                    "color": "#31333F",
                                                }
                                            }
                                        }
                                    )

                            elif message["content"]["type"] == "tools_card":
                                with mui.Card(key="missed_activity_reasons", sx={"p": 2, "mt": 2, "borderLeft": "4px solid #ff9800"}):
                                    with mui.CardHeader(
                                        title="Missed Activity Analysis",
                                        avatar=mui.icon.Quiz(color="warning")
                                    ):
                                        mui.Typography("Last 30 days", variant="body2", color="text.secondary")
                                    
                                    with mui.CardContent():
                                        with mui.List(dense=True):
                                            for activity in ["CBT Exercises", "Journaling", "Medication"]:
                                                with mui.ListItem(key=activity):
                                                    with mui.ListItemText(
                                                        primary=activity,
                                                        secondary=f"Missed {random.randint(1,5)} times"
                                                    ):
                                                        pass
                                                    with mui.TextField(
                                                        size="small",
                                                        placeholder="Reason...",
                                                        sx={"width": "200px"},
                                                        onChange=lambda e, a=activity: (
                                                            st.session_state.missed_reasons.update({a: e.target.value}),
                                                            st.rerun()
                                                        )
                                                    ):
                                                        pass

                                with mui.Card(key="auto_adjustments", sx={"p": 2, "mt": 2, "bgcolor": "#e8f5e9"}):
                                    with mui.CardHeader(
                                        title="Recommended Adjustments",
                                        action=mui.IconButton(mui.icon.AutoAwesome)
                                    ):
                                        pass
                                    
                                    with mui.CardContent():
                                        suggestions = {
                                            "CBT Exercises": "Try shorter 5-minute versions",
                                            "Journaling": "Switch to voice memos on busy days",
                                            "Medication": "Set phone reminders at 8AM/8PM"
                                        }
                                        
                                        for activity, suggestion in suggestions.items():
                                            with mui.Alert(
                                                severity="info",
                                                sx={"mb": 1},
                                                action=mui.Button("Apply", size="small")
                                            ):
                                                mui.Typography(f"{activity}: {suggestion}", variant="body2")

                                with mui.Card(key="next_steps", sx={"p": 2, "mt": 2, "borderLeft": "4px solid #1976d2"}):
                                    with mui.CardHeader(
                                        title="Immediate Next Steps",
                                        subheader="Based on adherence trends"
                                    ):
                                        pass
                                    
                                    with mui.CardContent():
                                        with mui.List(dense=False):
                                            steps = [
                                                ("Schedule medication review", "high-priority", mui.icon.Warning),
                                                ("Download CBT mobile app", "medium-priority", mui.icon.PhoneIphone),
                                                ("Journal prompt worksheets", "low-priority", mui.icon.Description)
                                            ]
                                            
                                            for text, priority, icon in steps:
                                                with mui.ListItem(key=text, secondaryAction=mui.Checkbox(edge="end")):
                                                    mui.ListItemAvatar(icon())
                                                    with mui.ListItemText(
                                                        primary=text,
                                                        secondary=priority.replace("-", " ").title(),
                                                        sx={
                                                            "& .MuiListItemText-secondary": {
                                                                "color": {
                                                                    "high-priority": "#f44336",
                                                                    "medium-priority": "#ff9800",
                                                                    "low-priority": "#4caf50"
                                                                }[priority]
                                                            }
                                                        }
                                                    ):
                                                        pass

                                DATA = [
                                    { "activities": "CBT Exercises", "target": 53, "week1": 31, "week2": 41 },
                                    { "activities": "Medication", "target": 51, "week1": 47, "week2": 51 },
                                    { "activities": "Journaling", "target": 56, "week1": 25, "week2": 31 },
                                    { "activities": "Sleep Routine", "target": 54, "week1": 30, "week2": 41 },
                                    { "activities": "Session Attendance", "target": 59, "week1": 54, "week2": 41 },
                                ]

                                with mui.Card(key="adherence_chart", sx={"p": 2, "my": 1, "bgcolor": "#FEFBF5"}):
                                    nivo.Radar(
                                        data=DATA,
                                        keys=[ "target", "week1", "week2" ],
                                        indexBy="activities",
                                        valueFormat=">-.2f",
                                        margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                                        borderColor={ "from": "color" },
                                        gridLabelOffset=36,
                                        dotSize=10,
                                        dotColor={ "theme": "background" },
                                        dotBorderWidth=2,
                                        motionConfig="wobbly",
                                        legends=[
                                            {
                                                "anchor": "top-left",
                                                "direction": "column",
                                                "translateX": -50,
                                                "translateY": -40,
                                                "itemWidth": 80,
                                                "itemHeight": 20,
                                                "itemTextColor": "#999",
                                                "symbolSize": 12,
                                                "symbolShape": "circle",
                                                "effects": [
                                                    {
                                                        "on": "hover",
                                                        "style": {
                                                            "itemTextColor": "#000"
                                                        }
                                                    }
                                                ]
                                            }
                                        ],
                                        theme={
                                            "background": "#FEFBF5",
                                            "textColor": "#31333F",
                                            "tooltip": {
                                                "container": {
                                                    "background": "#FEFBF5",
                                                    "color": "#31333F",
                                                }
                                            }
                                        }
                                    )

                            elif message["content"]["type"] == "cbt_suggestions":
                                with mui.Card(key="cbt_suggestions", sx={"p": 2, "my": 1, "borderLeft": "4px solid #4caf50"}):
                                    mui.CardHeader(title="CBT Techniques", 
                                                avatar=mui.icon.Psychology())
                                    mui.Divider()
                                    mui.CardContent(
                                        mui.List(
                                            *[mui.ListItem(
                                                mui.Stack(
                                                    mui.icon.CheckCircle(color="success"),
                                                    mui.Typography(tech),
                                                    direction="row", spacing=2
                                                )
                                            ) for tech in st.session_state.client_data["cbt_techniques"]]
                                        )
                                    )

        # User input handling
        if prompt := st.chat_input("Ask about treatment plans..."):
            # Add user message to chat
            st.session_state.treatment_chat.append({"role": "user", "content": prompt})
            
            response_map = {
                "check-in": {"type": "check-in", "trigger": ["check-in", "checking-in"]},
                "tools": {"type": "tools_card", "trigger": ["tool", "recommend", "material"]},
                "cbt": {"type": "cbt_suggestions", "trigger": ["cbt", "technique", "exercise", "activity"]}
            }

            # Process query and generate appropriate response
            response = None
            prompt_lower = prompt.lower()
            
            for key, config in response_map.items():
                if any(trigger in prompt_lower for trigger in config["trigger"]):
                    response = {"type": config["type"]}
                    break
            
            if any(w in prompt_lower for w in ["full case", "progress", "summary"]):
                response = {
                    "type": "progress_summary",
                    "data": {
                        "progress": st.session_state.latest_entry["progress"],
                        "goals": st.session_state.latest_entry["goals"],
                        "next_steps": st.session_state.latest_entry["next_steps"]
                    }
                }
            elif any(w in prompt_lower for w in ["risk", "urgent", "emergency"]):
                response = {
                    "type": "risk_analysis",
                    "data": {
                        "level": st.session_state.latest_entry["risk_level"],
                        "notes": st.session_state.latest_entry["risk_notes"]
                    }
                }                
            elif any(w in prompt_lower for w in ["add", "update", "treatment", "plan"]):
                try:
                    # Parse and update treatment plan
                    updated_entry = parse_and_update_treatment_plan(
                        prompt_lower, 
                        st.session_state.longitudinal_data
                    )
                    
                    # Update the displayed summary
                    st.session_state.latest_entry = {
                        "progress": updated_entry.get("progress", ""),
                        "goals": updated_entry.get("goals", ""),
                        "next_steps": updated_entry.get("next_steps", ""),
                        "risk_level": st.session_state.latest_entry.get("risk_level", "Medium"),
                        "risk_notes": "Monitor for increased hopelessness"
                    }
                    
                    # Add to chat
                    st.session_state.treatment_chat.append({
                        "role": "assistant",
                        "content": {
                            "type": "progress_summary",
                            "data": st.session_state.latest_entry
                        }
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Update failed: {str(e)}")
            
            if not response:
                response = """I can help with:

                🔹 **Treatment Progress**  
                - "Show progress summary"  
                - "Latest updates"  

                🔹 **Therapeutic Tools**  
                - "What tools are available?"  
                - "Show resources"  

                🔹 **CBT Techniques**  
                - "Suggest CBT exercises"  
                - "Cognitive techniques"  

                🔹 **Risk Assessment**  
                - "Current risk level"  
                - "Any concerns?"  

                Try being specific like: "Show me the client's recent progress" or "What CBT techniques would help with anxiety?"
                """

            # Add response to chat
            st.session_state.treatment_chat.append({
                "role": "assistant", 
                "content": response if isinstance(response, dict) else response
            })
            
            # Force UI update
            st.rerun()
