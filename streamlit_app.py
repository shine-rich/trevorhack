import streamlit as st
import openai
import time
import helpers
import emails
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

if 'should_populate_form_using_ai' not in st.session_state:
    st.session_state.should_populate_form_using_ai = False

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

    # Treatment Plans & Feedback Tab (Consolidated tab3)
    with tab3:
        st.header("Treatment Plans & Feedback")
        
        # Sort longitudinal_data by timestamp in descending order
        sorted_data = sorted(st.session_state.longitudinal_data, key=lambda x: x.get("timestamp", 0), reverse=True)

        if sorted_data:
            for entry in sorted_data:
                st.subheader(f"{time.strftime('%e %b %Y %H:%M:%S%p', time.localtime(entry['timestamp']))}")
                if "goals" in entry:
                    st.write(f"**Goals:** {entry['goals']}")
                if "coping_strategies" in entry:
                    st.write(f"**Coping Strategies:** {entry['coping_strategies']}")
                if "progress" in entry:
                    st.write(f"**Progress:** {entry['progress']}")
                if "feedback" in entry:
                    st.write(f"**Feedback:** {entry['feedback']}")
                if "next_steps" in entry:
                    st.write(f"**Next Steps:** {entry['next_steps']}")
                st.markdown("---")
        else:
            st.write("No treatment plans or feedback available.")