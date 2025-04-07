import streamlit as st
from streamlit.web import cli as stcli
from streamlit_elements import elements, mui, html, dashboard, nivo
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch
import time
import sys
import helpers
from session_state import get_session

state = get_session()

st.set_page_config(page_title="", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        /* Target the chat input inside columns */
        div[data-testid="column"] div[data-testid="stChatInput"] {
            width: 100% !important;
        }
        /* Ensure the input box expands to fill the column width */
        div[data-testid="column"] div[data-testid="stChatInput"] input {
            width: 100% !important;
        }
        .stChatMessage {padding: 12px 16px; border-radius: 18px}
        .suggest-button {
            margin-top: 10px;
            width: 100%;
        }
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        .streaming-cursor {
            display: inline-block;
            width: 0.5em;
            height: 1em;
            background-color: #555;
            animation: blink 1s infinite;
            vertical-align: middle;
            margin-left: 2px;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Define valid options for Suicidal Ideation
suicidal_ideation_options = ["No", "Passive", "Active with Plan", "Active without Plan"]
risk_level_options = ["Not Suicidal", "Low Risk", "Medium Risk", "High Risk", "Imminent Risk"]

def generate_assistant_response(assistant_input: str) -> str:
    """Generate and return assistant response, handle session end"""
    llm = get_llm()

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        for response in llm.stream_complete(f"Respond directly to: {assistant_input}"):
            full_response += response.delta
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    return full_response

# Pure conversation mode
@st.cache_resource
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

def get_form_value_from_convo(convo, form_value) -> str:
    return f"""You are a helpful assistant filling out a form. Extract the person's {form_value} from the following converstation in to input into the form. {convo}"""

def get_int_value_from_convo(convo, form_value) -> str:
    return f"""You are a helpful assistant filling out a form. Extract the person's {form_value} from the following converstation in to input into the form. {convo}"""

def get_risk_value_from_convo(convo) -> str:
    return f"""You are a helpful assistant filling out a form. Reply 0 if the person does not seem at risk based on the conversation. {convo}"""

# Longitudinal Database Integration
def save_to_longitudinal_database(data):
    state.longitudinal_data.append(data)

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

@st.cache_resource(show_spinner=False)
def client_summary() -> str:
    return helpers.CLIENT_SUMMARY

def populate_form_fields_with_llm():
    """Process chat history through LLM to populate form fields"""
    llm = get_llm()

    chathistory = "\n".join([msg['content'] for msg in state.messages])

    # Process each field with appropriate function
    state.case_form_data["first_name"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "First Name")
    )
    state.case_form_data["reason_for_contact"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Reason for Contact")
    )    
    state.case_form_data["brief_summary"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Brief Summary/Narrative")
    )
    state.case_form_data["next_session_date"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Next Session Date")
    )
    state.case_form_data["gender_identity"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Gender Identity")
    )
    state.case_form_data["preferred_pronouns"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Preferred Pronouns")
    )
    state.case_form_data["coping_strategies"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Current Coping Strategies")
    )
    state.case_form_data["goals"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Goals for This Session")
    )
    state.case_form_data["progress"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Progress Toward Previous Goals")
    )
    state.case_form_data["emergency_contact_name"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Emergency Contact Name")
    )
    state.case_form_data["relationship"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Relationship")
    )
    state.case_form_data["phone_number"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Phone Number")
    )
    state.case_form_data["previous_mental_health_history"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Previous Mental Health History")
    )
    state.case_form_data["follow_up_actions"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Follow-Up Actions")
    )

    # Handle numeric fields
    try:
        state.case_form_data["age"] = int(process_form_field(
            llm, get_int_value_from_convo(chathistory, "Age")
        ))
    except ValueError:
        state.case_form_data["age"] = 0
    
    # Process other string fields
    state.case_form_data["location"] = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Location (City, State)")
    )
    
    # Process special fields with options
    suicidal_response = process_form_field(
        llm, get_form_value_from_convo(chathistory, "Suicidal Ideation")
    )
    state.case_form_data["suicidal_ideation"] = (
        suicidal_response if suicidal_response in suicidal_ideation_options 
        else "No"
    )
    
    risk_response = process_form_field(
        llm, get_risk_value_from_convo(chathistory)
    )
    state.case_form_data["risk_level"] = (
        risk_response if risk_response in risk_level_options
        else "Medium Risk"
    )
    st.success("Case form populated with AI analysis.")

def process_form_field(llm, prompt):
    """Get cleaned response from LLM for form field population"""
    response = llm.complete(prompt).text
    return clean_response(response)

def populate_case_form_with_demo_data():
    # Populate Case Form data in session state with more realistic demo data
    state.case_form_data = {
        "first_name": "Alex",
        "reason_for_contact": "Feeling overwhelmed with work and personal life balance.",
        "age": 29,
        "location": "San Francisco, CA",
        "gender_identity": "Non-binary",
        "preferred_pronouns": "They/Them",
        "suicidal_ideation": "Passive",
        "risk_level": "Medium Risk",
        "brief_summary": "Alex has been feeling overwhelmed due to increasing work pressure and personal responsibilities. They have expressed passive suicidal ideation but no specific plans or means.",
        "coping_strategies": "Meditation, journaling, and occasional walks.",
        "goals": "Develop better time management skills and find effective stress-relief techniques.",
        "progress": "Alex has started practicing mindfulness exercises and is working on setting boundaries at work.",
        "emergency_contact_name": "Jordan Smith",
        "relationship": "Friend",
        "phone_number": "555-1234",
        "previous_mental_health_history": "Previous episodes of anxiety managed with therapy.",
        "follow_up_actions": "Schedule next session, provide resources on stress management, follow up with emergency contact if necessary.",
        "next_session_date": "2025-11-15"
    }
    st.success("Case form populated with detailed demo data!")

# Function to add a message to the conversation
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def display_messages(container, perspective="user"):
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        with container:
            with st.chat_message(role):
                if perspective == "user":
                    prefix = "You" if role == "user" else "Assistant"
                else:
                    prefix = "User" if role == "user" else "You"
                st.markdown(f"**{prefix}:** {content}")

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

def generate_ai_suggestion():
    last_user_message = next(
        (msg["content"] for msg in reversed(st.session_state.messages) 
         if msg["role"] == "user"), 
        None
    )

    if not last_user_message:
        st.warning("No user message found to respond to")
        return
    
    llm = get_llm()
    st.session_state.ai_streaming = True
    prompt = get_modified_prompt(last_user_message)
    ai_response = f"I understand you're saying: '{last_user_message[:50]}...'. " \
            "Here's what I would suggest as a helpful response..."

    # Create a chat message in the assistant container
    with st.session_state.assistant_container.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Initial empty message with cursor
        message_placeholder.markdown(f"**Processing AI Suggestion...:** <span class='streaming-cursor'></span>", 
                                   unsafe_allow_html=True)
        
        for response in llm.stream_complete(prompt):
            ai_response += response.delta
            message_placeholder.markdown(
                f"**You:** {ai_response}<span class='streaming-cursor'></span>", 
                unsafe_allow_html=True
            )        

    add_message("assistant", ai_response)    
    st.session_state.ai_streaming = False
    st.toast("AI suggestion complete!", icon="✅")

def main():
    state = get_session()
    # All fields will be type-checked by your IDE
    # Add this CSS to your existing style block
    st.markdown("""
    <style>
        .stTextArea textarea {
            min-height: 100px;
            resize: vertical;
        }
        .stFormSubmitButton button {
            margin-top: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Crisis Hotline Chat", "Case Form", "Treatment Plans & Feedback"])
    
    # Crisis intervention room tab
    with tab1:
        # Create two columns for side-by-side containers
        col1, col2, col3 = st.columns(3)

        # Left container: User's perspective
        with col1:
            st.header("User View")
            user_container = st.container()
            display_messages(user_container, "user")

            # User chat input
            if user_input := st.chat_input("User message...", key="user_input"):
                add_message("user", user_input)
                st.rerun()  # Refresh UI after adding a message
        
        # Right container: Assistant's perspective
        with col2:
            st.header("Assistant View")
            assistant_container = st.container()
            st.session_state.assistant_container = assistant_container  # Store for streaming
            display_messages(assistant_container, "assistant")

            # Input and controls
            if assistant_input := st.chat_input("Assistant reply...", key="assistant_input"):
                add_message("assistant", assistant_input)
                st.rerun()
    
            if st.button("🤖 Suggest Reply", 
                        on_click=generate_ai_suggestion, 
                        use_container_width=True,
                        disabled=st.session_state.ai_streaming):
                pass

        with col3:
            st.subheader("Live Insights 🔍")
            with st.expander("Case Overview", expanded=False):
                if len(state.messages) > 1:
                    with st.spinner("Loading..."):
                        st.write(client_summary())
            with st.expander("Emotional State & Risk Analysis", expanded=False):
                display_sentiment_analysis("")  # Custom function
                display_risk_dashboard("")
            
            st.subheader("Tools & Resources 🛠️")
            with st.expander("CBT Techniques", expanded=False):
                suggest_cbt_techniques("")      # Custom function
            
            with st.expander("Quick Actions", expanded=False):
                counselor_note = st.text_input("Add Session Note")
                st.button("🚨 Escalate Case")
                st.button("📝 Save Note to Form")
            
            st.subheader("Session Themes 📌")
            display_conversation_themes("")         # Custom function
            
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

    # Case Form Tab
    with tab2:
        st.header("Case Form")

        with st.container():
            header_col1, header_col2 = st.columns([1, 1])
            with header_col1:
                if st.button("Auto-Populate Case Form with LLM"):
                    with st.status("Autopopulating form...", expanded=True) as status:
                        populate_form_fields_with_llm()
                        save_to_longitudinal_database({
                            "timestamp": time.time(),
                            "name": state.case_form_data["first_name"],
                            "issue": state.case_form_data["reason_for_contact"],
                            "risk_level": state.case_form_data["risk_level"],
                            "coping_strategies": state.case_form_data["coping_strategies"],
                            "progress": "Moderate improvement noted."
                        })
                        status.update(label="Form populated successfully!", state="complete")                
            with header_col2:
                if st.button("Auto-Populate Case Form with Demo Data"):
                    with st.status("Autopopulating form...", expanded=True) as status:
                        populate_case_form_with_demo_data()
                        save_to_longitudinal_database({
                            "timestamp": time.time(),
                            "name": state.case_form_data["first_name"],
                            "issue": state.case_form_data["reason_for_contact"],
                            "risk_level": state.case_form_data["risk_level"],
                            "coping_strategies": state.case_form_data["coping_strategies"],
                            "progress": "Moderate improvement noted."
                        })
                        status.update(label="Form populated successfully!", state="complete")

        col_b1, col_b2 = st.columns([0.5, 0.5], gap="small")

        with col_b1:
            state.case_form_data["first_name"] = st.text_input("First Name", value=state.case_form_data["first_name"])
            state.case_form_data["reason_for_contact"] = st.text_input("Reason for Contact", value=state.case_form_data["reason_for_contact"])
            state.case_form_data["age"] = st.number_input("Age", value=state.case_form_data["age"])
            state.case_form_data["location"] = st.text_input("Location (City, State)", value=state.case_form_data["location"])
            state.case_form_data["gender_identity"] = st.text_input("Gender Identity", value=state.case_form_data["gender_identity"])
            state.case_form_data["preferred_pronouns"] = st.text_input("Preferred Pronouns", value=state.case_form_data["preferred_pronouns"])
            state.case_form_data["suicidal_ideation"] = st.selectbox(
                "Suicidal Ideation",
                suicidal_ideation_options,
                index=suicidal_ideation_options.index(state.case_form_data["suicidal_ideation"])
            )
            state.case_form_data["risk_level"] = st.selectbox(
                "Risk Level",
                risk_level_options,
                index=risk_level_options.index(state.case_form_data["risk_level"])
            )
        with col_b2:
            state.case_form_data["brief_summary"] = st.text_area("Brief Summary/Narrative", value=state.case_form_data["brief_summary"])
            state.case_form_data["coping_strategies"] = st.text_area("Current Coping Strategies", value=state.case_form_data["coping_strategies"])
            state.case_form_data["goals"] = st.text_area("Goals for This Session", value=state.case_form_data["goals"])
            state.case_form_data["progress"] = st.text_area("Progress Toward Previous Goals", value=state.case_form_data["progress"])
            state.case_form_data["emergency_contact_name"] = st.text_input("Emergency Contact Name", value=state.case_form_data["emergency_contact_name"])
            state.case_form_data["relationship"] = st.text_input("Relationship", value=state.case_form_data["relationship"])
            state.case_form_data["phone_number"] = st.text_input("Phone Number", value=state.case_form_data["phone_number"])
            state.case_form_data["previous_mental_health_history"] = st.text_area("Previous Mental Health History", value=state.case_form_data["previous_mental_health_history"])
            state.case_form_data["follow_up_actions"] = st.text_area("Follow-Up Actions", value=state.case_form_data["follow_up_actions"])
            state.case_form_data["next_session_date"] = st.text_input("Next Session Date", value=state.case_form_data["next_session_date"])

    with tab3:
        st.header("Treatment Plan Assistant")
        
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


if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())        