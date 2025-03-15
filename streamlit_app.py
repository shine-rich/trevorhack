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

if 'autopopulation' not in st.session_state:
    st.session_state.autopopulation = False

if 'should_prompt_agent' not in st.session_state:
    st.session_state.should_prompt_agent = False

if 'mood' not in st.session_state:
    st.session_state.mood = None

if 'suggested_reply1' not in st.session_state:
    st.session_state.suggested_reply1 = ""

if 'longitudinal_data' not in st.session_state:
    st.session_state.longitudinal_data = []  # For storing historical data

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

st.set_page_config(page_title="TrevorText, powered by LlamaIndex", page_icon="💬", layout="wide")

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
    st.sidebar.json(data)

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
    return f"""You are a helpful mental health assistant chatbot, helping to train a junior counselor by providing suggestions on responses to client chat inputs. What would you recommend that the consider could say if someone says or asks '{user_input}'? Keep your responses limited to 4-5 lines; do not ask if the client needs more resources. If the case is not high risk, check for resources to help inform your response. If you need to send an email to share therapist contacts, call that action.
    """

def send_chat_message():
    if st.session_state.custom_chat_input:  # Check if the input is not empty
        st.session_state.messages.append({"role": "user", "content": st.session_state.custom_chat_input})
        st.session_state.custom_chat_input = ""

def set_custom_chat_input():
    st.session_state.custom_chat_input = st.session_state.suggested_reply1

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

if st.session_state.openai_apikey:
    search_tool = FunctionTool.from_defaults(fn=search_for_therapists)
    escalate_tool = FunctionTool.from_defaults(fn=escalate)
    resource_tool = FunctionTool.from_defaults(fn=get_resource_for_response)
    index = load_data()
    agent = build_agent()

    if "query_engine" not in st.session_state.keys():
        st.session_state.query_engine = index.as_query_engine(similarity_top_k=3, verbose=True)

    st.sidebar.write("Longitudinal Data:")
    st.sidebar.json(st.session_state.longitudinal_data)
    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Crisis Hotline Chat", "Case Form", "Treatment Plan Dashboard", "Student Portal"])

    # Define valid options for Suicidal Ideation
    suicidal_ideation_options = ["No", "Passive", "Active with Plan", "Active without Plan"]
    risk_level_options = ["Not Suicidal", "Low Risk", "Medium Risk", "High Risk", "Imminent Risk"]

    # crisis intervention room tab
    with tab1:
        col_a1, col_a2 = st.columns([0.5, 0.5], gap="small")
        with col_a1:
            if "chat_engine" not in st.session_state.keys():
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"role": "user", "content": "Hi, welcome to TrevorText. Can you tell me a little bit about yourself?"}]
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            if st.session_state.messages[-1]["role"] != "assistant":
                # time.sleep(2)
                with st.chat_message("assistant"):
                    if st.session_state.script_idx < len(client_script):
                        response = client_script[st.session_state.script_idx][2:]
                        st.session_state.script_idx += 2
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)  # Add response to message history
                        st.session_state.should_prompt_agent = True
                        
                    elif not st.session_state.autopopulation:
                        st.info("Contact has left the chat")
                        with st.status("Autopopulating form...", expanded=True) as status:
                            st.write("Downloading chat history...")
                            chathistory = ""
                            for item in st.session_state.messages:
                                chathistory += item['content'] + '\n'
                            # time.sleep(1) 
                            st.write("Analyzing chat...")

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
                            if risk_level_response in suicidal_ideation_options:
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

                            st.write("Populating form...")
                            st.session_state.autopopulation = True
                            save_to_longitudinal_database({
                                "timestamp": time.time(),
                                "name": st.session_state.case_form_data["first_name"],
                                "issue": st.session_state.case_form_data["reason_for_contact"],
                                "risk_level": st.session_state.case_form_data["risk_level"],
                                "coping_strategies": st.session_state.case_form_data["coping_strategies"],
                                "progress": "Moderate improvement noted."
                            })
            
        with col_a2:          
            st.subheader("Companion Suggestions")
            if len(st.session_state.messages) > 1:
                with st.spinner("Loading..."):
                    # time.sleep(2)
                    st.write(client_summary())

            st.subheader("Suggested Reply")
            suggested_reply = ""
            source_file_names = ['cheatsheet_empathetic_language.txt', 'cheatsheet_maintaining_rapport.txt', 'cheatsheet_risk_assessment.txt']
            if st.session_state.messages[-1]["role"] == "assistant": 
                if st.session_state.should_prompt_agent:
                    try: 
                        response = agent.chat(get_modified_prompt(st.session_state.messages[-1]["content"]))
                    except:
                        response = client_script[st.session_state.script_idx-1][2:]
                    suggested_reply = str(response)
                    suggested_reply = suggested_reply.split('"')[1] if '"' in suggested_reply else suggested_reply
                    st.session_state.suggested_reply1 = suggested_reply  # Store the suggested reply in the session state
                    st.info(suggested_reply)
                    source_file_names = get_counselor_resources(response)
                    st.session_state.should_prompt_agent = False

            # Add a button to populate the custom input field with the suggested reply
            if st.button("Use Suggested Reply", on_click=set_custom_chat_input):
                pass
            st.subheader("Sources")
            source_links = []
            base_link = "https://github.com/zrizvi93/trevorhack/tree/main/data/{}"
            for file in source_file_names:
                source_links.append(base_link.format(file))
            i = 0
            sources_row = st.columns(len(source_links))
            for col in sources_row:
                with col.container(height=50):
                    st.markdown(f'<a href="{source_links[i]}" target="_blank">"{source_file_names[i]}"</a>', unsafe_allow_html=True)
                i += 1

        with col_a1:  
            if not st.session_state.autopopulation:
                # Send message in Chat
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
            
            # Use the validated value in the selectbox
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

    # Treatment Plan Dashboard Tab
    with tab3:
        st.header("Treatment Plan Dashboard")
        try:
            st.subheader(f"Student: {st.session_state.case_form_data['first_name']}")  # Dynamically display student name
            st.write(f"**Risk Level:** {st.session_state.case_form_data['risk_level']}")  # Dynamically display risk level
        except KeyError:
            pass

        # Input fields for updating the treatment plan
        st.write("**Update Treatment Plan**")

        # Update Current Goals
        st.write("**Current Goals:**")
        current_goals = st.text_area("Enter or update current goals:", value=st.session_state.longitudinal_data[-1].get("goals") if st.session_state.longitudinal_data else "")
        
        # Update Prescribed Coping Strategies
        st.write("**Prescribed Coping Strategies:**")
        coping_strategies = st.text_area("Enter or update coping strategies:", value=st.session_state.longitudinal_data[-1].get("coping_strategies") if st.session_state.longitudinal_data else "")
        
        # Update Progress Tracker
        st.write("**Progress Tracker:**")
        progress = st.text_area("Enter or update progress:", value=st.session_state.longitudinal_data[-1].get("progress", "") if st.session_state.longitudinal_data else "")
        
        # Update Next Steps
        st.write("**Next Steps:**")
        next_steps = st.text_area("Enter or update next steps:", value=st.session_state.longitudinal_data[-1].get("next_steps") if st.session_state.longitudinal_data else "")

        # Save button for updating the treatment plan
        if st.button("Save Treatment Plan"):
            # Save updated treatment plan data to the longitudinal database
            updated_data = {
                "timestamp": time.time(),
                "goals": current_goals,
                "coping_strategies": coping_strategies,
                "progress": progress,
                "next_steps": next_steps
            }
            save_to_longitudinal_database(updated_data)
            st.success("Treatment plan updated successfully!")

        # Display current treatment plan
        st.write("**Current Treatment Plan:**")
        if st.session_state.longitudinal_data:
            latest_data = st.session_state.longitudinal_data[-1]
            st.write("**Current Goals:**")
            st.write(f"- {latest_data.get('goals', 'No goals set yet.')}")
            st.write("**Prescribed Coping Strategies:**")
            st.write(f"- {latest_data.get('coping_strategies', 'No coping strategies prescribed yet.')}")
            st.write("**Progress Tracker:**")
            st.write(f"- {latest_data.get('progress', 'No progress tracked yet.')}")
            st.write("**Next Steps:**")
            st.write(f"- {latest_data.get('next_steps', 'No next steps planned yet.')}")
        else:
            st.write("No treatment plan data available.")

    # Student Portal Tab
    with tab4:
        try:
            st.subheader(f"Welcome, {st.session_state.case_form_data['first_name']}!")  # Dynamically display student name
        except KeyError:
            pass
        st.header("Student Portal")

        # Display current progress
        st.write("**Your Progress:**")
        try:
            st.write(f"- Risk Level: {st.session_state.case_form_data['risk_level']}")
        except KeyError:
            pass

        # Display current goals
        st.write("**Current Goals:**")
        if st.session_state.longitudinal_data:
            latest_data = st.session_state.longitudinal_data[-1]
            latest_goals = f"- {latest_data.get('goals', 'No goals set yet.')}"
            st.write(latest_goals)
            
            # Make goals actionable
            if "Practice mindfulness daily".lower() in latest_goals.lower():
                st.button("Start Guided Mindfulness Session", on_click=lambda: st.write("Redirecting to mindfulness app..."))
            if "mindfulness" in latest_goals.lower() and "progress" in latest_data:
                st.write("**Insight:** You've made progress on your mindfulness goal! Keep it up!")

        # Display prescribed coping strategies
        st.write("**Prescribed Coping Strategies:**")
        if st.session_state.longitudinal_data:
            latest_data = st.session_state.longitudinal_data[-1]
            latest_strategies = f"- {latest_data.get('coping_strategies', 'No coping strategies prescribed yet.')}"
            st.write(latest_strategies)
            
            # Make coping strategies actionable
            if "Take a walk".lower() in latest_strategies.lower():
                st.button("Find Walking Routes", on_click=lambda: st.write("Redirecting to maps..."))

        # Mood-based suggestions
        if st.session_state.mood == "Stressed":
            st.write("**Suggested Coping Strategy:** Try deep breathing or journaling.")
            st.button("Start Deep Breathing Exercise", on_click=lambda: st.write("Redirecting to deep breathing exercise..."))
        elif st.session_state.mood == "Overwhelmed":
            st.write("**Suggested Coping Strategy:** Prioritize tasks and take breaks.")
            st.button("View Time Management Tips", on_click=lambda: st.write("Redirecting to time management tips..."))

        # Display progress tracker
        st.write("**Progress Tracker:**")
        if st.session_state.longitudinal_data:
            latest_data = st.session_state.longitudinal_data[-1]
            st.write(f"- {latest_data.get('progress', 'No progress tracked yet.')}")

        # Display next steps
        st.write("**Next Steps:**")
        if st.session_state.longitudinal_data:
            latest_data = st.session_state.longitudinal_data[-1]
            latest_next_steps = f"- {latest_data.get('next_steps', 'No next steps planned yet.')}"
            st.write(latest_next_steps)

        # Daily Check-In
        st.write("**Daily Check-In:**")
        mood_selectbox()
        feedback = st.text_area("Any feedback on your coping strategies?")
        if st.button("Submit Check-In"):
            # Save check-in data to longitudinal database
            check_in_data = {
                "timestamp": time.time(),
                "mood": st.session_state.mood,
                "feedback": feedback
            }
            save_to_longitudinal_database(check_in_data)
            st.success("Check-in submitted successfully!")

        # Recent Feedback
        st.write("**Recent Feedback:**")
        if st.session_state.longitudinal_data:
            for entry in st.session_state.longitudinal_data:
                if "feedback" in entry:
                    st.write(f"- {entry['feedback']}")
        else:
            st.write("- No feedback submitted yet.")

        # Resources
        st.subheader("Resources")
        st.markdown("[Mindfulness Exercises](https://example.com)")
        st.markdown("[Time Management Tips](https://example.com)")

        # Upcoming Sessions
        st.write("**Upcoming Sessions:**")
        if st.session_state.longitudinal_data:
            for entry in st.session_state.longitudinal_data:
                if "next_session_date" in entry:
                    st.write(f"- Next Counseling Session: {entry['next_session_date']}")
        else:
            st.write("- No upcoming sessions scheduled yet.")
