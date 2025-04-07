"""Centralized session state management with type hints and organization"""
from typing import TypedDict, List, Dict, Optional
import streamlit as st

# Type definitions for better code completion and safety
class Message(TypedDict):
    role: str
    content: str

class CopingStrategy(TypedDict):
    name: str
    duration: str
    instructions: str
    last_used: Optional[str]
    helpful: Optional[bool]

class Case(TypedDict):
    id: int
    name: str
    priority: str
    last_session: str
    adherence_alerts: List[str]
    suggestions: List[str]

class CaseFormData(TypedDict):
    first_name: str
    reason_for_contact: str
    age: int
    # ... other fields ...

class SessionState(TypedDict):
    messages: List[Message]
    api_key: str
    dark_mode: bool
    model_settings: Dict[str, str]
    # ... all other fields ...

def init_defaults() -> None:
    """Initialize only missing session state values"""
    # Core chat defaults (only set if missing)
    core_defaults = {
        "should_suggest_reply_using_ai": False,
        "messages": [],
        "api_key": "sk-no-key-required",
        "dark_mode": False,
        "model_settings": {},
        "script_idx": 1,
        "custom_chat_input": "",
        "mood": None,
        "suggested_reply": "",
        "longitudinal_data": [],
        "longitudinal_data_history": [],
        "should_display_latest_treatment_plan": True,
        "ai_streaming": False
    }

    # Treatment plan data defaults
    treatment_plan_defaults = {
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
    }

    # Clinical data defaults
    clinical_defaults = {
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
        "risk_level": "Medium",
        "missed_reasons": {},
        "adherence_data": {
            "CBT": {"target": 5, "actual": 3},
            "Medication": {"target": 7, "actual": 4}
        }
    }

    # Case management defaults
    case_defaults = {
        "current_case": 0,
        "case_queue": [
            {
                "id": 101,
                "name": "Alex Chen",
                "priority": "High",
                "last_session": "2023-06-15",
                "adherence_alerts": ["Journaling", "Medication"],
                "suggestions": []
            }
        ],
        "case_form_data": {
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
    }

    # Initialize only missing keys
    for key, value in {**core_defaults, **clinical_defaults, **case_defaults, **treatment_plan_defaults}.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_session() -> SessionState:
    """Type-safe accessor that preserves existing values"""
    init_defaults()  # Only initializes missing values
    return st.session_state
