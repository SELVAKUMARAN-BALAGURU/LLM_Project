from typing import TypedDict, Dict, Any, Optional
import pandas as pd
from langgraph.graph import StateGraph, START, END

from tools.profiler import load_dataset, generate_profile
from tools.transformer import apply_cleaning_plan
from agents.planner_agent import generate_cleaning_plan

class AgentState(TypedDict):
    file_path: str
    df: Optional[pd.DataFrame]
    profile: Optional[Dict[str, Any]]
    cleaning_plan: Optional[Dict[str, Any]]
    approval_status: str # "approved", "rejected", "pending"
    feedback: Optional[str]
    cleaned_df: Optional[pd.DataFrame]
    summary: Optional[Dict[str, Any]]

def load_data_node(state: AgentState):
    print("\n--- [Agent 0] Loading Dataset ---")
    df = load_dataset(state["file_path"])
    return {"df": df}

def schema_understanding_node(state: AgentState):
    print("\n--- [Agent 1] Schema Understanding ---")
    profile = generate_profile(state["df"])
    print(f"Dataset Health Score: {profile['dataset_health_score']}")
    return {"profile": profile}

def cleaning_strategy_node(state: AgentState):
    print("\n--- [Agent 2] Cleaning Strategy Agent ---")
    # if there is feedback from user, pass it to planner
    # Provide visual queue that the LLM is working
    print("Analyzing profile and generating strategy via local LLaMA model...")
    plan = generate_cleaning_plan(state["profile"], state.get("feedback"))
    return {"cleaning_plan": plan, "approval_status": "pending"}

def human_approval_node(state: AgentState):
    print("\n--- [Agent 3] Human Approval ---")
    import json
    print("\nProposed Cleaning Plan:")
    print(json.dumps(state["cleaning_plan"], indent=2))
    print("\nPlease review the plan.")
    choice = input("Approve? (y/n) or type modification instructions: ").strip()
    
    if choice.lower() in ['y', 'yes', '']:
        return {"approval_status": "approved", "feedback": None}
    else:
        print(f"Re-evaluating plan with feedback: '{choice}'")
        return {"approval_status": "rejected", "feedback": choice}

def should_execute_cleaning(state: AgentState):
    if state["approval_status"] == "approved":
        return "execute"
    else:
        return "replan"

def cleaning_execution_node(state: AgentState):
    print("\n--- [Agent 4] Data Cleaning Execution ---")
    cleaned_df, summary = apply_cleaning_plan(state["df"], state["cleaning_plan"])
    print("\nTransformation Summary:")
    print(summary)
    cleaned_df.to_csv("cleaned_output.csv", index=False)
    print("\nCleaned dataset saved as cleaned_output.csv")
    return {"cleaned_df": cleaned_df, "summary": summary}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("schema_understanding", schema_understanding_node)
    workflow.add_node("cleaning_strategy", cleaning_strategy_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("cleaning_execution", cleaning_execution_node)
    
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "schema_understanding")
    workflow.add_edge("schema_understanding", "cleaning_strategy")
    workflow.add_edge("cleaning_strategy", "human_approval")
    
    workflow.add_conditional_edges(
        "human_approval",
        should_execute_cleaning,
        {
            "execute": "cleaning_execution",
            "replan": "cleaning_strategy"
        }
    )
    workflow.add_edge("cleaning_execution", END)
    
    return workflow.compile()
