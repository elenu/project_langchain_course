from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT
from langgraph.checkpoint.memory import InMemorySaver


# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph workflow.  
class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str] # Current user input
    messages: Annotated[List[BaseMessage], add_messages] # Conversation messages

    # Intent and routing
    intent: Optional[UserIntent] # Classified user intent
    next_step: str # Next node to execute in the graph

    # Memory and context
    conversation_summary: str # Summary of recent conversation
    active_documents: Optional[List[str]] # Document IDs currently discussed

    # Current task state
    current_response: Optional[Dict[str, Any]] # Response being built
    tools_used: List[str] # Tools used in current turn

    # Session management
    session_id: Optional[str] # Session management
    user_id: Optional[str] # User management

    # TODO: Modify actions_taken to use an operator.add reducer
    actions_taken: Annotated[List[str], operator.add] # List of agent nodes executed


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used


# TODO: Implement the classify_intent function.
# This function should classify the user's intent and set the next step in the workflow.
# The `classify_intent` function is the first node in the graph. 
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # TODO Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)

    # TODO Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt().format(
        user_input=state["user_input"],
        conversation_history=history
    )

    next_step = "qa"

    # TODO: Add conditional logic to set next_step based on intent
    # Invoke the structured LLM and get a `UserIntent` model back.
    intent_response = structured_llm.invoke(prompt)

    # The structured response uses the `intent_type` field (see schemas.UserIntent).
    # Support both model-like and dict-like responses for robustness.
    if hasattr(intent_response, "intent_type"):
        intent_type = getattr(intent_response, "intent_type")
    else:
        intent_type = intent_response.get("intent_type")

    # Return short routing keys that match the conditional router below
    if intent_type == "qa":
        next_step = "qa"
    elif intent_type == "summarization":
        next_step = "summarization"
    elif intent_type == "calculation":
        next_step = "calculation"
    else:
        next_step = "qa"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent_response,
        "next_step": next_step
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "user_input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the summarization_agent function. Refer to README.md Task 2.3
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "user_input": state["user_input"],
        "chat_history": state.get("messages",[]),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "user_input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    # Add all the nodes to the workflow
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)
    workflow.set_entry_point("classify_intent")

    def should_continue(state: AgentState) -> str:
        return state.get("next_step", "end")

    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa": "qa_agent",
            "summarization": "summarization_agent",
            "calculation": "calculation_agent",
            "end": END
        }
    )

    # Connect agents to update_memory and then to END
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=InMemorySaver())


# TODO: Finish implementing the update_memory function. Refer to README.md Task 2.4
# The state flows through nodes and gets updated at each step. Key principles:
# - Always return the updated state from node functions
# - Use the state to pass information between nodes
# - The state persists conversation context and intermediate results
def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    # Retrieve the LLM from config
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })
    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    def should_continue(state: AgentState) -> str:
        """Router function"""
        return state.get("next_step", "end")

    # (create_workflow is defined at module level)

    response = structured_llm.invoke(prompt_with_history)
    # UpdateMemoryResponse defines `summary` and `document_ids`
    # Support both model-like and dict-like responses
    if hasattr(response, "summary"):
        summary = getattr(response, "summary")
    else:
        summary = response.get("summary", "")

    if hasattr(response, "document_ids"):
        document_ids = getattr(response, "document_ids")
    else:
        document_ids = response.get("document_ids", [])

    return {
        "conversation_summary": summary,
        "active_documents": document_ids,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }
  