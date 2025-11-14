from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from model import build_llm


class ConversationState(TypedDict):
    messages: list[HumanMessage | AIMessage]


llm = build_llm()


def chat_node(state: ConversationState) -> ConversationState:
    response = llm.invoke(state["messages"])
    new_messages = state["messages"] + [response]
    return {"messages": new_messages}


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    return graph.compile()