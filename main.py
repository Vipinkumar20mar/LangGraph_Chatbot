import uuid
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    AnyMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="LangGraph Memory Chatbot",
    page_icon="🤖",
)

st.title("🤖 LangGraph Memory Chatbot")


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# STATE
# -----------------------------

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


# -----------------------------
# AGENT NODE
# -----------------------------

def llm_agent(state: MessageState):

    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "Use conversation memory properly."
            )
        )
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -----------------------------
# BUILD GRAPH
# -----------------------------

graph_builder = StateGraph(MessageState)

graph_builder.add_node(
    "llm_agent",
    llm_agent
)

graph_builder.add_edge(
    START,
    "llm_agent"
)

graph_builder.add_edge(
    "llm_agent",
    END
)


# -----------------------------
# SESSION THREAD
# -----------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# -----------------------------
# DATABASE
# -----------------------------

DATABASE_URL = os.getenv("DATABASE_URL")


# -----------------------------
# CHATBOT
# -----------------------------

with PostgresSaver.from_conn_string(
    DATABASE_URL
) as checkpointer:

    checkpointer.setup()

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    # Load old messages
    try:

        snapshot = graph.get_state(config)

        if snapshot.values:

            for msg in snapshot.values["messages"]:

                if isinstance(msg, HumanMessage):

                    with st.chat_message("user"):
                        st.markdown(msg.content)

                elif isinstance(msg, AIMessage):

                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

    except Exception:
        pass

    # User Input
    prompt = st.chat_input(
        "Ask anything..."
    )

    if prompt:

        with st.chat_message("user"):
            st.markdown(prompt)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=prompt
                    )
                ],
                "llm_calls": 0,
            },
            config=config,
        )

        answer = result["messages"][-1].content

        with st.chat_message("assistant"):
            st.markdown(answer)