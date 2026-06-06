import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AnyMessage,AIMessage,SystemMessage,HumanMessage
from langgraph.graph.message import add_messages
load_dotenv()

Api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# state class
class MessageState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    llm_calls:int

# agent
def llm_agent(state:MessageState):
    print("Agent is responding...")
    query=state["messages"]
    response=llm.invoke([
        SystemMessage(content="you are helpful AI assistant  and use conversation memory properly and answer based on previous messages")

    ] + query)

    return {
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }

# start graph build
app=StateGraph(MessageState) 

app.add_node("llm_agent",llm_agent)
app.add_edge(START,"llm_agent")
app.add_edge("llm_agent",END)
memory=InMemorySaver()

builder=app.compile(checkpointer=memory)
config={
    "configurable":{
        "thread_id":"user_id"
    }
}

builder.invoke({
    "messages":[HumanMessage(content="My Name is Vipin?")],
    "llm_calls":0

},config=config)
result=builder.invoke({
    "messages":[HumanMessage(content="What is My Name?")],
    "llm_calls":0

},config=config)
print("\n AI response 1:")
print(result["messages"][-1].content)


