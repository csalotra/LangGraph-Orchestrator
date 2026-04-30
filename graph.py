from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from state import AgentState
from assistant import assistant
from tools import ALL_TOOLS


# -------------------------
# Graph Builder
# -------------------------
builder = StateGraph(AgentState)

# Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(ALL_TOOLS))

# Entry point
builder.add_edge(START, "assistant")

# Conditional routing (auto-handled)
builder.add_conditional_edges(
    "assistant",
    tools_condition
)

# Loop back after tool execution
builder.add_edge("tools", "assistant")

# Compile with safety
graph = builder.compile(debug=True)