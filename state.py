from typing import List, TypedDict, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    input_file: Optional[str]
    messages: Annotated[List[AnyMessage], add_messages]