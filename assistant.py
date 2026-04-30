import json
from langchain_core.messages import HumanMessage, SystemMessage
from models.llm import get_llm
from tools import ALL_TOOLS
from state import AgentState

llm = get_llm()
llm_with_tools = llm.bind_tools(ALL_TOOLS)

def assistant(state: AgentState):
    messages = state["messages"]

    system_prompt = SystemMessage(
    content="""
        You are an AI agent solving tasks.

        Rules:
        - You MUST use available tools whenever required to solve the problem.
        - If  you don't find the suitable tool, only then use your own knowledge
        - After using tools, always compute the final answer from the tool output.
        - If tool output is insufficient → call tool again
        - Extract information step-by-step from tool results
        - Respond with ONLY the final answer.
        - Do NOT include explanations, reasoning, steps, or extra text.
        """
    )

    last_msg = messages[-1] if messages else None

    data = {}
    if last_msg and last_msg.type == "tool":
        try:
            data = json.loads(last_msg.content) if isinstance(last_msg.content, str) else last_msg.content
        except json.JSONDecodeError:
            data = {}

    if isinstance(data, dict) and "frames" in data:

        frames = data["frames"]

        vision_content = [
            {
                "type": "text",
                "text": f"Answer the question using these video frames: {messages[0].content}"
            }
        ]

        for frame in frames:
             vision_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })

        vision_msg = HumanMessage(content=vision_content)
        response =  llm.invoke([system_prompt, vision_msg])
    
        return {"messages": [response]}


    response = llm_with_tools.invoke(
        [system_prompt] + messages
    )

    return {"messages": [response]}