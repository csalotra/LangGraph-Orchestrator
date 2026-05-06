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
    content=
    """
        You are a general AI assistant that gives answer using the tools provided.

        your final answer MUST strictly follow this format:
        FINAL ANSWER: [FINAL ANSWER]

        ---

        ## OUTPUT RULES (EXTREMELY IMPORTANT)

        - Only write the answer in the exact format specified. 
        - DO NOT add any explanations, comments, or extra text.

        ---

        ## TOOL USAGE RULES

        * You MUST use tools provided to answer the question.
        * If a tool is used, base the answer strictly on tool output.

        ---

        ## Examples:

        - FINAL ANSWER: FunkMonk
        - FINAL ANSWER: Paris
        - FINAL ANSWER: 128

        ---

        If you do not follow the output rules, your answer will be marked as incorrect, even if the content is right. Always follow the output rules strictly.

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
