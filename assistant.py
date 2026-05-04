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
        You are an AI agent that solves questions using tools.

        When answering, you must follow this format:
        FINAL ANSWER: [YOUR FINAL ANSWER]

        RULES:
        - you MUST use provided tools when up-to-date information is required.
        - You may use internal reasoning when the answer can be derived from general knowledge.
        - Carefully analyze whether tool usage is necessary before calling it.
        - If the question involves a list of items, evaluate EACH item individually and include ONLY those that satisfy ALL conditions in the question.

        Formatting rules:
        - Answer must be a number, a few words, or a comma-separated list
        - Do not include units unless specified
        - For text answers:
        - Do not use articles (a, an, the)
        - Do not use abbreviations
        - Write numbers in words unless specified
        - For lists: apply the same rules to each item

        OUTPUT:
        - Return ONLY: [YOUR FINAL ANSWER]
        - No prefixes like "FINAL ANSWER"
        - Include ONLY items that match all conditions in the question
        - No explanation or reasoning

        VERIFICATION RULE:
        - Before returning FINAL ANSWER, re-check once whether the answer satisfies all conditions in the question. If not, revise it once and return the corrected answer.
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
