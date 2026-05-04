from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# Initialize
search = TavilySearchResults(
max_results=3
)

@tool
def web_tool(query: str) -> str:
    """
    Search the web for relevant information.

    Use this tool when:
    - the question requires factual or up-to-date information
    - the answer depends on external data (e.g., entities, dates, categories, counts)

    BEHAVIOR:
    - Extract only information relevant to the query
    - Prefer precise, structured facts (e.g., names, dates, categories, attributes)
    - Include all attributes needed to apply conditions in the query (e.g., type, date, location)
    - Ignore ads, navigation text, and irrelevant content
    - Do NOT include opinions or unnecessary text

    OUTPUT:
    - Return concise, relevant information only
    - Avoid duplication
    - Ensure results contain enough detail for filtering and computation

    Args:
        query (str): The search query for the web.
    """
    try:
        results = search.invoke(query)

        if not results:
            return "No relevant results found."

        cleaned = []

        for r in results:
            content = r.get("content", "").strip()

            cleaned.append(content)

        # bullet points
        return "\n".join(cleaned)

    except Exception as e:
        return f"Web search error: {str(e)}"

