from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

tavily = TavilySearchResults(max_results=5)

@tool
def web_tool(query: str) -> str:
    """
    Search the web (structured, reliable).

    Use ONLY when:
    - Wikipedia is insufficient
    - external/up-to-date info is needed
    """

    try:
        results = tavily.invoke(query)
        return str(results)
    except Exception as e:
        return f"Web search error: {str(e)}"