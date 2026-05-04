from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader

@tool
def wiki_tool(query: str) -> str:
    """
        Search Wikipedia and return information relevant to the query.

        Args:
            query (str): The search query for Wikipedia.
    """

    try:
        docs = WikipediaLoader(
            query=query,
            load_max_docs=2
        ).load()

        if not docs:
            return "No relevant Wikipedia results found."

        results = []

        for doc in docs:
            title = doc.metadata.get("title", "Unknown")
            content = doc.page_content

            results.append(
                f"Title: {title}\nContent:\n{content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Wikipedia error: {str(e)}"
