from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader


@tool
def wiki_tool(query: str) -> str:
    """
    Search Wikipedia and return relevant content.
    """

    try:
        docs = WikipediaLoader(
            query=query,
            load_max_docs=3
        ).load()

        if not docs:
            return "No results found"

        formatted = "\n\n---\n\n".join(
            f"Title: {doc.metadata.get('title', '')}\n{doc.page_content[:6000]}"
            for doc in docs
        )

        return formatted

    except Exception as e:
        return f"Wikipedia error: {str(e)}"