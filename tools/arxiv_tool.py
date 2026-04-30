from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 4 result.
    Args:
        query: The search query.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}