# File: /backend/src/agent/utils.py

from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


# --- Incompatible Utility Functions for Local LLM Setup ---
# The following utility functions (resolve_urls, insert_citation_markers, get_citations)
# were originally designed for the `grounding_metadata` provided by Google's genai.Client
# when using its native Google Search tool.
#
# With the switch to a local LLM (via LM Studio) and generic search tools
# (e.g., DuckDuckGoSearchRun or other LangChain tools), this specific metadata
# structure is no longer available directly from the search tool or the local LLM.
#
# Therefore, these functions are commented out as they would not work as intended.
# The citation and source handling in the `graph.py` (web_research and finalize_answer nodes)
# has been simplified to reflect this change. You would need to implement a new
# approach for detailed citation extraction and formatting if required with your
# local setup, potentially by prompting the LLM to extract sources from search results
# and format them, or by using more sophisticated parsing of search tool outputs.

# def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
#     """
#     Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
#     Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
#     (Incompatible with local LLM setup as it relied on Google Vertex AI Search URL structure)
#     """
#     prefix = f"https://vertexaisearch.cloud.google.com/id/"
#     urls = [site.web.uri for site in urls_to_resolve]

#     # Create a dictionary that maps each unique URL to its first occurrence index
#     resolved_map = {}
#     for idx, url in enumerate(urls):
#         if url not in resolved_map:
#             resolved_map[url] = f"{prefix}{id}-{idx}"

#     return resolved_map


# def insert_citation_markers(text, citations_list):
#     """
#     Inserts citation markers into a text string based on start and end indices.
#     (Incompatible as `citations_list` structure from `get_citations` is unavailable)
#     """
#     # Sort citations by end_index in descending order.
#     # If end_index is the same, secondary sort by start_index descending.
#     # This ensures that insertions at the end of the string don't affect
#     # the indices of earlier parts of the string that still need to be processed.
#     sorted_citations = sorted(
#         citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
#     )

#     modified_text = text
#     for citation_info in sorted_citations:
#         # These indices refer to positions in the *original* text,
#         # but since we iterate from the end, they remain valid for insertion
#         # relative to the parts of the string already processed.
#         end_idx = citation_info["end_index"]
#         marker_to_insert = ""
#         for segment in citation_info["segments"]:
#             marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
#         # Insert the citation marker at the original end_idx position
#         modified_text = (
#             modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
#         )

#     return modified_text


# def get_citations(response, resolved_urls_map):
#     """
#     Extracts and formats citation information from a Gemini model's response.
#     (Incompatible as it relies on `response.candidates[0].grounding_metadata` which
#      is specific to Google's genai.Client and its integrated search tool)
#     """
#     citations = []

#     # Ensure response and necessary nested structures are present
#     if not response or not response.candidates:
#         return citations

#     candidate = response.candidates[0]
#     if (
#         not hasattr(candidate, "grounding_metadata")
#         or not candidate.grounding_metadata
#         or not hasattr(candidate.grounding_metadata, "grounding_supports")
#     ):
#         return citations

#     for support in candidate.grounding_metadata.grounding_supports:
#         citation = {}

#         # Ensure segment information is present
#         if not hasattr(support, "segment") or support.segment is None:
#             continue  # Skip this support if segment info is missing

#         start_index = (
#             support.segment.start_index
#             if support.segment.start_index is not None
#             else 0
#         )

#         # Ensure end_index is present to form a valid segment
#         if support.segment.end_index is None:
#             continue  # Skip if end_index is missing, as it's crucial

#         # Add 1 to end_index to make it an exclusive end for slicing/range purposes
#         # (assuming the API provides an inclusive end_index)
#         citation["start_index"] = start_index
#         citation["end_index"] = support.segment.end_index

#         citation["segments"] = []
#         if (
#             hasattr(support, "grounding_chunk_indices")
#             and support.grounding_chunk_indices
#         ):
#             for ind in support.grounding_chunk_indices:
#                 try:
#                     chunk = candidate.grounding_metadata.grounding_chunks[ind]
#                     resolved_url = resolved_urls_map.get(chunk.web.uri, None)
#                     citation["segments"].append(
#                         {
#                             "label": chunk.web.title.split(".")[:-1][0],
#                             "short_url": resolved_url,
#                             "value": chunk.web.uri,
#                         }
#                     )
#                 except (IndexError, AttributeError, NameError):
#                     # Handle cases where chunk, web, uri, or resolved_map might be problematic
#                     # For simplicity, we'll just skip adding this particular segment link
#                     # In a production system, you might want to log this.
#                     pass
#         citations.append(citation)
#     return citations