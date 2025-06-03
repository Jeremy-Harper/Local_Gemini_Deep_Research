import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
# from google.genai import Client # Remove: No longer using genai_client directly

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
# from langchain_google_genai import ChatGoogleGenerativeAI # Remove
from langchain_openai import ChatOpenAI # Add
from langchain_community.tools import DuckDuckGoSearchRun # Add example search tool

from agent.utils import (
    # get_citations, # This will likely be incompatible
    get_research_topic,
    # insert_citation_markers, # This will likely be incompatible
    # resolve_urls, # This will likely be incompatible
)

load_dotenv()

# Remove GEMINI_API_KEY check, or adapt if you still use Google Search tools that need it
# if os.getenv("GEMINI_API_KEY") is None:
#     raise ValueError("GEMINI_API_KEY is not set")

# Remove genai_client
# genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Helper to get ChatOpenAI instance
def get_local_llm(configurable: Configuration, model_name_in_config: str, temperature: float = 0.7):
    model_identifier = getattr(configurable, model_name_in_config)
    api_key = configurable.openai_api_key if configurable.openai_api_key else "not_needed"
    return ChatOpenAI(
        model=model_identifier,
        openai_api_base=configurable.openai_api_base,
        openai_api_key=api_key,
        temperature=temperature,
        max_retries=2, # Optional
    )

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = get_local_llm(configurable, "query_generator_model", temperature=1.0)
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    
    print(f"--- Performing web research for query: {state['search_query']} ---")

    # Initialize a search tool (e.g., DuckDuckGo)
    # For Google Search, you'd use GoogleSearchRun and ensure GOOGLE_API_KEY/GOOGLE_CSE_ID are set
    search_tool = DuckDuckGoSearchRun()
    try:
        search_results_text = search_tool.run(state["search_query"])
    except Exception as e:
        print(f"Error during web search for '{state['search_query']}': {e}")
        search_results_text = "Error performing web search."

    # Use a local LLM to summarize the search results
    # The original web_searcher_instructions might need adjustment as it expected Google Search tool behavior
    llm = get_local_llm(configurable, "search_llm_model", temperature=0.0)
    
    summarization_prompt = f"""{web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"] 
    )}

    Search Results:
    {search_results_text}

    Please provide a concise summary of the findings based *only* on the provided search results.
    Include relevant URLs if found in the search results.
    """
    
    summary_response = llm.invoke(summarization_prompt)
    summary_content = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

    # --- Citation and Source Handling (Simplified / Placeholder) ---
    # The original citation mechanism (get_citations, insert_citation_markers, resolve_urls)
    # relied on Google's genai_client grounding_metadata, which is not available here.
    # You'll need to implement a new way to extract and format sources if desired.
    # For now, we'll just pass the raw summary and a simplified source list.
    
    # Example: try to extract URLs from the search_results_text (very basic)
    import re
    urls_found = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', search_results_text)
    
    simple_sources = [{"label": url.split('/')[2], "short_url": f"source-{state['id']}-{i}", "value": url} for i, url in enumerate(list(set(urls_found))[:5])] # Top 5 unique

    print(f"--- Web research for '{state['search_query']}' completed. Summary: {summary_content[:100]}... ---")

    return {
        "sources_gathered": simple_sources, # Simplified sources
        "search_query": [state["search_query"]],
        "web_research_result": [summary_content], # Summary from local LLM
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    
    # The original code used 'reasoning_model' from state or config.
    # We'll use 'reflection_model' as defined in our new Configuration.
    # If you want to keep 'reasoning_model' as a dynamic override, adjust accordingly.
    reasoning_model_name_key = "reflection_model" # state.get("reasoning_model") or configurable.reflection_model

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    llm = get_local_llm(configurable, reasoning_model_name_key, temperature=1.0)
    structured_llm = llm.with_structured_output(Reflection)
    
    result = structured_llm.invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState: # Type hint was OverallState, but it returns str or list of Send
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    # Similar to reflection, using 'answer_model' from our new Configuration
    answer_model_name_key = "answer_model" # state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = get_local_llm(configurable, answer_model_name_key, temperature=0.0)
    result = llm.invoke(formatted_prompt)
    result_content = result.content if hasattr(result, 'content') else str(result)

    # --- Simplified Citation Handling for Final Answer ---
    # The original citation mechanism is incompatible.
    # We can append the simplified sources to the end of the content, or just rely on the summary.
    # For now, we'll just use the LLM's generated content.
    # If you have simple_sources in state["sources_gathered"] from web_research, you could append them.
    
    final_content = result_content
    # Example: Appending sources if they exist and are simple
    # if state.get("sources_gathered"):
    #     sources_text = "\n\nSources:\n"
    #     for src in state["sources_gathered"]:
    #         sources_text += f"- [{src.get('label', src.get('value'))}]({src.get('value')})\n" # Basic formatting
    #     final_content += sources_text
        
    # The `unique_sources` logic based on `short_url` needs to be removed or adapted.
    # For now, we'll pass all gathered sources if any, or let the LLM handle citations if it can.
    # The structure of 'sources_gathered' has changed, so the frontend might need adjustment
    # if it heavily relied on the old structure.
    
    # What the frontend expects for sources_gathered:
    # [{'label': 'youtube', 'short_url': '...', 'value': '...'}]
    # Our simplified `simple_sources` in `web_research` node tries to match this.

    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": state.get("sources_gathered", []), # Pass along the (simplified) sources
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent-local-llm")