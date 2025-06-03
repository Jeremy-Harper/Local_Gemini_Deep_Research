import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    # --- OpenAI/LM Studio Specific Configuration ---
    openai_api_base: str = Field(
        default="http://localhost:1234/v1",
        metadata={"description": "The API base URL for the OpenAI-compatible LLM (e.g., LM Studio)."},
    )
    openai_api_key: Optional[str] = Field(
        default="not_needed", # LM Studio often doesn't require a key
        metadata={"description": "The API key for the OpenAI-compatible LLM."},
    )

    # --- Model Names for LM Studio (these should match identifiers in LM Studio) ---
    query_generator_model: str = Field(
        default="local-model", # Replace with your actual default model in LM Studio
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )
    # This model was used by genai_client for search tool interaction.
    # If using a different search tool, this might be used by the LLM processing search results.
    search_llm_model: str = Field(
        default="local-model", # Replace with your actual default model in LM Studio
        metadata={
            "description": "The name of the language model to use for processing search results."
        },
    )
    reflection_model: str = Field(
        default="local-model", # Replace with your actual default model in LM Studio
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )
    answer_model: str = Field(
        default="local-model", # Replace with your actual default model in LM Studio
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    # --- Original Configuration Fields (can remain) ---
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )
    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        # Ensure OPENAI_API_KEY is truly optional and doesn't become an empty string if not set
        if "openai_api_key" in values and values["openai_api_key"] == "not_needed":
            values["openai_api_key"] = None


        return cls(**values)