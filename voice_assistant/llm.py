"""Ollama LLM client with streaming token generation."""

from collections.abc import AsyncGenerator

import ollama
from loguru import logger


DEFAULT_MODEL = "gemma3:4b"


async def generate_response(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama for the given conversation.

    Yields individual token strings as they arrive from the model.
    The caller is responsible for buffering tokens into sentence chunks.

    Args:
        messages: The conversation history including system prompt.
        model: Ollama model name to use.

    Yields:
        Token strings as they are generated.
    """
    logger.debug(
        "Starting LLM generation with model={model}, {n} messages",
        model=model,
        n=len(messages),
    )

    client = ollama.AsyncClient()
    full_response = []

    try:
        async for chunk in await client.chat(
            model=model,
            messages=messages,
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                full_response.append(token)
                yield token
    except ollama.ResponseError as e:
        logger.error("Ollama API error: {err}", err=e)
        raise
    except Exception as e:
        logger.error("Unexpected error during LLM generation: {err}", err=e)
        raise
    finally:
        total_text = "".join(full_response)
        logger.info(
            "LLM generation complete: {n} tokens, {chars} chars",
            n=len(full_response),
            chars=len(total_text),
        )


async def check_connection(model: str = DEFAULT_MODEL) -> bool:
    """Verify Ollama is running and the model is available.

    Returns True if the model is ready, False otherwise.
    """
    try:
        client = ollama.AsyncClient()
        model_list = await client.list()
        available = [m.model for m in model_list.models]
        if model in available:
            logger.info("Ollama connected. Model '{model}' is available.", model=model)
            return True
        else:
            logger.warning(
                "Model '{model}' not found. Available models: {available}",
                model=model,
                available=available,
            )
            # Try to find partial match (e.g., "gemma3:4b" might be listed as "gemma3:4b-latest")
            partial_matches = [m for m in available if model.split(":")[0] in m]
            if partial_matches:
                logger.info(
                    "Partial matches found: {matches}", matches=partial_matches
                )
            return False
    except Exception as e:
        logger.error("Cannot connect to Ollama: {err}", err=e)
        return False
