"""Ollama LLM client with streaming token generation."""

import re
from collections.abc import AsyncGenerator

import ollama
from loguru import logger


DEFAULT_MODEL = "gemma3:4b"

# Sentence boundary pattern: split on .!? followed by space or end-of-string
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split text into complete sentences and a remaining buffer.

    Returns:
        A tuple of (complete_sentences, remaining_buffer).
        Complete sentences end with sentence-ending punctuation.
        The remaining buffer is text that hasn't been terminated yet.
    """
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        # No sentence boundary found — everything stays in buffer
        return [], text

    # All parts except the last are complete sentences
    complete = parts[:-1]
    remaining = parts[-1]
    return complete, remaining


async def generate_sentences(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> AsyncGenerator[str, None]:
    """Stream sentences from Ollama for the given conversation.

    Buffers tokens internally and yields complete Sentence Chunks.

    Args:
        messages: The conversation history including system prompt.
        model: Ollama model name to use.

    Yields:
        Complete sentence strings as they are generated.
    """
    logger.debug(
        "Starting LLM generation with model={model}, {n} messages",
        model=model,
        n=len(messages),
    )

    client = ollama.AsyncClient()
    token_buffer = ""
    total_tokens = 0

    try:
        async for chunk in await client.chat(
            model=model,
            messages=messages,
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                total_tokens += 1
                token_buffer += token

                # Check for sentence boundaries
                sentences, token_buffer = _split_sentences(token_buffer)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence

        # Handle any remaining text in the buffer
        if token_buffer.strip():
            yield token_buffer.strip()

    except ollama.ResponseError as e:
        logger.error("Ollama API error: {err}", err=e)
        raise
    except Exception as e:
        logger.error("Unexpected error during LLM generation: {err}", err=e)
        raise
    finally:
        logger.info(
            "LLM generation complete: {n} tokens processed",
            n=total_tokens,
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
