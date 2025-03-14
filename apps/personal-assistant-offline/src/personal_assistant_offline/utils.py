import random
import string

import tiktoken
from loguru import logger


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dictionaries with list handling."""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def generate_random_hex(length: int) -> str:
    """Generate a random hex string of specified length.

    Args:
        length: The desired length of the hex string.

    Returns:
        str: Random hex string of the specified length.
    """

    hex_chars = string.hexdigits.lower()
    return "".join(random.choice(hex_chars) for _ in range(length))


def clip_tokens(text: str, max_tokens: int, model_id: str) -> str:
    """Clip the text to a maximum number of tokens using the tiktoken tokenizer.
    Supports both OpenAI models and Ollama models with appropriate fallbacks.

    Args:
        text: The input text to clip.
        max_tokens: Maximum number of tokens to keep.
        model_id: The model name to determine encoding.

    Returns:
        str: The clipped text that fits within the token limit.
    """
    # For Ollama models, extract the base model name if possible
    if model_id.startswith("ollama/"):
        # Remove the "ollama/" prefix
        base_model = model_id.split("/")[1]
    else:
        # Just use the model_id as is, it might already be without the prefix
        base_model = model_id
        
        # Extract the model family name (before the colon if present)
        if ":" in base_model:
            base_model = base_model.split(":")[0]
            
        logger.debug(f"Using base model name '{base_model}' for tokenization")
        
        # Map known Ollama models to appropriate tiktoken encodings
        model_mapping = {
            "qwen": "cl100k_base",  # Use cl100k_base for Qwen models
            "qwen2": "cl100k_base",  # Use cl100k_base for Qwen2 models
            "qwen2.5": "cl100k_base",  # Use cl100k_base for Qwen2.5 models
            "llama": "cl100k_base",  # Use cl100k_base for Llama models
            "llama2": "cl100k_base",  # Use cl100k_base for Llama2 models
            "mistral": "cl100k_base",  # Use cl100k_base for Mistral models
            "phi": "cl100k_base",  # Use cl100k_base for Phi models
        }
        
        # If we have a mapping for this model, use it; otherwise fall back to cl100k_base
        if base_model in model_mapping:
            model_id = model_mapping[base_model]
        else:
            logger.warning(f"Unknown Ollama model '{model_id}', falling back to cl100k_base encoding")
            model_id = "cl100k_base"

    try:
        # Try to get the encoding for the specific model
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError:
        # Fallback to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002)
        logger.warning(f"No specific encoding found for {model_id}, falling back to cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text and check token count
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        logger.debug(f"Text is within token limit ({len(tokens)}/{max_tokens})")
        return text

    # Clip to max_tokens
    logger.info(f"Text exceeds token limit ({len(tokens)} > {max_tokens}), clipping to {max_tokens} tokens")
    return encoding.decode(tokens[:max_tokens])
