import time
import config
from google import genai
from huggingface_hub import InferenceClient
from services.logger import logger


def generate_response(prompt: str, model_name: str) -> str:
    """
    Unified public interface to route a prompt to the selected LLM and retrieve the generated response.

    Args:
        prompt (str): The constructed prompt string.
        model_name (str): The display name of the model to use (from config.LLM_MODELS).

    Returns:
        str: The generated text response from the model.

    Raises:
        ValueError: If the prompt is empty or model name is unsupported.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if model_name == config.MODEL_GEMINI:
        return _generate_gemini_response(prompt)
    elif model_name == config.MODEL_QWEN:
        return _generate_qwen_response(prompt)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def _generate_gemini_response(prompt: str) -> str:
    """
    Generates a response using Google's Gemini API via the GenAI SDK.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        llm_start = time.time()
        response = client.models.generate_content(
            model=config.GEMINI_LLM_MODEL,
            contents=prompt
        )
        generation_time = round(time.time() - llm_start, 2)
        logger.info("LLM Generation model=%s generation_time=%.2fs", config.GEMINI_LLM_MODEL, generation_time)
        if not response.text:
            return "Error: Gemini returned an empty response."
        return response.text
    except Exception as e:
        logger.error("LLM Generation model=%s error=%s", config.GEMINI_LLM_MODEL, e, exc_info=True)
        return f"Gemini API Error: {str(e)}"

def _generate_qwen_response(prompt: str) -> str:
    """
    Generates a response using Qwen-2.5 via Hugging Face Inference Client.
    """
    if not config.HF_API_KEY:
        raise ValueError("Hugging Face API key (HF_API_KEY) is not configured in the environment variables.")

    try:
        client = InferenceClient(api_key=config.HF_API_KEY)

        llm_start = time.time()
        response = client.chat.completions.create(
            model=config.QWEN_LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
        )
        generation_time = round(time.time() - llm_start, 2)
        logger.info("LLM Generation model=%s generation_time=%.2fs", config.QWEN_LLM_MODEL, generation_time)

        return response.choices[0].message.content
    except Exception as e:
        logger.error("LLM Generation model=%s error=%s", config.QWEN_LLM_MODEL, e, exc_info=True)
        return f"QWEN API Error: {str(e)}"
