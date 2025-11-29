import time
from dataclasses import dataclass
from typing import Optional
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from src.config import Config

litellm.suppress_instrumentation = True

@dataclass
class LLMResponse:
    """
    Objeto padrão de resposta
    """
    content: str
    model: str
    usage: dict

@retry(
    wait=wait_random_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    stop=stop_after_attempt(Config.MAX_RETRIES),
    retry=retry_if_exception_type((litellm.APIConnectionError, 
                                   litellm.RateLimitError, 
                                   litellm.ServiceUnavailableError))
)
def get_completion(model: str, prompt: str, temperature: float = 0) -> LLMResponse:
    """
    Wrapper do litellm com retry e output em LLMResponse.
    Inclui delay síncrono configurado.
    
    Args:
        model: The model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'ollama/llama3').
        prompt: The full input text.
        temperature: Creativity parameter (0 to 1).

    Returns:
        LLMResponse objeto da classe de resposta (LLMResponse)
    """
    try:
        print(prompt)
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""
        
        usage = response.usage.model_dump() if hasattr(response, 'usage') else {}

        # Delay fixo para evitar rate limits. Com 80 questões leva 8 minutos
        time.sleep(Config.DELAY_BETWEEN_REQUESTS)

        return LLMResponse(
            content=content,
            model=model,
            usage=usage
        )

    except Exception as e:
        raise e