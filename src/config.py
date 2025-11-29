import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 1  #segundos
    RETRY_MAX_WAIT = 10 #segundos
    DELAY_BETWEEN_REQUESTS = 5 #segundos

    @classmethod
    def validate(cls):
        """Ensures critical keys are present based on usage."""
        if not cls.OPENAI_API_KEY and not \
        cls.ANTHROPIC_API_KEY and not \
        cls.GEMINI_API_KEY:
            print("""AVISO: Nenhuma chave de API encontrada no .env. 
                  Certifique-se de que seu provedor está configurado.""")

Config.validate()
