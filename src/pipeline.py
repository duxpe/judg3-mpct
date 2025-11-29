import time
from threading import Lock
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import Config
from src.llm_client import get_completion as _llm_get_completion

_request_lock = Lock()
_last_request_time = 0.0


def get_completion(*args, **kwargs):
    global _last_request_time
    with _request_lock:
        now = time.monotonic()
        wait_time = Config.DELAY_BETWEEN_REQUESTS - (now - _last_request_time)
        if wait_time > 0:
            time.sleep(wait_time)
        _last_request_time = time.monotonic()
    return _llm_get_completion(*args, **kwargs)


class VestibularEvaluator:
    def __init__(self, input_path: Path, output_dir: Path, model_name: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.temperatures = [0, 0.5, 1]
        self.max_workers = 4

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found at {self.input_path}")
        return pd.read_csv(self.input_path)

    def process_row(self, row: pd.Series, temp: float) -> Dict[str, Any]:
        """
        Constructs the prompt and calls the LLM.
        """
        prompt = (
            f"Responda a questão a seguir, usando apenas seus conhecimentos, sem acessar a internet."
            f"A resposta deve ser somente a letra correspondente a alternativa correta.\n"
            f"---\n"
            f"Questão: {row['questão']}\n"
            f"A) {row['alternativa_a']}\n"
            f"B) {row['alternativa_b']}\n"
            f"C) {row['alternativa_c']}\n"
            f"D) {row['alternativa_d']}\n"
            f"E) {row['alternativa_e']}\n\n"
            f"Responda apenas com uma letra, a letra da alternativa correta (A, B, C, D, E).\n\n"
            f"---\n"
            f"Exemplo de resposta 1: A\n"
            f"Exemplo de resposta 2: D"
        )

        start_time = time.time()
        raw_response = ""

        try:
            response = get_completion(model=self.model_name, prompt=prompt, temperature=temp)
            raw_response = response.content
            
            clean_content = raw_response.strip().upper()
            if clean_content:
                answer = clean_content[0]
            else:
                answer = "!EMPTY!"
                
        except Exception as e:
            print(f"Erro no request LLM: {e}")
            answer = "!ERROR!"
            if not raw_response:
                raw_response = str(e)

        elapsed_time = time.time() - start_time

        return {
            "alternativa escolhida pela ia": answer,
            "resposta original": raw_response, # Nova coluna com o output bruto
            "modelo": self.model_name,
            "processing time": round(elapsed_time, 4),
            "temperature used during test": temp
        }

    def _process_wrapper(self, args: Tuple[int, pd.Series, float]) -> Tuple[int, Dict[str, Any]]:
        """
        Wrapper interno para passar argumentos e retornar o índice original.
        Necessário para reordenar os dados após o multithreading.
        """
        index, row, temp = args
        try:
            result = self.process_row(row, temp)
            return index, result
        except Exception as e:
            print(f"Erro fatal na thread {index}: {e}")
            return index, {
                "alternativa escolhida pela ia": "FATAL_ERROR",
                "resposta original": str(e),
                "modelo": self.model_name,
                "processing time": 0,
                "temperature used during test": temp
            }

    def run_evaluation(self):
        df_original = self.load_data()
        print(f"Iniciando avaliação para o modelo: {self.model_name}")

        for temp in self.temperatures:
            print(f"Processing Temperature: {temp} | Threads: {self.max_workers}")
            
            # Dicionário para armazenar resultados desordenados: {index: result_dict}
            results_map = {}
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submissão das tarefas
                futures = []
                for index, row in df_original.iterrows():
                    future = executor.submit(self._process_wrapper, (index, row, temp))
                    futures.append(future)

                # Processamento conforme as threads terminam
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Temp {temp}"):
                    index, metadata = future.result()
                    results_map[index] = metadata

            # Reconstrói a lista de resultados na ordem correta dos índices (0 a N)
            ordered_results = [results_map[i] for i in range(len(df_original))]

            # Criação do DataFrame final
            df_results = pd.DataFrame(ordered_results)
            df_final = pd.concat([df_original, df_results], axis=1)

            safe_model_name = self.model_name.replace("/", "-")
            filename = f"judg3_{safe_model_name}_t{temp}.csv"
            output_path = self.output_dir / filename
            
            df_final.to_csv(output_path, index=False)
            print(f"Resultados salvos em {output_path}")