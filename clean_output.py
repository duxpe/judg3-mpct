import pandas as pd
import re
from pathlib import Path

def extract_last_valid_option(text: str) -> str:
    """
    Percorre o texto e retorna a ÚLTIMA letra entre A, B, C, D, E encontrada.
    Ignora pontuação, espaços e outros caracteres após a letra.
    
    Exemplos:
    "A resposta correta é B." -> Retorna "B"
    "Calculando x, temos 10. Logo, C" -> Retorna "C"
    "Alternativa E" -> Retorna "E"
    """
    if not isinstance(text, str):
        return "EMPTY"
    
    # Normaliza para maiúsculas
    text = text.upper()
    
    # Regex busca: (Letra A-E) seguida apenas por caracteres que NÃO são A-E até o fim da string
    # Ou seja, pega a última ocorrência de A, B, C, D ou E no texto inteiro.
    matches = re.findall(r'[A-E]', text)
    
    if matches:
        return matches[-1]
    
    return "INVALID"

def clean_directory_csvs(input_dir: Path):
    """
    Processa todos os CSVs no diretório alvo.
    """
    files = list(input_dir.glob("*.csv"))
    
    if not files:
        print(f"Nenhum arquivo .csv encontrado em {input_dir}")
        return

    print(f"Encontrados {len(files)} arquivos para processar...")

    for file_path in files:
        # Pula arquivos que já foram limpos para evitar loop ou duplicação
        if "_cleaned" in file_path.name:
            continue

        try:
            df = pd.read_csv(file_path)
            
            # Verifica se a coluna de resposta bruta existe
            if "resposta original" not in df.columns:
                print(f"Pular {file_path.name}: coluna 'resposta original' ausente.")
                continue

            # Aplica a limpeza
            # Criamos uma coluna de backup do parser original apenas para comparação (opcional)
            df['parser_antigo'] = df['alternativa escolhida pela ia']
            
            # Sobrescreve a coluna principal com a nova lógica
            df['alternativa escolhida pela ia'] = df['resposta original'].apply(extract_last_valid_option)

            # Salva com sufixo _cleaned
            output_name = file_path.stem + "_cleaned.csv"
            output_path = file_path.parent / output_name
            
            df.to_csv(output_path, index=False)
            print(f"Processado: {file_path.name} -> {output_name}")
            
            # Mostra estatística rápida de mudanças
            changes = df[df['parser_antigo'] != df['alternativa escolhida pela ia']].shape[0]
            if changes > 0:
                print(f"   -> {changes} respostas corrigidas neste arquivo.")

        except Exception as e:
            print(f"Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    # Define o diretório onde os raw_runs estão (mesmo do pipeline principal)
    RAW_RUNS_DIR = Path("data/output/raw_runs")
    
    if RAW_RUNS_DIR.exists():
        clean_directory_csvs(RAW_RUNS_DIR)
    else:
        print(f"Diretório {RAW_RUNS_DIR} não existe. Rode o pipeline primeiro.")