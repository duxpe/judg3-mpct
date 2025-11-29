import argparse
from pathlib import Path
from src.pipeline import VestibularEvaluator
from src.analysis import ResultAnalyzer

DATA_INPUT = Path("data/input/questions.csv")
DATA_OUTPUT_RAW = Path("data/output/raw_runs")
DATA_OUTPUT_ANALYSIS = Path("data/output/analysis")

def main():
    parser = argparse.ArgumentParser(description="Judg3: Juiz de LLMs para vestibulares do Brasil;")
    
    parser.add_argument('--mode', choices=['run', 'analyze', 'all'], default='all', 
                        help="Escolha 'run' para executar testes, 'analyze' para gerar gráficos, ou 'all'.")
    parser.add_argument('--model', type=str, default='gpt-4o', 
                        help="Nome do modelo para testar (e.g., gpt-4o, claude-3-5-sonnet).")

    args = parser.parse_args()

    if args.mode in ['run', 'all']:
        print(f"--- Começando os testes para {args.model} ---")
        evaluator = VestibularEvaluator(
            input_path=DATA_INPUT,
            output_dir=DATA_OUTPUT_RAW,
            model_name=args.model
        )
        try:
            evaluator.run_evaluation()
        except FileNotFoundError as e:
            print(f"Erro no teste: {e}")
            return

    if args.mode in ['analyze', 'all']:
        print("--- Começando a análise ---")
        analyzer = ResultAnalyzer(
            result_dir=DATA_OUTPUT_RAW,
            output_dir=DATA_OUTPUT_ANALYSIS
        )
        try:
            analyzer.generate_accuracy_report()
        except Exception as e:
            print(f"Erro na análise: {e}")

if __name__ == "__main__":
    main()