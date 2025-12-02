import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

class ResultAnalyzer:
    def __init__(self, result_dir: Path, output_dir: Path):
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_processed_files(self) -> pd.DataFrame:
        cleaned_files = list(self.result_dir.glob("*_cleaned.csv"))
        if cleaned_files:
            print(f"--- Loading {len(cleaned_files)} cleaned files (_cleaned.csv) ---")
            files_to_load = cleaned_files
        else:
            all_files = list(self.result_dir.glob("*.csv"))
            files_to_load = [f for f in all_files if f.stat().st_size > 0]
            print(f"--- Loading {len(files_to_load)} raw files (recommend running clean_runs.py) ---")
        
        if not files_to_load:
            raise FileNotFoundError("No CSV files found in the output directory.")
            
        df_list = []
        for file in files_to_load:
            try:
                df = pd.read_csv(file)
                if 'alternativa_correta' not in df.columns and 'alternativa correta' in df.columns:
                    df.rename(columns={'alternativa correta': 'alternativa_correta'}, inplace=True)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        df_combined = pd.concat(df_list, ignore_index=True)
        
        input_path = Path("data/input/questions.csv")
        if input_path.exists():
            print(f"--- Syncing metadata from {input_path} ---")
            try:
                df_input = pd.read_csv(input_path)
                if 'eixo de conhecimento' in df_input.columns:
                    df_input.rename(columns={'eixo de conhecimento': 'area_conhecimento'}, inplace=True)
                
                if 'area_conhecimento' in df_input.columns and 'questão' in df_input.columns:
                    if 'area_conhecimento' in df_combined.columns:
                        df_combined = df_combined.drop(columns=['area_conhecimento'])
                    
                    merge_keys = ['questão']
                    if 'vestibular' in df_input.columns and 'vestibular' in df_combined.columns:
                        merge_keys.append('vestibular')
                        
                    cols_to_merge = merge_keys + ['area_conhecimento']
                    df_combined = pd.merge(df_combined, df_input[cols_to_merge], on=merge_keys, how='left')
                    print("--- Metadata sync successful ---")
            except Exception as e:
                print(f"Warning: Could not merge with input file: {e}")
        
        return df_combined

    def generate_accuracy_report(self):
        df = self.load_processed_files()

        df['alternativa_correta'] = df['alternativa_correta'].astype(str).str.strip().str.upper()
        df['alternativa escolhida pela ia'] = df['alternativa escolhida pela ia'].astype(str).str.strip().str.upper()
        
        df['is_correct'] = df['alternativa_correta'] == df['alternativa escolhida pela ia']
        
        sns.set_theme(style="whitegrid")
        
        self._plot_per_model_accuracy(df)
        self._plot_vestibular_comparison(df)
        self._plot_accuracy_by_eixo(df)
        self._plot_comparative_models_by_temp(df)

    def _plot_per_model_accuracy(self, df: pd.DataFrame):
        models = df['modelo'].unique()
        model_graph_dir = self.output_dir / "by_model"
        model_graph_dir.mkdir(exist_ok=True)

        for model in models:
            plt.figure(figsize=(8, 6))
            df_model = df[df['modelo'] == model]
            accuracy_df = df_model.groupby('temperature used during test')['is_correct'].mean().reset_index()
            
            sns.barplot(
                data=accuracy_df, 
                x='temperature used during test', 
                y='is_correct', 
                hue='temperature used during test',
                palette="magma", 
                legend=False
            )
            
            plt.title(f'Desempenho Individual: {model}')
            plt.xlabel('Temperatura')
            plt.ylabel('Precisão')
            plt.ylim(0, 1.05)
            
            for index, row in accuracy_df.iterrows():
                plt.text(row.name, row.is_correct + 0.02, f"{row.is_correct*100:.1f}%", color='black', ha="center")

            safe_name = model.replace("/", "-").replace(" ", "_")
            filename = model_graph_dir / f"accuracy_{safe_name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Graph saved: {filename}")

    def _plot_vestibular_comparison(self, df: pd.DataFrame):
        plt.figure(figsize=(14, 8))
        
        vest_df = df.groupby(['vestibular', 'modelo'])['is_correct'].mean().reset_index()
        
        sns.barplot(
            data=vest_df,
            x='vestibular',
            y='is_correct',
            hue='modelo',
            palette="viridis"
        )
        
        plt.title('Benchmark Geral: Precisão por Vestibular e Modelo (Média das Temperaturas)')
        plt.xlabel('Vestibular')
        plt.ylabel('Precisão Média')
        plt.ylim(0, 1.05)
        plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = self.output_dir / "benchmark_vestibular_model.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")

    def _plot_accuracy_by_eixo(self, df: pd.DataFrame):
        if 'area_conhecimento' not in df.columns:
            print("Warning: 'area_conhecimento' column not found. Skipping Heatmap.")
            return

        plt.figure(figsize=(14, 10))
        
        pivot = df.pivot_table(
            index='area_conhecimento', 
            columns='modelo', 
            values='is_correct', 
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".0%", 
            cmap="RdYlGn", 
            vmin=0, 
            vmax=1,
            cbar_kws={'label': 'Precisão'}
        )
        
        plt.title('Mapa de Calor de Desempenho: Área de Conhecimento vs Modelo')
        plt.ylabel('Área de Conhecimento')
        plt.xlabel('Modelo')
        
        plt.tight_layout()
        filename = self.output_dir / "heatmap_knowledge_area.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")

    def _plot_comparative_models_by_temp(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 8))
        
        sns.barplot(
            data=df,
            x='temperature used during test',
            y='is_correct',
            hue='modelo',
            palette="tab10"
        )
        
        plt.title('Comparativo Global: Performance de Todos os Modelos por Temperatura')
        plt.xlabel('Temperatura')
        plt.ylabel('Precisão (com Intervalo de Confiança 95%)')
        plt.ylim(0, 1.05)
        plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = self.output_dir / "comparison_all_models_by_temp.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")