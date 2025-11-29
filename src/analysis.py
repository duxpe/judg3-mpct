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
        """
        Loads CSVs. Prioritizes files ending in '_cleaned.csv' if they exist.
        ALSO merges metadata (area_conhecimento) from the original input file
        to ensure categories are up-to-date.
        """
        # Search for cleaned files first
        cleaned_files = list(self.result_dir.glob("*_cleaned.csv"))
        
        if cleaned_files:
            print(f"--- Loading {len(cleaned_files)} cleaned files (_cleaned.csv) ---")
            files_to_load = cleaned_files
        else:
            # Fallback to standard files if cleanup wasn't run
            all_files = list(self.result_dir.glob("*.csv"))
            # Filter out files that might be temp/lock files
            files_to_load = [f for f in all_files if f.stat().st_size > 0]
            print(f"--- Loading {len(files_to_load)} raw files (recommend running clean_runs.py) ---")

        if not files_to_load:
            raise FileNotFoundError("No CSV files found in the output directory.")
            
        df_list = []
        for file in files_to_load:
            try:
                df = pd.read_csv(file)
                # Ensure compatibility if column names vary slightly
                if 'alternativa_correta' not in df.columns and 'alternativa correta' in df.columns:
                    df.rename(columns={'alternativa correta': 'alternativa_correta'}, inplace=True)
                
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        df_combined = pd.concat(df_list, ignore_index=True)

        # --- LOGIC TO SYNC WITH INPUT FILE ---
        # Loads data/input/questions.csv to get the definitive 'area_conhecimento'
        input_path = Path("data/input/questions.csv")
        if input_path.exists():
            print(f"--- Syncing metadata from {input_path} ---")
            try:
                df_input = pd.read_csv(input_path)
                
                # Normalize input column names (handle 'eixo de conhecimento' vs 'area_conhecimento')
                if 'eixo de conhecimento' in df_input.columns:
                    df_input.rename(columns={'eixo de conhecimento': 'area_conhecimento'}, inplace=True)
                
                # Check if we have the necessary columns for merge
                if 'area_conhecimento' in df_input.columns and 'questão' in df_input.columns:
                    # Drop existing area_conhecimento in processed df to prefer the input file version
                    if 'area_conhecimento' in df_combined.columns:
                        df_combined = df_combined.drop(columns=['area_conhecimento'])
                    
                    # Prepare merge keys (Questão + Vestibular is safer than just Questão)
                    merge_keys = ['questão']
                    if 'vestibular' in df_input.columns and 'vestibular' in df_combined.columns:
                        merge_keys.append('vestibular')
                        
                    cols_to_merge = merge_keys + ['area_conhecimento']
                    
                    # Perform Left Merge: Keep all run results, attach correct area info
                    df_combined = pd.merge(df_combined, df_input[cols_to_merge], on=merge_keys, how='left')
                    print("--- Metadata sync successful ---")
                    
            except Exception as e:
                print(f"Warning: Could not merge with input file: {e}")
        
        return df_combined

    def generate_accuracy_report(self):
        df = self.load_processed_files()

        # Normalization
        df['alternativa_correta'] = df['alternativa_correta'].astype(str).str.strip().str.upper()
        df['alternativa escolhida pela ia'] = df['alternativa escolhida pela ia'].astype(str).str.strip().str.upper()
        
        # Calculate Logic
        df['is_correct'] = df['alternativa_correta'] == df['alternativa escolhida pela ia']
        
        # General Settings for Graphs
        sns.set_theme(style="whitegrid")
        
        # 1. Graph per Model (Accuracy vs Temp)
        self._plot_per_model_accuracy(df)

        # 2. General Graph (Vestibular vs Model)
        self._plot_vestibular_comparison(df)

        # 3. Heatmap by Knowledge Axis (Extra)
        self._plot_accuracy_by_eixo(df)

    def _plot_per_model_accuracy(self, df: pd.DataFrame):
        """Generates one graph per model comparing temps."""
        models = df['modelo'].unique()
        
        # Create subdirectory for organization
        model_graph_dir = self.output_dir / "by_model"
        model_graph_dir.mkdir(exist_ok=True)

        for model in models:
            plt.figure(figsize=(8, 6))
            
            df_model = df[df['modelo'] == model]
            
            # Grouping
            accuracy_df = df_model.groupby('temperature used during test')['is_correct'].mean().reset_index()
            
            # Plot
            sns.barplot(
                data=accuracy_df, 
                x='temperature used during test', 
                y='is_correct', 
                hue='temperature used during test',
                palette="magma", 
                legend=False
            )
            
            # Aesthetics
            plt.title(f'Performance: {model}')
            plt.xlabel('Temperature')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.05)
            
            # Annotate values on bars
            for index, row in accuracy_df.iterrows():
                plt.text(row.name, row.is_correct + 0.02, f"{row.is_correct*100:.1f}%", color='black', ha="center")

            safe_name = model.replace("/", "-").replace(" ", "_")
            filename = model_graph_dir / f"accuracy_{safe_name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Graph saved: {filename}")

    def _plot_vestibular_comparison(self, df: pd.DataFrame):
        """Aggregated graph: Vestibular vs Model (Average of all temps)."""
        plt.figure(figsize=(14, 8))
        
        # Calculate mean accuracy per Model per Vestibular
        vest_df = df.groupby(['vestibular', 'modelo'])['is_correct'].mean().reset_index()
        
        sns.barplot(
            data=vest_df,
            x='vestibular',
            y='is_correct',
            hue='modelo',
            palette="viridis"
        )
        
        plt.title('General Benchmark: Accuracy by Vestibular by Model (Avg of Temps)')
        plt.xlabel('Vestibular')
        plt.ylabel('Average Accuracy')
        plt.ylim(0, 1.05)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = self.output_dir / "benchmark_vestibular_model.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")

    def _plot_accuracy_by_eixo(self, df: pd.DataFrame):
        """Heatmap: Knowledge Axis vs Model (More interesting than Temp for this view)."""
        # Ensure we don't crash if area_conhecimento is missing or NaN
        if 'area_conhecimento' not in df.columns:
            print("Warning: 'area_conhecimento' column not found. Skipping Heatmap.")
            return

        plt.figure(figsize=(14, 10))
        
        # Changed to Model vs Axis, as it is more useful for comparison
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
            cbar_kws={'label': 'Accuracy'}
        )
        
        plt.title('Performance Heatmap: Knowledge Area vs Model')
        plt.ylabel('Knowledge Area')
        plt.xlabel('Model')
        
        plt.tight_layout()
        filename = self.output_dir / "heatmap_knowledge_area.png"
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")