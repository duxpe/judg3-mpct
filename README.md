# judg3

Ferramenta de benchmarking automatizada para comparar o desempenho de Modelos de Linguagem (LLMs) na resolução de questões de vestibulares brasileiros.  
O sistema executa testes em múltiplas temperaturas (`0`, `0.5`, `1`) e gera análises estatísticas de precisão e tempo de processamento.

---

## Requisitos

- **Python 3.13+**
- **Chaves de API** (OpenAI, Anthropic, ou endpoint local compatível com OpenAI/LiteLLM).

---

## Instalação

### Versão do Python
3.13.8

### Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Dependências

```bash
pip install -r requirements.txt
```

---

## Configuração

Crie um arquivo `.env` na raiz do projeto contendo as chaves de API necessárias para os modelos que deseja testar:

```env
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Estrutura de Dados

### Entrada (`data/input/questions.csv`)

O arquivo CSV de entrada deve conter estritamente as seguintes colunas:

- **vestibular**: Nome da instituição (ex: FUVEST, ENEM).
- **questão**: Enunciado da questão.
- **alternativa a**: Texto da alternativa A.
- **alternativa b**: Texto da alternativa B.
- **alternativa c**: Texto da alternativa C.
- **alternativa d**: Texto da alternativa D.
- **alternativa e**: Texto da alternativa E.
- **alternativa correta**: Apenas a letra da resposta (A, B, C, D ou E).
- **eixo de conhecimento**: Área da matéria (ex: História, Matemática).

---

## Execução

O ponto de entrada é o script `main.py`. Utilize os argumentos de linha de comando para controlar o fluxo.

### 1. Executar Avaliação e Análise (Padrão)

Executa os testes nas três temperaturas e gera os gráficos imediatamente.

```bash
python main.py --mode all --model gpt-4o
```

### 2. Apenas Execução (Geração de Dados)

Processa as questões e salva os CSVs brutos em `data/output/raw_runs/`.

```bash
python main.py --mode run --model gpt-4o
```

### 3. Apenas Análise (Geração de Gráficos)

Lê os arquivos existentes em `data/output/raw_runs/` e recria os gráficos.

```bash
python main.py --mode analyze
```

---

## Saída de Dados

### Arquivos Processados (`data/output/raw_runs/`)

O sistema gera três arquivos por execução (um para cada temperatura: `0`, `0.5`, `1`).  
O CSV de saída mantém as colunas originais e adiciona:

- **alternativa escolhida pela ia**: A letra extraída da resposta do modelo.
- **modelo**: O identificador do modelo utilizado.
- **processing time**: Tempo de latência da resposta em segundos.
- **temperature used during test**: Temperatura configurada (`0`, `0.5` ou `1`).

### Relatórios Gráficos (`data/output/analysis/`)

- `accuracy_vs_temp.png`: Gráfico de barras comparando a precisão global por temperatura.
- `accuracy_heatmap_eixo.png`: Mapa de calor mostrando o desempenho por Eixo de Conhecimento vs. Temperatura.

---

## Tratamento de Erros

- **Falhas de API**: O sistema utiliza a biblioteca `tenacity` para realizar até 3 novas tentativas com exponential backoff em caso de `RateLimitError` ou falhas de conexão.
- **Parsing**: Se o modelo não retornar uma letra válida, o sistema tenta extrair a primeira ocorrência de `[A-E]` no texto. Caso falhe, registra como `"INVALID"`.