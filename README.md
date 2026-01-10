# Detecting Doubt in Reflective Learning

This repository contains the code and experiments for the research paper:

**"Detecting Doubt in Reflective Learning: A Learning Analytics Study with Large and Small Language Models"**
Submitted to *Proceedings of The International Conference on Learning Analytics & Knowledge (LAK 2026)*

## Overview

This project investigates the use of Large Language Models (LLMs) and Small Language Models (SLMs) to automatically detect expressions of doubt in student learning reflections. The repository implements multiple AI models, multi-agent deliberation strategies, and ensemble methods to classify student reflections as expressing doubt or not expressing doubt.

## Main Scripts

The repository contains five main Jupyter notebooks in the root folder:

### Core Experiment Notebooks

#### `llm_doubts.ipynb`
The main pipeline for testing individual models (both LLMs and SLMs). This notebook:
- Implements zero-shot, one-shot, and few-shot prompting strategies
- Tests various LLMs (GPT-4, Claude Sonnet 4, Gemini 2.5 Flash)
- Tests various SLMs (Llama 3.2, Mistral 3.1, DeepSeek R1, Qwen 3)
- Saves individual model results to `output/llm/` and `output/slm/` directories

### Multi-Agent Deliberation (MAD) Notebooks

#### `llm_doubts_mad_judge.ipynb`
Implements a judge-based multi-agent system where:
- Multiple models analyze the same reflection
- A third-party judge model evaluates competing arguments
- Results saved to `output/mad/` directory

#### `llm_doubts_mad_self_consistency.ipynb`
Implements a self-consistency approach where:
- The same model runs multiple times on each reflection
- Final classification determined by majority voting
- Provides robust predictions through consensus

#### `llm_doubts_mad_two_agents.ipynb`
Implements a two-agent deliberation system:
- One agent acts as a "prosecutor" (arguing for doubt)
- Another acts as a "defender" (arguing against doubt)
- Debate format generates reasoning before final classification

### Analysis Notebook

#### `llm_summary.ipynb`
Consolidates and analyzes results from all experiments:
- Merges PKL files from individual model runs
- Analyzes LLM, SLM, and MAD performance
- Generates ensemble methods combining multiple models
- Produces visualizations (ROC curves, confusion matrices)
- Evaluates comprehensive metrics (accuracy, precision, recall, F1, F2, specificity)

## Project Structure

```
doubt-llm/
├── llm_doubts.ipynb                       # Main individual model testing
├── llm_doubts_mad_judge.ipynb             # Judge-based MAD
├── llm_doubts_mad_self_consistency.ipynb  # Self-consistency MAD
├── llm_doubts_mad_two_agents.ipynb        # Two-agent debate MAD
├── llm_summary.ipynb                      # Results analysis
├── data/                                  # Student reflection datasets
│   ├── processed_dataset.csv              # Main dataset
│   └── ...
├── output/                                # Experiment results
│   ├── llm/                               # Large Language Model results
│   ├── slm/                               # Small Language Model results
│   └── mad/                               # Multi-Agent Deliberation results
└── output_merged/                         # Consolidated results
```

## Setup

### Quickstart

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate
# install deps listed in Step 1 of llm_doubts.ipynb
cp .env.example .env
# update .env with your API keys before running notebooks
jupyter notebook  # open llm_doubts.ipynb and run all cells
```

### Requirements

- Python 3.9+
- Jupyter Notebook
- API keys for: OpenAI, Anthropic, Google (Gemini), Mistral, DeepSeek, Qwen
- Ollama (for local SLMs)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies (see Step 1 in each notebook for specific requirements)

3. Set up environment variables:
   ```bash
   # Copy the example file and add your API keys
   cp .env.example .env
   # update .env with your API keys before running notebooks
   ```
   Then edit `.env` and replace the placeholder values with your actual API keys

4. For local SLMs, install and start Ollama:
   ```bash
   # Install required models
   ollama pull mistral-small3.1:latest
   ollama pull deepseek-r1:latest
   ollama pull qwen3:8b-q8_0
   ollama pull llama3.2:latest

   # Start Ollama server
   ollama serve
   ```

## Usage

1. **Test individual models**: Run `llm_doubts.ipynb`
2. **Run MAD experiments**: Execute the desired MAD notebook
3. **Analyze results**: Use `llm_summary.ipynb` to consolidate and visualize findings

Each notebook is self-contained with step-by-step instructions.

## Data

The dataset contains student learning reflections with binary labels:
- **Label 1**: Reflection expresses doubt about learning
- **Label 0**: Reflection does not express doubt

## Evaluation Metrics

Models are evaluated using:
- Accuracy, Precision, Recall, Specificity
- F1 Score (harmonic mean of precision and recall)
- F2 Score (emphasizes recall over precision)
- ROC curves and AUC
- Confusion matrices

## Contact

For more information about this research, please contact:

**prompttutorproject@gmail.com**

## Citation

If you use this code or find this research useful, please cite our paper:

```
Detecting Doubt in Reflective Learning: A Learning Analytics Study with Large and Small Language Models
Proceedings of The International Conference on Learning Analytics & Knowledge (LAK 2026)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
