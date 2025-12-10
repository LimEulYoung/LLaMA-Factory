# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLaMA-Factory is a unified framework for fine-tuning 100+ large language models. It supports various training approaches (pre-training, SFT, DPO, PPO, KTO, ORPO, SimPO) and methods (full fine-tuning, freeze tuning, LoRA, QLoRA, OFT).

## Common Commands

### Installation
```bash
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Training
```bash
# Basic training with YAML config
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Override parameters on command line
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml learning_rate=1e-5

# Multi-GPU training (auto-detected, or force with FORCE_TORCHRUN=1)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config.yaml

# Multi-node training
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train config.yaml
```

### Inference & Export
```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml    # CLI chat
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml # Web chat
llamafactory-cli api examples/inference/llama3_lora_sft.yaml     # OpenAI-style API
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml # Merge LoRA & export
```

### Web UI
```bash
llamafactory-cli webui  # Launch LlamaBoard GUI
```

### Development
```bash
make style   # Auto-fix linting with ruff
make quality # Check linting without fixing
make test    # Run pytest (uses tests/ and tests_v1/)
```

## Code Architecture

### Source Layout (`src/llamafactory/`)

- **`cli.py` / `launcher.py`**: Entry points. Commands dispatch to appropriate modules (train, chat, api, export, webui).

- **`hparams/`**: Argument dataclasses parsed from YAML configs
  - `ModelArguments` - model path, quantization, adapter settings
  - `DataArguments` - dataset, template, tokenization settings
  - `TrainingArguments` - learning rate, batch size, optimizer settings
  - `FinetuningArguments` - training stage (pt/sft/rm/ppo/dpo/kto), LoRA rank, freeze layers
  - `GeneratingArguments` - inference parameters
  - `parser.py` - `get_train_args()`, `get_infer_args()` for loading configs

- **`data/`**: Dataset loading and processing
  - `loader.py` - loads datasets from HuggingFace, local files, or cloud storage
  - `template.py` - chat templates for different models (add custom templates here)
  - `processor/` - data processors for different training stages (supervised, pairwise, feedback, pretrain)
  - `collator.py` - data collators for batching
  - Dataset format defined in `data/dataset_info.json` (alpaca or sharegpt format)

- **`model/`**: Model loading and adaptation
  - `loader.py` - loads pretrained models with quantization support
  - `adapter.py` - applies LoRA/QLoRA/OFT adapters via PEFT
  - `patcher.py` - patches models for various optimizations
  - `model_utils/` - attention, quantization, rope scaling, visual encoders

- **`train/`**: Training workflows organized by stage
  - `tuner.py` - main entry, routes to appropriate trainer
  - `pt/` - pre-training
  - `sft/` - supervised fine-tuning
  - `rm/` - reward modeling
  - `ppo/` - PPO reinforcement learning
  - `dpo/` - DPO/ORPO/SimPO preference learning
  - `kto/` - KTO training
  - Each stage has `workflow.py` (orchestration) and `trainer.py` (custom Trainer class)

- **`chat/`**: Inference engines
  - `hf_engine.py` - HuggingFace Transformers
  - `vllm_engine.py` - vLLM backend
  - `sglang_engine.py` - SGLang backend

- **`api/`**: OpenAI-compatible API server using FastAPI

- **`webui/`**: Gradio-based LlamaBoard GUI

- **`extras/`**: Utilities
  - `constants.py` - supported models list and default templates
  - `env.py` - version info and environment detection
  - `packages.py` - optional dependency checks

- **`v1/`**: Experimental next-gen training backend (enabled with `USE_V1=1`)

## Configuration

Training configs are YAML files. Key parameters:
- `model_name_or_path`: HuggingFace model ID or local path
- `template`: Chat template name (must match model, see `extras/constants.py`)
- `dataset`: Dataset name(s) from `data/dataset_info.json`
- `stage`: Training stage (pt, sft, rm, ppo, dpo, kto, simpo, orpo)
- `finetuning_type`: full, freeze, lora, oft
- `quantization_bit`: 4 or 8 for QLoRA

See `examples/` for complete config examples.

## Adding Custom Datasets

1. Add dataset file to `data/` directory
2. Register in `data/dataset_info.json` with column mappings
3. Supports alpaca format (`instruction`, `input`, `output`) and sharegpt format (`conversations`)

## Custom Reward Function for PPO

Use `PPO (Custom Reward)` stage to train with your own reward function instead of a trained reward model.

### Usage

1. In GUI: Select `PPO (Custom Reward)` from the Training Stage dropdown
2. In CLI: Set `stage: ppo_custom` in your YAML config

### Modifying the Reward Function

Edit `custom_rewards/reward_functions.py`:

```python
def custom_reward(query: str, response: str) -> float:
    """
    Args:
        query: The input prompt/question
        response: The model's generated response
    Returns:
        A reward value (float)
    """
    reward = 0.0

    # Example: Length-based reward
    if len(response) > 50:
        reward += 0.5

    # Example: Keyword reward
    if "thank you" in response.lower():
        reward += 0.3

    # Example: External API call
    # reward += call_sentiment_api(response)

    return reward
```

### File Structure

```
LLaMA-Factory/
├── custom_rewards/
│   ├── __init__.py
│   └── reward_functions.py    # Edit this file
└── src/llamafactory/train/ppo_custom/
    ├── __init__.py
    ├── trainer.py             # CustomRewardPPOTrainer
    └── workflow.py
```
