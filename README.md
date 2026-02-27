# SemEval 2026 Task 12: Abductive Reasoning with Llama 3.2

This repository contains a complete pipeline for training, evaluating, and generating test predictions for SemEval 2026 Task 12. The task requires a model to identify the most probable cause of a target event using abductive reasoning over a set of provided context documents.

The solution fine-tunes a 4-bit quantized Llama-3.2-1B-Instruct model using Unsloth and Low-Rank Adaptation (LoRA) for efficient, resource-friendly training.

## Overview

The code performs the following primary operations:
1. Installs the necessary dependencies (Unsloth, TRL, PEFT, etc.).
2. Downloads and processes the SemEval 2026 Task 12 dataset.
3. Fine-tunes a Llama-3.2-1B-Instruct model using LoRA.
4. Evaluates the model on the development set using a custom SemEval scoring metric.
5. Generates formatting-compliant predictions for the test set.

## Requirements and Installation

The notebook runs on a GPU-enabled environment (e.g., Google Colab with a T4 GPU). The following main libraries are required:

- python >= 3.10
- unsloth
- trl == 0.22.0
- xformers
- peft
- accelerate
- bitsandbytes
- jsonlines
- pandas
- datasets

These can be installed directly as shown in the notebook:

```bash
pip install "unsloth[colab-new]"
pip install trl==0.22.0
pip install --no-deps xformers peft accelerate bitsandbytes
pip install jsonlines
```

## Dataset Structure

The dataset is automatically cloned from `https://github.com/sooo66/semeval2026-task12-dataset.git`. 
It consists of train, dev, and test sets. 

The pipeline parses the documents and constructs structured prompts containing:
- **Context:** Truncated text from associated documents.
- **Event:** The target event to explain.
- **Options:** Four possible multiple-choice options (A, B, C, D).
- **Answer:** The golden label(s) used for SFT training.

## Model and Training Details

- **Base Model:** `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **Max Sequence Length:** 2048
- **LoRA Configuration:** 
  - Rank (r): 16
  - Alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Hyperparameters:**
  - Max Steps: 300
  - Learning Rate: 2e-4
  - Batch Size per device: 2
  - Gradient Accumulation Steps: 4
  - Optimizer: adamw_8bit

## Evaluation Metric

The notebook includes a custom evaluation block for the Development Set. It parses the model's text generation using regular expressions to extract predicted options and scores them against the golden answer using the SemEval evaluation logic:
- **1.0 point**: Full Match (The predicted set exactly matches the golden set).
- **0.5 points**: Partial Match (The predicted set is a subset of the golden set).
- **0.0 points**: Incorrect (No overlap or predicted options contain incorrect choices).

## Output

After evaluation, the pipeline iterates over the blind Test Set to generate the final predictions. 
The outputs are formatted as lists of strings (e.g., `["A", "C"]`) and saved to:
`predictions.jsonl`

This file is formatted specifically for submission to the Codabench platform.

## How to Run

1. Open the provided `main.ipynb` in a Jupyter/Colab environment.
2. Ensure you are connected to a GPU runtime.
3. Execute the notebook cells sequentially. The script will handle downloading the dataset, training the model, outputting the evaluation metrics, and producing the `predictions.jsonl` file.
