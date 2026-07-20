# Vietnamese Aspect-Based Sentiment Analysis (ABSA) using Instruction-Tuned Qwen2.5-7B

This repository contains the source code, data preprocessing pipelines, and evaluation scripts for our Natural Language Processing course project. We approach the Vietnamese Aspect-Based Sentiment Analysis (ABSA) task as a **Generative Sequence-to-Sequence** problem, fine-tuning the state-of-the-art **Qwen2.5-7B-Instruct** model using **QLoRA** on the **UIT-ViSD4SA** dataset.

## Project Overview

- **Objective**: Extract triplets of `(Aspect, Sentiment, Span)` from mobile phone reviews written in Vietnamese.
- **Dataset**: `UIT-ViSD4SA` (11,122 reviews, split into 70% Train, 10% Dev, 20% Test). It contains 10 target aspects (e.g., `BATTERY`, `SCREEN`, `PERFORMANCE`, `CAMERA`, etc.) and 3 sentiment polarities (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
- **Methodology**: Rather than treating ABSA as a token-level sequence labeling task (which struggles with overlapping/implicit aspects), we model it as **Conditional Text Generation** (Sequence-to-Sequence). We instruction-tune the LLM to output a structured list of JSON objects directly.
- **Training Acceleration**: Quantized Low-Rank Adaptation (**QLoRA** in 4-bit) via the **Unsloth** framework, enabling training on resource-constrained hardware (e.g., a single NVIDIA T4 GPU in Google Colab).
- **Post-Processing**: A custom hybrid **Neuro-Symbolic** framework integrating strict JSON parsing, RegEx fallback, and span-based semantic heuristic mapping to resolve LLM hallucinations and structural syntax errors.

---

## System Architecture

1. **Instruction Engineering & Data Preprocessing**:
   - Converts raw character-level span annotations into Alpaca-style instruction tuning prompts.
   - Outputs standardized JSON formats: `{"aspect": "...", "sentiment": "...", "span": "..."}`.
2. **Efficient Fine-Tuning**:
   - Quantized base model: `Qwen2.5-7B-Instruct` loaded in 4-bit.
   - LoRA target modules: Attention layers & MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Hyperparameters: `Rank = 16`, `Alpha = 16`, `Learning Rate = 2e-4`, `Batch Size = 8` (gradient accumulation).
3. **Neuro-Symbolic Post-Processing**:
   - **Strict JSON Parsing**: Attempts to parse the model output as standard JSON.
   - **Regex Fallback**: Uses regex to extract keys if the JSON syntax is broken.
   - **Rule-based Normalization**: Corrects typical mistakes (e.g., mapping "cam đẹp" from `DESIGN` back to `CAMERA` using semantic span checks).

---

## Experimental Results

We evaluated our model on three tasks: Aspect Extraction, Polarity Detection, and Strict Matching (which requires predicting both aspect and sentiment correctly).

Our generative approach outperforms the **XLM-RoBERTa Large** baseline from the original UIT-ViSD4SA paper (Nguyen et al., 2021) by a wide margin, especially in sentiment detection.

| Task | Baseline (XLM-RoBERTa Large) F1-Macro (%) | Our Model (Qwen2.5-7B) F1-Micro (%) | Our Model (Qwen2.5-7B) F1-Macro (%) |
| :--- | :---: | :---: | :---: |
| **Aspect Extraction** | 62.76 | **74.79** | **71.02** |
| **Polarity Detection** | 49.77 | **84.36** | **71.86** |
| **Strict Matching** | 45.70 | **65.73** | **46.83** |

---

## Repository Structure

```text
├── data_llm/             # Processed Alpaca-format instruction-tuning datasets
├── raw_data/             # Original UIT-ViSD4SA dataset splits (.jsonl)
├── trained_model/        # Saved LoRA adapter weights after fine-tuning
├── docs/
│   └── NLP_report.pdf    # Detailed research and project report (Vietnamese)
├── Preprocessing.ipynb   # Notebook for data preparation and Alpaca format conversion
├── Model.ipynb           # Notebook for QLoRA training and metrics calculation
├── DEMO.ipynb            # Notebook for local inference and interactive testing
├── requirements.txt      # Required dependencies
└── .gitattributes
```

---

## How to Run

### 1. Data
The raw dataset (`raw_data/`) and preprocessed instruction-tuning data (`data_llm/`) are already included in this repository. No separate download is required.

### 2. Install Dependencies
Ensure you have Python 3.10+ and a GPU-enabled environment. Install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Preprocessing
Run the `Preprocessing.ipynb` notebook to transform the raw `.jsonl` files in `raw_data/` into structured Alpaca-style instruction records stored in `data_llm/`.

### 4. Training & Evaluation
Open `Model.ipynb` in Google Colab (or any GPU server):
- Follow the notebook cells to install the `unsloth` framework.
- Load the model, attach the LoRA adapters, and train.
- Run the evaluation cell (`calculate_metrics_optimized`) to compute the final micro and macro F1 scores using our post-processing logic.

### 5. Inference Demo
Open `DEMO.ipynb` to load the saved LoRA adapter from `trained_model/` and run interactive inference on custom Vietnamese review sentences without retraining.
