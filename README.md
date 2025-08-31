# LLM_from_scratch


## Prerequisites:
1. Python 3.10.6
2. Nvidia GPU

## Setup:

1. Create venv
```
py -3.10 -m venv cuda
```

2. Activate venv
```
cuda activate
```

3. Install libs
```
pip install matplotlib numpy pylzma ipkernel jupyter
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

4. Install a new kernel for Jupyter Notebook 
```
python -m ipykernel install --user --name=cuda --display-name "cuda-gpt"
```

5. Start Jupyter Notebook
```
jupyter notebook
```




https://github.com/user-attachments/assets/2dc55b97-d442-4d0b-bce1-b819160e41c6




---

## 🏗️ How to Reproduce / Extend

1. **Download & Prepare Dataset**  
   - Download OpenWebText2 (~27 GB) from https://openwebtext2.readthedocs.io/en/latest/ and unpack all `.jsonl.zst` files into `./openwebtext2/`.

2. **Extract & Tokenize Data**  
   - Open and run `data-extract-v10.ipynb`, which:  
     - Streams and decompresses the `.zst` files.  
     - Filters for English-language texts.  
     - Tokenizes using tiktoken tokenizer.  
     - Outputs `output_v10/encoded_data/encoded_output_v10_accuracy.npy` (~107 GB).

3. **Train Base GPT Model**  
   - Run `gpt-v14.ipynb` end-to-end to:  
     - Configure hyperparameters (depth, heads, learning rate schedule, etc.).  
     - Execute the training loop, logging train/validation losses.  
     - Save the checkpoint (e.g., `output_v14\pre_training\run_<unix_timestamp>/gpt_v14_model.pt`).

4. **Fine-Tune for Classification**  
   - Use `finetuning-classification-v1.ipynb` to adapt the pre-trained checkpoint for a binary classification task (e.g., spam vs. ham).

5. **Fine-Tune for Instruction-Following**  
   - Use `finetuning-instruction-answer-v4.ipynb` to train on instruction–response pairs and improve the model’s conversational ability.

6. **Evaluate Fine-Tuned Models**  
   - Open `evaluate-finetuned-llm.ipynb` to:  
     - Compute performance metrics (accuracy, loss) on held-out data.  
     - Compare your fine-tuned outputs against the Ollama LLaMA 3.2 3B reference baseline.

---

## 📈 Example Results

> *Sample generation from `gpt-v14.ipynb` after 17 epochs:*

```text
Prompt: "I like apple juice - I drink it"
→ "I like apple juice - I drink it for about 30 minutes or even 1/20 minutes. In fact it was so common, so if it would be melted and the calories for me. And for me it was a pretty cool product"
```

> *Sample generation from `gpt-v17.ipynb` after 100k steps:*

Note: gpt-v17 has been trained on 100BT subset of fineweb-edu dataset using infinite streaming form *.paraquet files.
```text
Prompt: "I like apple juice - I drink it"
→ "I like apple juice - I drink it daily to keep track of blood pressure, especially if you are at home or if I get sick and then drink to check my urine to be sure I am up to my normal range."
```


## 🚧 Next Steps
🎛️ Wrap notebooks into CLI scripts (train.py, generate.py).

🌐 Build a small Gradio/Streamlit demo for live inference.

📊 Integrate Weights & Biases or TensorBoard for experiment tracking.

🇵🇱 Experiment with Polish-language fine-tuning on local corpora.



