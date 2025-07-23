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


https://github.com/user-attachments/assets/6e35fa4d-d4a5-46bc-8711-a81d7c2fedd0

