## To create and activate the conda environment:

```
conda env create -f capstone-env.yml
conda activate capstone
```

## To download PDFs:

```
python download-arxiv-papers.py
```

## To create embedding and store them on the DB:

```
python load_pdfs.py
```

## To init the chat UI with the RAG:

1. Make sure you have the embeddings
2. Replace the "..." on login(token="...") from `model.py` file with your huggingface token
3. Run:

```
streamlit run user-interface.py
```
