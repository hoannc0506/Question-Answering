# Open domain question answering

## Dataset
- Stanford Question Answering Dataset (SQuAD): [Squadv2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)


## Methodology
![e2e QA pipeline](docs/e2eQA.png)

- Reader: [qa-squadv2-distilbert](https://huggingface.co/hoannc0506/qa-squadv2-distilbert)
- Retriever: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

## Results
| Model | F1 |
| ----------- | ----------- |
| qa-squadv2-distilbert | 52.42 |


## Uses
- Visualize data preprocessing: `DatasetVisualize.ipynb`
- Test Retriever: `Retriever.ipynb`
- Evaluate Reader: `Evaluate.ipynb`
- How to train reader:
    - Modify `config.py`
    - Run `python train_reader.py`


## To-Do
- [ ] Evaluate with LLM
- [ ] LLM + FAISS
- [ ] LLM + RAG
