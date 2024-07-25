# Question Answering

## Datasets
- Open Domain: [Squadv2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)

## Pipelines and models
### Open Domain Question Answeing
| Search Pipeline | Extractive QA model |
| ----------- | ----------- |
| ![e2e QA pipeline](docs/e2eQA.png) | ![e2e QA pipeline](docs/extractive_approach.png) |

- Dataset: [Squadv2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)
- Reader: [qa-squadv2-distilbert](https://huggingface.co/hoannc0506/qa-squadv2-distilbert)
- Retriever: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Visualize data preprocessing: `DatasetVisualize.ipynb`
- Test Retriever: `Retriever.ipynb`
- Evaluate Reader: `Evaluate.ipynb`
- Train extractive reader:
    - Modify `config.py`
    - Run `python train_reader.py`

## Referneces
1. [Question answering - HF course](https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt)