import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import Dataset
from config import QAConfig

class SquadDataProcessor:
    def __init__(self, cfg: QAConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME)
        
    def process_data(self, raw_dataset, data_type):
        print(f"Processing {data_type} data")
        if data_type == "train":
            dataset = raw_dataset.map(
                self.__preprocess_training_examples,
                batched=True,
                desc=f"Tokenizing {data_type} data",
                num_proc=self.cfg.NUM_PROC,
                remove_columns = raw_dataset.column_names
            )
        else:
            dataset = raw_dataset.map(
                self.__preprocess_validation_examples,
                batched=True,
                desc=f"Tokenizing {data_type} data",
                num_proc=self.cfg.NUM_PROC, 
                remove_columns = raw_dataset.column_names
            )
        return dataset

    def __preprocess_training_examples(self, examples):
        '''
            preprocess training data per batch
        '''
        # Preprocess batch questions
        questions = [q.strip() for q in examples["question"]]
    
        # Tokenize inputs
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.cfg.MAX_LENGTH,
            truncation="only_second",
            stride=self.cfg.STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    
        # Extract offset mappings and remove from inputs
        offset_mapping = inputs.pop("offset_mapping")
    
        # Extract sample mapping and remove from inputs
        sample_map = inputs.pop("overflow_to_sample_mapping")
    
        # Extract answers
        answers = examples["answers"]
    
        # Initialize start and end positions
        start_positions = []
        end_positions = []
    
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            sequence_ids = inputs.sequence_ids(i)
    
            # Find context start and end
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
    
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
    
            # Get answer
            answer = answers[sample_idx]
    
            if len(answer['text']) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Get start and end character positions
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
    
                # Check if answer spans are within context
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Find start position
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)
        
                    # Find end position
                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
    
        # Update inputs with start and end positions
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
    
        return inputs
    
    def __preprocess_validation_examples(self, examples):
        '''
            preprocess validation data per batch
        '''
        # Preprocess batch questions
        questions = [q.strip() for q in examples["question"]]
    
        # Tokenize inputs
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.cfg.MAX_LENGTH,
            truncation="only_second",
            stride=self.cfg.STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    
        # Extract offset mappings and remove from inputs
        # offset_mapping = inputs.pop("offset_mapping")
        example_ids = []
    
        # Extract sample mapping and remove from inputs
        sample_map = inputs.pop("overflow_to_sample_mapping")
    
        # Modify answer offset
        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])
            
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
    
            # remove unuse offset
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None \
                    for k, o in enumerate(offset)
            ]
    
        # Update inputs with start and end positions
        inputs["example_id"] = example_ids
        return inputs

if __name__ == "__main__":
    cfg = QAConfig()
    # raw_train_dataset = load_dataset(cfg.DATASET_NAME, split ="train", num_proc=cfg.NUM_PROC)
    raw_val_dataset = load_dataset(cfg.DATASET_NAME, split ="validation", num_proc=cfg.NUM_PROC)
    data_processor = SquadDataProcessor(cfg)
    # train_data = data_processor.process_data(raw_train_dataset, data_type="train")
    val_data = data_processor.process_data(raw_val_dataset, data_type="validation")
    # print(len(raw_train_dataset), len(train_data))
    print(len(raw_val_dataset), len(val_data))