import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
import evaluate

device = torch.device("cuda:1")

# setup config
MODEL_NAME = "distilbert-base-uncased"
# document max tokens
MAX_LENGTH = 384 
STRIDE = 128 

# setup Dataset
DATASET_NAME = "squad_v2"
raw_dataset = load_dataset(DATASET_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_training_examples(examples):
    # Preprocess question
    questions = [q.strip() for q in examples["question"]]

    # Tiến hành mã hóa thông tin đầu vào sử dụng tokenizer
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Trích xuất offset_mapping từ inputs và loại bỏ nó ra khỏi inputs
    offset_mapping = inputs.pop("offset_mapping")

    # Trích xuất sample map từ inputs và loại bỏ nó ra khỏi inputs
    sample_map = inputs.pop("overflow_to_sample_mapping")

    # Trích xuất thông tin về câu trả lời (answers) từ examples
    answers = examples["answers"]

    # Khởi tạo danh sách các vị trí bắt đầu và kết thúc câu trả lời
    start_positions = []
    end_positions = []

    for i in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        sequence_ids = inputs.sequence_ids[i]

        # get idex_context_start and index_context_end
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # Trích xuất thông tin về câu trả lời cho mẫu này
        answer = answers [ sample_idx ]
        
        if len( answer [’text ’]) == 0:
            start_positions . append (0)
            end_positions . append (0)
        else :
            # Xác định vị trí ký tự bắt đầu và kết thúc của câu trả lời
            # trong ngữ cảnh
            start_char = answer [" answer_start " ][0]
            end_char = answer [" answer_start " ][0] + len( answer [" text " ][0])
        
        # Nếu câu trả lời không nằm hoàn toàn trong ngữ cảnh ,
        # gán nhãn là (0, 0)
        if offset [ context_start ][0] > start_char or offset [ context_end ][1] < end_char :
            start_positions.append(0)
            end_positions.appendp(0)
        else:
            # Nếu không , gán vị trí bắt đầu và kết thúc dựa trên
            # vị trí của các mã thông tin
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx-1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx-=1

            end_positions.append(idx + 1)
            
    # Thêm thông tin vị trí bắt đầu và kết thúc vào inputs
    inputs [" start_positions "] = start_positions
    inputs [" end_positions "] = end_positions
    
    return inputs


