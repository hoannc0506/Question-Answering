from pydantic import BaseModel

class QAConfig(BaseModel):
    # Model
    MODEL_NAME: str = "distilbert-base-uncased"
    
    # document max tokens
    MAX_LENGTH: int = 384
    STRIDE: int = 128

    # setup Dataset
    DATASET_NAME: str = "squad_v2"
    NUM_PROC: int = 8
    
    # Training parameters
    device: str = "cuda"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    fp16: bool = True
    eval_batch_size: int = 16
    num_train_epochs: int = 3
    save_total_limit: int = 1
    eval_steps: int = 1000
    ckpt_dir: str = "distilbert-finetuned-squadv2"

    # Evaluate
    N_BEST: int = 20 
    MAX_ANS_LENGTH: int = 30
