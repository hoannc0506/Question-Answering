from transformers import AutoTokenizer, AutoModel

class TextEncoder:
    def __init__(self, pretrained_name, device="cuda"):
        self.model =  AutoModel.from_pretrained(pretrained_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.device = device

        print(f"freezing {pretrained_name} parameters")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        model_output = self.model(**encoded_input)
    
        return model_output.last_hidden_state[:, 0, :]