from transformers import DistilBertTokenizer, DistilBertModel
import torch 

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = torch.tensor(
    tokenizer.encode('Hello, my dog is cute', add_special_token=True))\
        .unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]
print(last_hidden_states.shape)
