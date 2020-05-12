from transformers import DistilledBertModel

class DistilledClassifer(nn.module):

    def __init__(self, model):
        super(model).__init__()
        self.base = DistilledBertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(786, 1)

    def forward(self, seq, attn_marks):
        
        # use attention mask to avoid contribution from 
        # padded sequence members 
        ctx = self.base(seq, attention_mask=attn_mask) 
        class_rep = ctx[:, 0]
        logits = self.linear(class_rep)
