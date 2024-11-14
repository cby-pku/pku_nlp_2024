import numpy as np
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x))) + x

class RobertaWithAdapter(RobertaModel):
    def __init__(self, model_name_or_path, adapter_dim=64):
        super(RobertaWithAdapter, self).__init__(RobertaConfig.from_pretrained(model_name_or_path))
        self.roberta = RobertaModel.from_pretrained(model_name_or_path)
        self.adapter = Adapter(self.config.hidden_size, adapter_dim)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        adapted_output = self.adapter(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(adapted_output.view(-1, self.config.hidden_size), labels.view(-1))
            return loss
        
        return adapted_output