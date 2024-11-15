from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import torch

class AdapterModule(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super(AdapterModule, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        adapter_hidden = self.down_project(x)
        adapter_hidden = self.activation(adapter_hidden)
        adapter_hidden = self.up_project(adapter_hidden)
        return x + adapter_hidden  

class RobertaWithAdapter(RobertaModel):
    def __init__(self, config):
        super(RobertaWithAdapter, self).__init__(config)
        self.num_labels = config.num_labels 
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  
        self.loss_fct = nn.CrossEntropyLoss()
        # add adapter modules to each layer
        self.adapters = nn.ModuleList([AdapterModule(config.hidden_size) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = super(RobertaWithAdapter, self).forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        adapter_outputs = []
        for i, adapter in enumerate(self.adapters):
            adapter_outputs.append(adapter(hidden_states[i+1]))
        last_hidden_state = adapter_outputs[-1]
        logits = self.classifier(last_hidden_state[:, 0, :])  

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}