from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):

        # Prepare query, key, value matrix
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # each of them has shape (batch_size, seq_len, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]

        # NOTE change and process KV-cache
        if layer_past is not None: # NOTE get KV-cache from past_key_values and update key and value
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        # print(f"use_cache: {use_cache}")
        if use_cache:
            present = (key, value)
            # print(f"present: {present}")
        else:
            present = None

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, present) if use_cache else attn_output


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        residual = hidden_states
        # print(f"use_cache: {use_cache}")

        # self-attention (class `CustomizedGPT2AttentionWithFasterCache`)
        hidden_states = self.ln_1(hidden_states)
        
        # NOTE change and add KV-cache parameters
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        # residual connection
        if use_cache:
            attn_output, present = attn_outputs
        else:
            attn_output = attn_outputs
            present = None

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        # NOTE change 4 

        if use_cache:
            return (hidden_states, present)
        return hidden_states


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        # print(f"use_cache: {use_cache}")

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        
        # NOTE 根据是否有past_key_values调整position_ids
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            # NOTE the key and values we have stored in past_key_values
            # DEBUG 1 
            past_length = past_key_values[0][0].size(-2)
            # print(f"past_key_values: {past_key_values}")
            # past_key_values = tuple([past_key_values[i] for i in range(len(past_key_values))])
            
        # print(f"past_length: {past_length}")
        # NOTE 1: The first change of position_ids
        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        # NOTE Debug
        # position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]) # [batch_size, seq_len]
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # [batch_size, seq_len]
        
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds


        # Prepare Attention mask.
        # attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        # attention_mask = attention_mask[:, None, None, :]
        # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        # NOTE 2: The second change of attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            # 
            if past_length > 0:
                attention_mask = torch.cat([torch.ones(batch_size, past_length, device=device), attention_mask], dim=-1)
                
        attention_mask = attention_mask[:, None, None, :] if attention_mask is not None else None
        attention_mask = attention_mask.to(dtype=self.dtype)
        
        
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None

        # # Iterate over all GPT2 layer, i.e. `block`
        # NOTE Change: collect KV cache
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)): # self.h is a list of blocks
            # print(f"layer_past: {layer_past}")
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                hidden_states, present = outputs
                # print(f"present: {present}")
                presents = presents + (present,)
            else:
                hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if use_cache:
            return hidden_states, presents
        return hidden_states


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # NOTE change remember to pass the config with use_cache=True
        self.transformer = CustomizedGPT2Model(config)
        # print(f'Use cache: {config.use_cache}')
        self.use_cache = config.use_cache

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ):

        # print(f'Use cache: {use_cache}')
        if use_cache is None:
            use_cache = self.use_cache
        # print(f"past_key_values: {past_key_values}")
        # print(f'Use cache: {use_cache}')
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            hidden_states, presents = transformer_outputs
        else:
            hidden_states = transformer_outputs

        lm_logits = self.lm_head(hidden_states)
        # print(f"presents: {presents}")

        return {
            'logits': lm_logits,
            'past_key_values': presents if use_cache else None
        }