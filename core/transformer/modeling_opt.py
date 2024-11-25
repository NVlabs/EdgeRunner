'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

from typing import List, Optional, Tuple, Union

import math
import torch
from torch import nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from core.transformer.attention import attention

import kiui

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32)
        exponent = torch.exp(exponent / half_dim)
        exponent = exponent.unsqueeze(0).unsqueeze(0) # [1, 1, half_dim]
        self.register_buffer('exponent', exponent)


    def forward(self, timesteps):
        # timesteps: [B, L]
        emb = timesteps.float().unsqueeze(-1) * self.exponent
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, sample):
        # sample: [B, C]
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample

class ShapeOPTConfig(PretrainedConfig):

    model_type = "shape_opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50272,
        max_position_embeddings=2048,
        hidden_dim=1024,
        intermediate_dim=4096,
        num_hidden_layers=24,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=16,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        num_cond_tokens=257,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim # deepspeed compatibility
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        self.num_cond_tokens = num_cond_tokens


class OptFlashAttention2(nn.Module):

    def __init__(
        self,
        config: ShapeOPTConfig,
        return_cache: bool = False,
        is_causal: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = is_causal

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.return_cache = return_cache

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, _, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.return_cache:
            past_key_value = (key_states, value_states)
        
        query_length = query_states.shape[1]
        kv_length = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, kv_length, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, kv_length, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = attention(query_states, key_states, value_states, mask_q=attention_mask, mask_kv=attention_mask, dropout=attn_dropout, causal=self.is_causal)

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights_reshaped)

        if not output_attentions:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value

class OPTDecoderLayer(nn.Module):
    def __init__(self, config: ShapeOPTConfig, layer_id: int = -1):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.self_attn = OptFlashAttention2(config=config, return_cache=True)
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Self Attention
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class ShapeOPTPreTrainedModel(PreTrainedModel):
    config_class = ShapeOPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_flash_attn_2 = True

class ShapeOPTDecoder(ShapeOPTPreTrainedModel):
    config_class = ShapeOPTConfig
    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.config = config
        
        self.embd = nn.Embedding(config.vocab_size, config.hidden_dim, config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_dim)

        self.layers = nn.ModuleList([OPTDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
       
        self.gradient_checkpointing = False
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None: # input_embeds asserts to be None
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            inputs_embeds = self.embd(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        
        # 2d mask is passed through the layers
        causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device) if attention_mask is None else attention_mask
       
        # position embeddings
        position_ids = torch.arange(past_key_values_length, mask_seq_length, dtype=torch.long, device=inputs_embeds.device)
        pos_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.config.layerdrop:
                    continue
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShapeOPT(ShapeOPTPreTrainedModel, GenerationMixin):
    config_class = ShapeOPTConfig
    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.config = config
        self.model = ShapeOPTDecoder(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # state machine for next-token-type during generate (inference)
        self.ntt_state = {
            'counter': None,
        }

        # correctly initialize the weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.num_hidden_layers))

    def _init_weights(self, module):
        # return
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # override, we are not using it anyway
    def tie_weights(self):
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
      
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous() # [B, N, C]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # cross entropy
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, num_tokens=None, **kwargs
    ):

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            B = inputs_embeds.shape[0]
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            B = input_ids.shape[0]
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "num_tokens": num_tokens, # extra input
            "attention_mask": attention_mask,
        })

        return model_inputs