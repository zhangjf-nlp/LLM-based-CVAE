from typing import Dict, Any, Tuple, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaPreTrainedModel, LlamaModel, LlamaAttention, LlamaForCausalLM, LlamaRMSNorm, LlamaDecoderLayer
from transformers import DynamicCache, Cache

from utils import load_sharded_prefix_checkpoint
from utils_latent import sampling, compute_kl_penalty, log_pdf, exp_mean_log, compute_mi


# CVAE structure outline
# latent_encoder(posterior_input_ids, posterior_attention_mask) -> latent_mean_logvar
# latent_decoder(latent_mean_logvar, input_ids, attention_mask, labels) -> loss, lm_logits

# modules API
# latent_encoder:
#     posterior_input_ids: torch.LongTensor = None,
#     posterior_attention_mask: torch.FloatTensor = None,
# 	  ->
# 	  (mean, logvar)
# latent_decoder:
#     latent_mean_logvar: Tuple[torch.Tensor] = None,
#     input_ids: torch.LongTensor = None,
#     attention_mask: torch.FloatTensor = None,
#     labels: torch.LongTensor = None,
#     return_dict: bool = None,
#     scalar_loss: bool = True,
#     ->
#     (loss, lm_logits, (others))


class LlamaCVAEConfig(LlamaConfig):
    def __init__(
        self,
        # for model structure
        num_q = 4,
        dim_z = 32,
        num_p = 4,
        num_latent_encoder_layers = 2,
        # for training details
        frozen_stage = 2, # 0: full parameter; 1: frozen backbone llm; 2: frozen adapter attn and encoder embed
        vae_type = "DG-VAE",
        add_skip_connection = 0,
        add_gskip_connection = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # for model structure
        self.num_q = num_q
        self.dim_z = dim_z
        self.num_p = num_p
        self.num_latent_encoder_layers = num_latent_encoder_layers
        # for training details
        self.frozen_stage = frozen_stage
        self.vae_type = vae_type
        self.add_skip_connection = add_skip_connection
        self.add_gskip_connection = add_gskip_connection
        # others
        self.pad_token_id = self.eos_token_id


class LlamaForLatentEncoder(LlamaPreTrainedModel):
    config_class = LlamaCVAEConfig
    main_input_name = "posterior_input_ids"
    is_parallelizable = False

    def __init__(self, config: LlamaCVAEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_latent_queries = nn.Embedding(config.num_q, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_latent_encoder_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.h2z = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.dim_z * 2 // config.num_q)
        )
        self.embed_tokens.requires_grad_(config.frozen_stage<2)
        self.post_init()

    def load_pretrained(self, pretrained_model: LlamaForCausalLM):
        self.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
        for i in range(self.config.num_latent_encoder_layers):
            self.layers[i].load_state_dict(pretrained_model.model.layers[i].state_dict())
        self.init_embed_latent_queries()
        self.standardize()

    def init_embed_latent_queries(self):
        max_indice = min(10000, self.config.vocab_size)
        max_indice = max_indice - max_indice % self.config.num_q
        copied_weight = self.embed_tokens.weight.data[:max_indice, :]
        copied_weight = copied_weight.view(
            max_indice // self.config.num_q,
            self.config.num_q,
            self.config.hidden_size
        ).mean(dim=0)
        assert self.embed_latent_queries.weight.data.shape == copied_weight.shape, \
            "Shape mismatch: embed_latent_queries.weight.data.shape {} vs copied_weight.shape {}".format(
                self.embed_latent_queries.weight.data.shape, copied_weight.shape)
        self.embed_latent_queries.weight.data.copy_(copied_weight)

    def standardize(self):
        # regularize the output latent distribution to standard gaussian
        self.h2z[-1].weight.data.zero_()
        self.h2z[-1].bias.data.zero_()

    def forward(
        self,
        posterior_input_ids: torch.LongTensor = None,
        posterior_attention_mask: torch.FloatTensor = None,
    ):
        batch_size, seq_len = posterior_input_ids.shape

        posterior_embeds = self.embed_tokens(posterior_input_ids)
        latent_query_embeds = self.embed_latent_queries(torch.arange(self.config.num_q).to(self.device))
        inputs_embeds = torch.cat([posterior_embeds, latent_query_embeds[None, :, :].expand(batch_size, -1, -1)], dim=1)

        queries_attention_mask = posterior_attention_mask.new_ones(size=[batch_size, self.config.num_q])
        attention_mask = torch.cat([posterior_attention_mask, queries_attention_mask], dim=-1)
        sequence_length, dtype, device = inputs_embeds.shape[1], inputs_embeds.dtype, inputs_embeds.device
        causal_mask = torch.triu(torch.full(
            size=(sequence_length, sequence_length),
            fill_value=torch.finfo(dtype).min,
            dtype=dtype, device=device
        ), diagonal=1)
        attention_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1) + attention_mask[:, None, None, :]
        position_ids = torch.arange(sequence_length)[None, :].expand(batch_size, -1).to(device=device)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        latent = self.h2z(self.norm(hidden_states[:, -self.config.num_q:, :]))
        mean = latent[:, :, :latent.shape[-1] // 2].reshape(batch_size, self.config.dim_z)
        logvar = latent[:, :, latent.shape[-1] // 2:].reshape(batch_size, self.config.dim_z)
        return mean, logvar


class LlamaForLatentAdapter(LlamaPreTrainedModel):
    # latent adapter
    config_class = LlamaCVAEConfig
    main_input_name = "latent_mean_logvar"
    is_parallelizable = False

    def __init__(self, config: LlamaCVAEConfig, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.z2h = nn.Sequential(
            nn.Linear(config.dim_z, int(config.hidden_size ** 0.5)),
            nn.GELU(),
            nn.Linear(int(config.hidden_size ** 0.5), config.hidden_size * config.num_p),
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.self_attn.requires_grad_(config.frozen_stage<2)

    def load_pretrained(self, pretrained_model: LlamaForCausalLM):
        self.self_attn.load_state_dict(pretrained_model.model.layers[self.layer_idx].self_attn.state_dict())

    def forward(
        self,
        zs: torch.FloatTensor,
        past_key_values: DynamicCache,
    ):
        batch_size = zs.shape[0]
        latent_hidden_states = self.z2h(zs).view(batch_size, self.config.num_p, self.config.hidden_size)
        latent_hidden_states = self.norm(latent_hidden_states)
        # past_key_value or past_key_values, this must be a typo in llama source code
        attn_output, attn_weights, past_key_values = self.self_attn(
            hidden_states=latent_hidden_states,
            position_ids=torch.arange(self.config.num_p)[None, :].expand(batch_size, -1).to(self.device),
            past_key_value=past_key_values,
            use_cache=True,
        )
        return past_key_values


class LlamaForLatentDecoder(LlamaPreTrainedModel):
    config_class = LlamaCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: LlamaCVAEConfig):
        super().__init__(config)
        self.latent_adapters = nn.ModuleList([
            LlamaForLatentAdapter(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        if config.add_skip_connection == 1 or config.add_gskip_connection == 1:
            self.skip_connection = nn.Sequential(
                nn.Linear(config.dim_z, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        elif config.add_skip_connection == 2 or config.add_gskip_connection == 2:
            self.skip_connection = nn.Sequential(
                nn.Linear(config.dim_z, config.hidden_size),
            )

        if config.vae_type == "EMB-VAE" or config.vae_type == "EMB-DG-VAE":
            self.z2emb = nn.Sequential(
                nn.Linear(config.dim_z, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.requires_grad_(config.frozen_stage<1)
        self.lm_head.requires_grad_(config.frozen_stage<1)

    def load_pretrained(self, pretrained_model: LlamaForCausalLM):
        for layer_idx in range(self.config.num_hidden_layers):
            self.latent_adapters[layer_idx].load_pretrained(pretrained_model)
        self.model.load_state_dict(pretrained_model.model.state_dict())
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

    def convert_latent_into_skip_residue(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        latent_sampling_times: int = 32,
    ):
        mean, logvar = latent_mean_logvar
        zs = sampling(mean, logvar, latent_sampling_times) # [bs, dim_z, nz]
        zs = zs.to(self.dtype)
        skip_residue = self.skip_connection(zs.transpose(1, 2)).mean(dim=1, keepdim=True) # [bs, 1, hidden_size]
        if self.config.add_gskip_connection:
            skip_residue = (skip_residue - skip_residue.detach()) # equals to 0 but has gradient
        return skip_residue

    def convert_latent_into_emb(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        latent_sampling: bool = True,
    ):
        mean, logvar = latent_mean_logvar
        if latent_sampling:
            zs = sampling(mean, logvar).squeeze(-1) # [batch_size, dim_z]
        else:
            zs = mean
        emb = self.z2emb(zs).unsqueeze(1) # [batch_size, 1, hidden_size]
        return emb

    def convert_latent_into_kv_cache(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        latent_sampling: bool = True,
    ):
        mean, logvar = latent_mean_logvar
        if latent_sampling:
            zs = sampling(mean, logvar, 1).squeeze(-1)
        else:
            zs = mean
        zs = zs.to(self.dtype)
        past_key_values = DynamicCache()
        for layer_idx in range(self.config.num_hidden_layers):
            past_key_values = self.latent_adapters[layer_idx].forward(
                zs=zs, past_key_values=past_key_values
            )
        return past_key_values

    def forward(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        past_skip_residue: Tuple[torch.FloatTensor] = None,
        past_emb: Tuple[torch.FloatTensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None, # specified for generation loop
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        scalar_loss: bool = True,
        latent_sampling: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]

        if past_key_values is None and latent_mean_logvar is not None:
            # latent -> latent_kv_cache
            past_key_values = self.convert_latent_into_kv_cache(
                latent_mean_logvar = latent_mean_logvar,
                latent_sampling = latent_sampling,
            )
            latent_attention_mask = attention_mask.new_ones(size=[batch_size, self.config.num_p])
            attention_mask = torch.cat([latent_attention_mask, attention_mask], dim=1)

        if self.config.vae_type == "EMB-VAE" or self.config.vae_type == "EMB-DG-VAE":
            if past_emb is not None:
                inputs_embeds = self.model.embed_tokens(input_ids) + past_emb[0]
                input_ids = None
            else:
                assert latent_mean_logvar is not None
                emb = self.convert_latent_into_emb(latent_mean_logvar = latent_mean_logvar)
                inputs_embeds = self.model.embed_tokens(input_ids) + emb
                input_ids = None

        # latent_kv_cache + target_inputs -> labels
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        if self.config.add_skip_connection or self.config.add_gskip_connection:
            if past_skip_residue is None:
                skip_residue = self.convert_latent_into_skip_residue(latent_mean_logvar)
            else:
                skip_residue = past_skip_residue[0]
            hidden_states = hidden_states + skip_residue.repeat(1, hidden_states.shape[1], 1)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Negative Log-Likelihood
            loss = F.cross_entropy(
                input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
                target=labels[:, 1:].contiguous(),
                reduction='none'
            ).sum(dim=-1)
            if scalar_loss:
                loss = loss.mean()

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "past_skip_residue": kwargs.get("past_skip_residue", None),
                "past_emb": kwargs.get("past_emb", None),
            }
        )
        return model_inputs


class LlamaForCVAE(LlamaPreTrainedModel):
    config_class = LlamaCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: LlamaCVAEConfig):
        super().__init__(config)
        self.latent_encoder = LlamaForLatentEncoder(config)
        self.latent_decoder = LlamaForLatentDecoder(config)
        if config.vae_type == "BOW-VAE":
            self.z2logits = nn.Sequential(
                nn.Linear(config.dim_z, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            )

    def load_pretrained(self, pretrained_model_name_or_path, device_map=None, torch_dtype=None):
        pretrained_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        self.latent_encoder.load_pretrained(pretrained_model)
        self.latent_decoder.load_pretrained(pretrained_model)
        if self.config.vae_type == "BOW-VAE":
            self.z2logits[-1].load_state_dict(pretrained_model.lm_head.state_dict())

    def update_config(self, new_config):
        attr_mutable = ["frozen_stage", "add_contra_loss", "add_skip_connection", "add_gskip_connection"]
        fail = False
        for attr in new_config:
            if attr in attr_mutable:
                self.config.update({attr: new_config[attr]})
            else:
                self_value, new_value = getattr(self.config, attr), new_config[attr]
                if not self_value == new_value:
                    print(f"immutable attr {attr} not match: {self_value} != {new_value}")
                    fail = True
        self.latent_encoder.embed_tokens.requires_grad_(self.config.frozen_stage<2)
        for adapter in self.latent_decoder.latent_adapters:
            adapter.self_attn.requires_grad_(self.config.frozen_stage<2)
        self.latent_decoder.model.requires_grad_(self.config.frozen_stage<1)
        self.latent_decoder.lm_head.requires_grad_(self.config.frozen_stage<1)
        assert not fail

    # for latent_encoder
    #     posterior_input_ids: torch.LongTensor = None,
    #     posterior_attention_mask: torch.FloatTensor = None,
    # for latent_decoder
    #     context_input_ids: torch.LongTensor = None,
    #     context_attention_mask: torch.FloatTensor = None,
    #     target_input_ids: torch.LongTensor = None,
    #     target_attention_mask: torch.FloatTensor = None,
    #     labels: torch.LongTensor = None,
    def forward(
        self,
        posterior_input_ids: torch.LongTensor = None,
        posterior_attention_mask: torch.FloatTensor = None,
        group_posterior_input_ids: torch.LongTensor = None,
        group_posterior_attention_mask: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        reference_nlls: torch.FloatTensor = None,
        return_loss = True, # this is not used but necessary to ensure Trainer.can_return_loss and including eval_loss in evaluation
    ):
        batch_size = labels.shape[0]
        latent_mean_logvar = self.latent_encoder(
            posterior_input_ids = posterior_input_ids,
            posterior_attention_mask = posterior_attention_mask
        )
        mean, logvar = latent_mean_logvar
        if self.config.vae_type == "BOW-VAE":
            loss_kld = compute_kl_penalty(mean, logvar, vae_type="Vanilla")
            latent_logits = self.z2logits(sampling(mean, logvar, n_samples=labels.shape[1]).transpose(1, 2))
            loss_bow = F.cross_entropy(
                input=latent_logits[:, :-1, :].contiguous().transpose(1, 2),
                target=labels[:, 1:].contiguous(),
                reduction='mean'
            )
            loss_kld = loss_kld + loss_bow
        else:
            loss_kld = compute_kl_penalty(mean, logvar, vae_type=self.config.vae_type,
                                          train_steps=getattr(self.config, "train_steps", 1000),
                                          eval_steps=getattr(self.config, "eval_steps", 1000))
        loss_lm = self.latent_decoder(
            latent_mean_logvar = latent_mean_logvar,
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )[0]
        mi = compute_mi(mean, logvar)
        loss = loss_kld + loss_lm
        return loss, loss_kld, loss_lm, mi

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        posterior_input_ids: Optional[torch.Tensor] = None,
        posterior_attention_mask: Optional[torch.Tensor] = None,
        latent_sampling: Optional[bool] = None,
        latent_mean_logvar: Tuple[torch.Tensor] = None,
        encode_input_ids_into_latent: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        batch_size = input_ids.shape[0]

        if latent_mean_logvar is None:
            if encode_input_ids_into_latent:
                latent_mean_logvar = self.latent_encoder(
                    posterior_input_ids = input_ids,
                    posterior_attention_mask = attention_mask
                )
            elif posterior_input_ids is not None:
                latent_mean_logvar = self.latent_encoder(
                    posterior_input_ids = posterior_input_ids,
                    posterior_attention_mask = posterior_attention_mask
                )
            else:
                mean = torch.zeros([batch_size, self.config.dim_z], device=self.device, dtype=self.dtype)
                logvar = torch.zeros([batch_size, self.config.dim_z], device=self.device, dtype=self.dtype)
                latent_mean_logvar = (mean, logvar)

        kv_expand_size = kwargs.get("num_return_sequences", 1) * kwargs.get("num_beams", 1)
        if kv_expand_size > 1:
            latent_mean_logvar = (
                latent_mean_logvar[0].repeat_interleave(kv_expand_size, dim=0),
                latent_mean_logvar[1].repeat_interleave(kv_expand_size, dim=0)
            )

        if self.config.add_skip_connection or self.config.add_gskip_connection:
            skip_residue = self.latent_decoder.convert_latent_into_skip_residue(latent_mean_logvar=latent_mean_logvar)
            past_skip_residue = (skip_residue,)
        else:
            past_skip_residue = None

        past_key_values = self.latent_decoder.convert_latent_into_kv_cache(
            latent_mean_logvar = latent_mean_logvar,
            latent_sampling = latent_sampling,
        )
        if self.config.vae_type == "EMB-VAE" or self.config.vae_type == "EMB-DG-VAE":
            past_emb = self.latent_decoder.convert_latent_into_emb(
                latent_mean_logvar = latent_mean_logvar,
                latent_sampling = latent_sampling,
            )
            past_emb = (past_emb,)
        else:
            past_emb = None
        attention_mask = torch.cat([attention_mask.new_ones(size=[batch_size, self.config.num_p]), attention_mask], dim=1)
        input_ids = torch.cat([input_ids.new_zeros(size=[batch_size, self.config.num_p]), input_ids], dim=1)

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["past_key_values"] = past_key_values
        kwargs["past_skip_residue"] = past_skip_residue
        kwargs["past_emb"] = past_emb

        return self.latent_decoder.generate(
            inputs = inputs,
            generation_config = generation_config,
            logits_processor = logits_processor,
            stopping_criteria = stopping_criteria,
            **kwargs
        )[:, self.config.num_p:]


class LlamaForLatentDPO(LlamaPreTrainedModel):
    config_class = LlamaCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: LlamaCVAEConfig, beta=0.1):
        super().__init__(config)
        self.beta = beta
        self.posterior_latent_encoder = LlamaForLatentEncoder(config)
        self.posterior_latent_encoder.requires_grad_(False)
        self.policy_latent_encoder = LlamaForLatentEncoder(config)
        self.policy_latent_encoder.standardize()

    def load_cvae_checkpoint(self, pretrained_model_name_or_path):
        load_sharded_prefix_checkpoint(
            self.posterior_latent_encoder,
            pretrained_model_name_or_path,
            prefix="latent_encoder", strict=True, prefer_safe=True
        )
        self.policy_latent_encoder.load_state_dict(self.posterior_latent_encoder.state_dict())
        self.policy_latent_encoder.standardize()

    def forward(
        self,
        input_ids,
        attention_mask,
        chosen_ids,
        chosen_attention_mask,
        rejected_ids,
        rejected_attention_mask,
        n_samples=16,
        return_loss=True,
    ):
        # step1. construct latent preference samples (x,z_w,z_l)
        with torch.no_grad():
            chosen_mean, chosen_logvar = self.posterior_latent_encoder(
                posterior_input_ids=chosen_ids,
                posterior_attention_mask=chosen_attention_mask
            )
            rejected_mean, rejected_logvar = self.posterior_latent_encoder(
                posterior_input_ids=rejected_ids,
                posterior_attention_mask=rejected_attention_mask
            )
        policy_mean, policy_logvar = self.policy_latent_encoder(input_ids, attention_mask)

        with torch.no_grad():
            policy_zs = sampling(policy_mean, policy_logvar, n_samples=n_samples)
            chosen_zs = sampling(chosen_mean, chosen_logvar, n_samples=n_samples)
            rejected_zs = sampling(rejected_mean, rejected_logvar, n_samples=n_samples)
            zs = torch.cat([policy_zs, chosen_zs, rejected_zs], dim=-1)
            # [batch_size, dim_z, n_samples]
            n_samples = zs.shape[-1]

            # -log q(z|x,y_w)
            zs_nlls_given_chosen = -log_pdf(chosen_mean, chosen_logvar, zs).sum(dim=1)
            # -log q(z|x,y_l)
            zs_nlls_given_rejected = -log_pdf(rejected_mean, rejected_logvar, zs).sum(dim=1)
            # log q(z|x,y_w)/q(z|x,y_l), greater is better
            scores_zs = zs_nlls_given_rejected - zs_nlls_given_chosen
            # batch_size, n_samples
            _, chosen_indices = torch.topk(scores_zs, n_samples//4, dim=-1, largest=True)
            _, rejected_indices = torch.topk(scores_zs, n_samples//4, dim=-1, largest=False)
            chosen_zs_ = torch.gather(zs, -1, chosen_indices.unsqueeze(1).repeat(1,self.config.dim_z,1))
            rejected_zs_ = torch.gather(zs, -1, rejected_indices.unsqueeze(1).repeat(1,self.config.dim_z,1))
            # [batch_size, dim_z, n_samples//4]

        # step2. perform DPO on p(z|x) with  (x,z_w,z_l)
        refer_mean, refer_logvar = torch.zeros_like(policy_mean), torch.zeros_like(policy_logvar)
        policy_chosen_nlls = -log_pdf(policy_mean, policy_logvar, chosen_zs_).sum(dim=1)
        policy_rejected_nlls = -log_pdf(policy_mean, policy_logvar, rejected_zs_).sum(dim=1)
        with torch.no_grad():
            refer_chosen_nlls = -log_pdf(refer_mean, refer_logvar, chosen_zs_).sum(dim=1)
            refer_rejected_nlls = -log_pdf(refer_mean, refer_logvar, rejected_zs_).sum(dim=1)
        chosen_reward = refer_chosen_nlls.detach() - policy_chosen_nlls
        rejected_reward = refer_rejected_nlls.detach() - policy_rejected_nlls

        loss = -F.logsigmoid((chosen_reward - rejected_reward) * self.beta)
        acc = (chosen_reward > rejected_reward).float()
        return loss.mean(), acc.mean()