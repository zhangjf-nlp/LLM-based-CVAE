from typing import Dict, Any

import torch
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.models.gpt2.modeling_gpt2 import *
from utils_latent import sampling, compute_kl_penalty

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


class GPT2CVAEConfig(GPT2Config):
    def __init__(
        self,
        # for model structure
        num_q = 4,
        dim_z = 32,
        num_p = 4,
        num_latent_encoder_layers = 2,
        # for training details
        frozen_pretrained = 1,
        use_standard_prior = 1,
        vae_type = "DG-VAE",
        add_contra_loss = 1,
        add_skip_connection = 0,
        add_ghost_skip_connection = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # for model structure
        self.num_q = num_q
        self.dim_z = dim_z
        self.num_p = num_p
        self.num_latent_encoder_layers = num_latent_encoder_layers
        # for training details
        self.frozen_pretrained = frozen_pretrained
        self.use_standard_prior = use_standard_prior
        self.vae_type = vae_type
        self.add_contra_loss = add_contra_loss
        self.add_skip_connection = add_skip_connection
        self.add_ghost_skip_connection = add_ghost_skip_connection
        # others
        self.pad_token_id = self.eos_token_id


class GPT2ForLatentEncoder(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "posterior_input_ids"
    is_parallelizable = False

    def __init__(self, config: GPT2CVAEConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.latent_queries = nn.Parameter(torch.Tensor(1, config.num_q, config.n_embd))
        self.latent_queries.data.normal_(mean=0.0, std=config.initializer_range)
        self.ln_input = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.blocks = nn.ModuleList([
            GPT2Block(config)
            for i in range(config.num_latent_encoder_layers)
        ])
        self.ln_output = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.h2z = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.dim_z * 2 // config.num_q)
        )

    def load_pretrained(self, pretrained_model: GPT2LMHeadModel):
        self.wte.load_state_dict(pretrained_model.transformer.wte.state_dict())
        for i in range(self.config.num_latent_encoder_layers):
            self.blocks[i].load_state_dict(pretrained_model.transformer.h[i].state_dict())
        self.standardize()

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

        input_embeds = self.wte(posterior_input_ids)
        input_embeds = torch.cat([input_embeds, self.latent_queries.repeat(batch_size, 1, 1)], dim=1)

        queries_attention_mask = posterior_attention_mask.new_ones(size=[batch_size, self.config.num_q])
        attention_mask = torch.cat([posterior_attention_mask, queries_attention_mask], dim=-1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.ln_input(input_embeds)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
            )[0]
        latent = self.h2z(self.ln_output(hidden_states[:, -self.config.num_q:, :]))
        mean = latent[:, :, :latent.shape[-1] // 2].reshape(batch_size, self.config.dim_z)
        logvar = latent[:, :, latent.shape[-1] // 2:].reshape(batch_size, self.config.dim_z)
        return mean, logvar


class GPT2ForLatentTranslator(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "latent_mean_logvar"
    is_parallelizable = False

    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.z2h = nn.Sequential(
            nn.Linear(config.dim_z, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd * config.num_p),
        )
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)

    def load_pretrained(self, pretrained_model: GPT2LMHeadModel):
        self.attn.load_state_dict(pretrained_model.transformer.h[self.layer_idx].attn.state_dict())

    def forward(
        self,
        zs: torch.FloatTensor
    ):
        batch_size = zs.shape[0]
        latent_hidden_states = self.z2h(zs).view(batch_size, self.config.num_p, self.config.n_embd)
        latent_hidden_states = self.ln(latent_hidden_states)
        present = self.attn(
            hidden_states=latent_hidden_states,
            use_cache=True,
        )[1]
        return present


class GPT2ForLatentDecoder(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPT2CVAEConfig):
        super().__init__(config)
        self.latent_translators = nn.ModuleList([
            GPT2ForLatentTranslator(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])
        if config.add_skip_connection or config.add_ghost_skip_connection:
            self.skip_connection = nn.Sequential(
                nn.Linear(self.config.dim_z, self.config.n_embd),
                nn.GELU(),
                nn.Linear(self.config.n_embd, self.config.n_embd)
            )

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.requires_grad_(not config.frozen_pretrained)
        self.lm_head.requires_grad_(not config.frozen_pretrained)

    def load_pretrained(self, pretrained_model: GPT2LMHeadModel):
        for layer_idx in range(self.config.n_layer):
            self.latent_translators[layer_idx].load_pretrained(pretrained_model)
        self.transformer.load_state_dict(pretrained_model.transformer.state_dict())
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

    def convert_latent_into_skip_residue(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        latent_sampling_times: int = 32,
    ):
        mean, logvar = latent_mean_logvar
        zs = sampling(mean, logvar, latent_sampling_times) # [bs, dim_z, nz]
        zs = zs.to(self.dtype)
        skip_residue = F.layer_norm(
            self.skip_connection(zs.transpose(1, 2)),
            normalized_shape=[self.config.n_embd],
            eps=self.config.layer_norm_epsilon
        ).mean(dim=1, keepdim=True) # [bs, 1, n_embd]
        if self.config.add_ghost_skip_connection:
            skip_residue = (skip_residue - skip_residue.detach()) # equals to 0 but has gradient
        return skip_residue

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
        past_key_values = ()
        for layer_idx in range(self.config.n_layer):
            present = self.latent_translators[layer_idx](zs)
            past_key_values = past_key_values + (present, )
        return past_key_values

    def forward(
        self,
        latent_mean_logvar: Tuple[torch.FloatTensor] = None,
        past_skip_residue: Tuple[torch.FloatTensor] = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None, # specified for generation loop
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = None,
        scalar_loss: bool = True,
        latent_sampling: bool = True,
    ):
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

        # latent_kv_cache + target_inputs -> labels
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions or True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if self.config.add_skip_connection or self.config.add_ghost_skip_connection:
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
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "past_skip_residue": kwargs.get("past_skip_residue", None),
            }
        )

        return model_inputs


class GPT2ForCVAE(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPT2CVAEConfig):
        super().__init__(config)
        self.latent_encoder = GPT2ForLatentEncoder(config)
        self.latent_decoder = GPT2ForLatentDecoder(config)

    def load_pretrained(self, pretrained_model_name_or_path, device_map=None, torch_dtype=None):
        pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        self.latent_encoder.load_pretrained(pretrained_model)
        self.latent_decoder.load_pretrained(pretrained_model)

    def update_config(self, new_config):
        attr_mutable = ["frozen_pretrained", "add_contra_loss", "add_skip_connection", "add_ghost_skip_connection"]
        fail = False
        for attr in new_config:
            if attr in attr_mutable:
                self.config.update({attr: new_config[attr]})
            else:
                self_value, new_value = getattr(self.config, attr), new_config[attr]
                if not self_value == new_value:
                    print(f"immutable attr {attr} not match: {self_value} != {new_value}")
                    fail = True
        self.latent_decoder.transformer.requires_grad_(not self.config.frozen_pretrained)
        self.latent_decoder.lm_head.requires_grad_(not self.config.frozen_pretrained)
        assert not fail

    def kld_func(self, mean, logvar):
        loss_kld = compute_kl_penalty(mean, logvar, vae_type=self.config.vae_type)
        return mean, logvar, loss_kld

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
        loss_kld = compute_kl_penalty(mean, logvar, vae_type=self.config.vae_type)
        if not self.config.add_contra_loss:
            loss_lm = self.latent_decoder(
                latent_mean_logvar = latent_mean_logvar,
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
            )[0]
            loss_contra = torch.zeros_like(loss_lm)
        else:
            cross_nlls = []
            for bias in range(batch_size):
                biased_latent_mean_logvar = [torch.roll(_, shifts=bias, dims=0) for _ in latent_mean_logvar]
                biased_nlls = self.latent_decoder(
                    latent_mean_logvar = biased_latent_mean_logvar,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                    scalar_loss = False,
                )[0]
                cross_nlls.append(biased_nlls)
            cross_nlls = torch.stack(cross_nlls, dim=0) # [0, :] is positive, and [1:, :] are negative
            loss_lm = cross_nlls[0, :].mean()
            loss_contra = F.cross_entropy(
                input = -cross_nlls.T,
                target = torch.zeros(batch_size, dtype=torch.int64, device=cross_nlls.device),
                reduction = 'mean'
            )

        loss = loss_kld + loss_lm + loss_contra
        return loss, loss_kld, loss_lm, loss_contra

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        posterior_input_ids: Optional[torch.Tensor] = None,
        posterior_attention_mask: Optional[torch.Tensor] = None,
        latent_sampling: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        batch_size = input_ids.shape[0]

        if posterior_input_ids is not None:
            latent_mean_logvar = self.latent_encoder(
                posterior_input_ids = posterior_input_ids,
                posterior_attention_mask = posterior_attention_mask
            )
        else:
            mean = torch.zeros([batch_size, self.config.dim_z], device=self.device, dtype=torch.float32)
            logvar = torch.zeros([batch_size, self.config.dim_z], device=self.device, dtype=torch.float32)
            latent_mean_logvar = (mean, logvar)

        if "num_return_sequences" in kwargs:
            expand_size = kwargs["num_return_sequences"]
            latent_mean_logvar = (
                latent_mean_logvar[0].repeat_interleave(expand_size, dim=0),
                latent_mean_logvar[1].repeat_interleave(expand_size, dim=0)
            )

        if self.config.add_skip_connection or self.config.add_ghost_skip_connection:
            skip_residue = self.latent_decoder.convert_latent_into_skip_residue(latent_mean_logvar=latent_mean_logvar)
            past_skip_residue = (skip_residue,)
        else:
            past_skip_residue = None

        past_key_values = self.latent_decoder.convert_latent_into_kv_cache(
            latent_mean_logvar = latent_mean_logvar,
            latent_sampling = latent_sampling,
        )
        attention_mask = torch.cat([attention_mask.new_ones(size=[batch_size, self.config.num_p]), attention_mask], dim=1)
        input_ids = torch.cat([input_ids.new_zeros(size=[batch_size, self.config.num_p]), input_ids], dim=1)

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["past_key_values"] = past_key_values
        kwargs["past_skip_residue"] = past_skip_residue

        return self.latent_decoder.generate(
            inputs = inputs,
            generation_config = generation_config,
            logits_processor = logits_processor,
            stopping_criteria = stopping_criteria,
            **kwargs
        )[:, self.config.num_p:]