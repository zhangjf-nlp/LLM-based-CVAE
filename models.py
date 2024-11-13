import copy

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import load_sharded_checkpoint

from modeling_llama_cvae import LlamaForCVAE, LlamaForLatentDPO
from modeling_gpt2_cvae import GPT2ForCVAE
from utils import load_sharded_prefix_checkpoint

supported_cvae_architectures = {
    "GPT2ForCVAE": GPT2ForCVAE,
    "LlamaForCVAE": LlamaForCVAE,
}
supported_causallm_architectures = {
    "GPT2LMHeadModel": GPT2ForCVAE,
    "LlamaForCausalLM": LlamaForCVAE,
}
CVAEClass2LatentDPOClass = {
    LlamaForCVAE: LlamaForLatentDPO
}
class AutoModelForCVAE:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cvae_structure_args=None, device_map=None, torch_dtype=None):
        cvae_structure_args = {} if cvae_structure_args is None else cvae_structure_args
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        architectures = getattr(config, "architectures", [])
        for arch in architectures:
            if arch in supported_cvae_architectures:
                cvae_class = supported_cvae_architectures[arch]
                model = cvae_class.from_pretrained(pretrained_model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
                model.update_config(cvae_structure_args)
                return model
        for arch in architectures:
            if arch in supported_causallm_architectures:
                cvae_class = supported_causallm_architectures[arch]
                config_class = cvae_class.config_class
                config = config_class.from_pretrained(pretrained_model_name_or_path)
                config.update(cvae_structure_args)
                model = cvae_class(config).to(device=device_map, dtype=torch_dtype)
                model.load_pretrained(pretrained_model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
                return model


class AutoModelForLatentDPO:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, beta=0.1, device_map=None, torch_dtype=None):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        architecture = getattr(config, "architectures", [])[0]
        lpo_class = CVAEClass2LatentDPOClass[supported_cvae_architectures[architecture]]
        model = lpo_class(config=config, beta=beta).to(device=device_map, dtype=torch_dtype)
        model.load_cvae_checkpoint(pretrained_model_name_or_path=pretrained_model_name_or_path)
        return model


class AutoModelForDPO:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, lora_dim=0, beta=0.1, *args, **kargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kargs)
        class DPOModel(PreTrainedModel):
            config_class = model.config_class
            main_input_name = "chosen_ids"
            def __init__(self, config, model, beta):
                super().__init__(config)
                self.actor_model = model
                if lora_dim == 0:
                    self.actor_model = model
                else:
                    from peft import get_peft_model, LoraConfig, TaskType
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=lora_dim, lora_alpha=lora_dim, lora_dropout=0.0
                    )
                    self.actor_model = get_peft_model(model, peft_config)

                self.refer_model = copy.deepcopy(model)
                self.refer_model.requires_grad_(False)
                self.refer_model.eval()
                self.beta = beta

            def save_pretrained(self, *args, **kwargs):
                self.actor_model.save_pretrained(*args, **kwargs)

            def compute_nlls(self, input_ids, attention_mask, mode="actor"):
                model = self.actor_model if mode == "actor" else self.refer_model
                lm_logits = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
                shift_logits = lm_logits[..., :-1, :].contiguous()
                labels = torch.where(attention_mask.eq(1), input_ids, -100)
                shift_labels = labels[..., 1:].contiguous()
                nlls = F.cross_entropy(
                    input=shift_logits.transpose(1, 2),
                    target=shift_labels,
                    reduction='none'
                ).sum(dim=-1)
                return nlls

            def forward(
                self,
                chosen_ids,
                chosen_attention_mask,
                rejected_ids,
                rejected_attention_mask,
                return_loss=True,
            ):
                actor_chosen_nlls = self.compute_nlls(input_ids=chosen_ids, attention_mask=chosen_attention_mask, mode="actor")
                with torch.no_grad():
                    refer_chosen_nlls = self.compute_nlls(input_ids=chosen_ids, attention_mask=chosen_attention_mask, mode="refer")
                chosen_reward = refer_chosen_nlls.detach() - actor_chosen_nlls

                actor_rejected_nlls = self.compute_nlls(input_ids=rejected_ids, attention_mask=rejected_attention_mask, mode="actor")
                with torch.no_grad():
                    refer_rejected_nlls = self.compute_nlls(input_ids=rejected_ids, attention_mask=rejected_attention_mask, mode="refer")
                rejected_reward = refer_rejected_nlls.detach() - actor_rejected_nlls

                loss = -F.logsigmoid((chosen_reward-rejected_reward)*self.beta)
                acc = (chosen_reward > rejected_reward).float()
                return loss.mean(), acc.mean()

        return DPOModel(config=model.config, model=model, beta=beta)