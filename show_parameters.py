import torch
from modeling_llama_cvae import LlamaForCVAE

for fs in [2,3]:
    model = LlamaForCVAE.from_pretrained(
        "/cfs/hadoop-aipnlp/zhangjianfei09/models/llama3-8b",
        add_gskip_connection=True,
        frozen_stage=fs,
        device_map="cuda:0"
    )

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"frozen stage: {fs}")
    print(f"Latent Encoder: {count_parameters(model.latent_encoder):,}")
    print(f"Latent Adapter: {count_parameters(model.latent_decoder.latent_adapters):,}")
    print(f"G-Skip connection: {count_parameters(model.latent_decoder.skip_connection):,}")
    print(f"LLM backbone: {count_parameters(model.latent_decoder.model)+count_parameters(model.latent_decoder.lm_head):,}")

    del(model)
    torch.cuda.empty_cache()


"""
frozen stage: 0 # all
Latent Encoder: 978,427,920
Latent Adapter: 1,376,454,656
G-Skip connection: 16,920,576
LLM backbone: 8,030,261,248

frozen stage: 2 # trainable
Latent Encoder: 453,091,344
Latent Adapter: 34,277,376
G-Skip connection: 16,920,576
LLM backbone: 0

frozen stage: 3 # train from scratch
Latent Encoder: 16,867,344
Latent Adapter: 34,277,376
G-Skip connection: 16,920,576
LLM backbone: 0
"""