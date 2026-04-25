# albert_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Literal
from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel


@dataclass
class AlbertCfg:
    model_name: str = "albert-base-v2"
    seq_len:   int  = 512
    fp16:      bool = False
    device:    torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile:   bool = torch.__version__.startswith("2.")               # torch.compile if PyTorch 2


class AlbertEncoder(nn.Module):
    """ALBERT encoder returning pooled or per-token embeddings."""
    def __init__(self, cfg: AlbertCfg = AlbertCfg(), pooled: bool=True):
        super().__init__()
        self.cfg = cfg
        self.pooled = pooled
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)

        model: PreTrainedModel = AutoModel.from_pretrained(
            cfg.model_name, add_pooling_layer=True
        )
        if cfg.fp16 and cfg.device.type == "cuda":
            model = model.half()
        self.model = model.to(cfg.device)
        if cfg.compile:
            self.model = torch.compile(self.model)

        self.hidden_size = self.model.config.hidden_size

    def forward(
        self, text: Union[str, List[str]]
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        batch = self.tok(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.cfg.seq_len,
        ).to(self.cfg.device)

        with torch.set_grad_enabled(self.training):
            outs = self.model(**batch)

        return outs.pooler_output if self.pooled else outs.last_hidden_state



if __name__ == "__main__":
    texts = ["Pass the coffee.", "Hold the book."]
    enc_pooled = AlbertEncoder(pooled=True).eval()
    enc_token = AlbertEncoder(pooled=False).eval()
    print("pooled:", enc_pooled(texts).shape)  # D = 768, sequence lenght = 6
    print("tokens:", enc_token(texts).shape)

    enc_pooled.train()
    out = enc_pooled(texts)
    loss = nn.MSELoss()(out, torch.randn_like(out))
    loss.backward()
    print("loss:", loss.item())
