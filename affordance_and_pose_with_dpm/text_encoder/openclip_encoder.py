# openclip_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union
from contextlib import nullcontext

import torch
from torch import nn
import open_clip


@dataclass
class CLIPCfg:
    model_name: str = "ViT-B-32"
    # pretrained: str = "laion400m_e32"
    pretrained: str = "laion2b_s34b_b79k"
    fp16: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile: bool = torch.__version__.startswith("2.")


class OpenCLIPEncoder(nn.Module):
    """OpenCLIP encoder returning pooled or per-token embeddings."""
    def __init__(self, cfg: CLIPCfg = CLIPCfg(), pooled: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pooled = pooled

        model, _, _ = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        if cfg.fp16 and cfg.device.type == "cuda":
            model = self.model.half()
        if cfg.compile:
            model = torch.compile(self.model)
            
        self.model = model.to(cfg.device).eval()
        self.tok = open_clip.get_tokenizer(cfg.model_name)
        self.embed_dim = self.model.text_projection.shape[1]
        self.ctx_len = getattr(self.model, "context_length", 77)

    def forward(
        self, text: Union[str, List[str]]
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        tokens = self.tok(text).to(self.cfg.device)
        with torch.set_grad_enabled(self.training):
            if self.pooled:                                    # (B, D)
                return self.model.encode_text(tokens)
            x = self.model.token_embedding(tokens)        # (B, L, d_m)
            x = x + self.model.positional_embedding
            x = self.model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.model.ln_final(x)                    # (B, L, d_m)
            return x @ self.model.text_projection         # (B, L, D)


if __name__ == "__main__":
    texts = ["Pass the coffee.", "Hold the book."]
    enc_pooled = OpenCLIPEncoder(cfg=CLIPCfg(fp16=False), pooled=True).eval()
    enc_token = OpenCLIPEncoder(cfg=CLIPCfg(fp16=False), pooled=False).eval()  # D = 512
    print("pooled:", enc_pooled(texts).shape)
    print("tokens:", enc_token(texts).shape)

    enc_token.train()
    out = enc_token(texts)
    loss = nn.MSELoss()(out, torch.randn_like(out))
    loss.backward()
    print("loss:", loss.item())
