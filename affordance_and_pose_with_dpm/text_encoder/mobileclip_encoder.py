# mobileclip_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Literal
from contextlib import nullcontext

import torch
import open_clip


# config
@dataclass
class MobileCLIPCfg:
    model_name: str = "MobileCLIP-B"          
    pretrained: str = "datacompdr"
    fp16: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile: bool = torch.__version__.startswith("2.")   # torch 2.x → compile



class MobileCLIPEncoder(torch.nn.Module):
    """
    MobileCLIP text encoder returning
    -  sentence embeddings  ("pooled")
    - per-token embeddings        ("tokens")
    """

    def __init__(self, cfg: MobileCLIPCfg = MobileCLIPCfg(), pooled: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pooled = pooled

        model, _, _ = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained, device=cfg.device
        )

        if cfg.fp16 and cfg.device.type == "cuda":
            model = model.half()
        if cfg.compile:
            model = torch.compile(model)

        self.model = model.to(cfg.device).eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

        # MobileCLIP puts the text tower inside `.text`; classic OpenCLIP doesn't.
        self.text_tower = self.model.text if hasattr(self.model, "text") else self.model
        self.ctx_len = getattr(self.text_tower, "context_length", 77)

        # embed dim
        try:
            self.embed_dim: int = self.text_tower.text_projection.shape[1]
        except AttributeError:
            dummy = self.tokenizer(["dummy"], context_length=self.ctx_len).to(cfg.device)
            with torch.no_grad():
                self.embed_dim = self.model.encode_text(dummy).shape[-1]

    # convenience
    @property
    def device(self):  # noqa: D401
        return self.cfg.device

  
    def forward(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(text, context_length=self.ctx_len).to(self.device)
        with torch.set_grad_enabled(self.training):
            # Fast path for "pooled" only
            if self.pooled:
                return self.model.encode_text(tokens)

            tt = self.text_tower  # shorthand
            x = tt.token_embedding(tokens)                   # (B, L, d_m)
            x = x + tt.positional_embedding
            x = tt.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = tt.ln_final(x)                               # (B, L, d_m)
            tokens_out = x @ tt.text_projection              # (B, L, D)
            return tokens_out
            
            
            
if __name__ == "__main__":
    texts = ["Pass the coffee.", "Hold the book."]
    enc_pooled = MobileCLIPEncoder(cfg=MobileCLIPCfg(fp16=False), pooled=True).eval()  
    enc_token = MobileCLIPEncoder(cfg=MobileCLIPCfg(fp16=False), pooled=False).eval()
    print("pooled:", enc_pooled(texts).shape)  # D = 512
    print("tokens:", enc_token(texts).shape)
        
    enc_pooled.train()
    out = enc_pooled(texts)
    loss = torch.nn.MSELoss()(out, torch.randn_like(out))
    loss.backward()
    print("loss:", loss.item())

    import time
    def _sync():
        """Synchronise the right clock (GPU if present, else CPU)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def bench(fn, n_iter: int = 10):
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        _sync()
        return (time.perf_counter() - t0) / n_iter


    enc = MobileCLIPEncoder(pooled=True).eval()        
    texts = ["Pass the coffee.", "Hold the book."] * 16

    _ = enc(texts); _sync()

    time_tokens = bench(lambda: enc(texts))
    time_pooled = bench(lambda: enc(texts))  



    print(f"average wall-time  (batch={len(texts)}):")
    print(f"  tokens : {time_tokens * 1e3:.2f} ms")
    print(f"  pooled : {time_pooled * 1e3:.2f} ms\n")
