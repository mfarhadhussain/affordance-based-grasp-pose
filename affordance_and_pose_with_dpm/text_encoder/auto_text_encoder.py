# auto_text_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
from enum import Enum, auto
import time
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import open_clip


class ModelType(Enum):
    """Pre-configured model types"""
    # HuggingFace models
    MINI_LM = auto()        # sentence-transformers/all-MiniLM-L6-v2
    MOBILEBERT = auto()     # google/mobilebert-uncased
    TINY_BERT = auto()      # huawei-noah/TinyBERT_General_4L_312D
    ALBERT = auto()         # albert-base-v2
    DISTILBERT = auto()     # distilbert-base-uncased
    
    # CLIP models
    CLIP_VIT_B32 = auto()   # openai ViT-B/32
    CLIP_RN50 = auto()      # OpenAI RN50
    CLIP_VIT_L14 = auto()   # openai ViT-L/14
    MOBILE_CLIP = auto()    # apple/mobileclip-small



@dataclass
class BenchmarkResult:
    model_type: str
    pooled_time: float
    token_time: float
    embed_dim: int
    param_count: int
    batch_size: int
    seq_length: int
    device: str


@dataclass
class TextEncoderCfg:
    model_type: ModelType = ModelType.MINI_LM
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name: Optional[str] = None


class TextEncoder(nn.Module):
    """Unified text encoder supporting multiple model families"""
    
    HF_MODEL_MAP = {
        ModelType.MINI_LM: "sentence-transformers/all-MiniLM-L6-v2",
        ModelType.MOBILEBERT: "google/mobilebert-uncased",
        ModelType.TINY_BERT: "huawei-noah/TinyBERT_General_4L_312D",
        ModelType.ALBERT: "albert-base-v2",
        ModelType.DISTILBERT: "distilbert-base-uncased",
    }
    
    CLIP_MODEL_MAP = {
        ModelType.CLIP_VIT_B32: ("ViT-B-32", "laion400m_e32"),
        ModelType.CLIP_RN50: ("RN50", "openai"),
        ModelType.CLIP_VIT_L14: ("ViT-L-14", "openai"),
        ModelType.MOBILE_CLIP: ("MobileCLIP-B", "datacompdr"),
    }

    def __init__(self, cfg: TextEncoderCfg = TextEncoderCfg(), pooled: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pooled = pooled
        self.is_clip = cfg.model_type in self.CLIP_MODEL_MAP
        
        if self.is_clip:
            model_name, pretrained = self.CLIP_MODEL_MAP[cfg.model_type]
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=cfg.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            
            # Handle MobileCLIP's different architecture
            if cfg.model_type == ModelType.MOBILE_CLIP:
                self.text_tower = self.model.text
                self.ctx_len = getattr(self.text_tower, "context_length", 77)
                try:
                    self.embed_dim = self.text_tower.text_projection.shape[-1]
                except AttributeError:
                    # Fallback for MobileCLIP if text_projection isn't directly accessible
                    dummy = self.tokenizer(["dummy"], context_length=self.ctx_len).to(cfg.device)
                    with torch.no_grad():
                        self.embed_dim = self.model.encode_text(dummy).shape[-1]
            else:
                # Standard CLIP models
                self.text_tower = self.model
                self.ctx_len = getattr(self.text_tower, "context_length", 77)
                self.embed_dim = self.model.text_projection.shape[-1]
        else:
            model_name = cfg.model_name or self.HF_MODEL_MAP[cfg.model_type]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(cfg.device).eval()
            self.embed_dim = self.model.config.hidden_size

    @property
    def device(self):
        return self.cfg.device

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        if self.is_clip:
            # CLIP model processing
            tokens = self.tokenizer(text, context_length=getattr(self, 'ctx_len', None)).to(self.device)
            with torch.set_grad_enabled(self.training):
                if self.pooled:
                    return self.model.encode_text(tokens)
                # For token-level outputs
                if self.cfg.model_type == ModelType.MOBILE_CLIP:
                    # MobileCLIP specific processing
                    x = self.text_tower.token_embedding(tokens)
                    x = x + self.text_tower.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.text_tower.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.text_tower.ln_final(x)
                    return x @ self.text_tower.text_projection
                else:
                    # Standard CLIP processing
                    x = self.model.token_embedding(tokens)
                    x = x + self.model.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.model.transformer(x)
                    return x.permute(1, 0, 2)  # LND -> NLD
        else:
            # HuggingFace model processing
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.set_grad_enabled(self.training):
                outputs = self.model(**enc)
                if self.pooled:
                    last_hidden = outputs.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1)
                    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                return outputs.last_hidden_state


def benchmark_model(
    cfg: TextEncoderCfg, 
    texts: List[str], 
    warmup: int = 3, 
    runs: int = 10
) -> BenchmarkResult:
    """Comprehensive model benchmarking"""
    device = cfg.device
    
    # Initialize and warmup
    enc = TextEncoder(cfg, pooled=True).eval()
    for _ in range(warmup):
        _ = enc(texts)
    
    # Benchmark pooled
    torch.cuda.synchronize() if str(device) == "cuda" else None
    start = time.time()
    for _ in range(runs):
        _ = enc(texts)
    torch.cuda.synchronize() if str(device) == "cuda" else None
    pooled_time = (time.time() - start) / runs
    
    # Benchmark token-level
    enc = TextEncoder(cfg, pooled=False).eval()
    torch.cuda.synchronize() if str(device) == "cuda" else None
    start = time.time()
    for _ in range(runs):
        _ = enc(texts)
    torch.cuda.synchronize() if str(device) == "cuda" else None
    token_time = (time.time() - start) / runs
    
    # Get model statistics
    param_count = sum(p.numel() for p in enc.model.parameters())
    
    # Get sequence length
    if enc.is_clip:
        tokens = enc.tokenizer(texts)
        seq_length = tokens.shape[-1] if len(tokens.shape) > 1 else len(tokens)
    else:
        tokenized = enc.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        seq_length = tokenized["input_ids"].shape[1]
    
    return BenchmarkResult(
        model_type=cfg.model_type.name,
        pooled_time=pooled_time,
        token_time=token_time,
        embed_dim=enc.embed_dim,
        param_count=param_count,
        batch_size=len(texts),
        seq_length=seq_length,
        device=str(device),
    )


def save_results(
    results: List[BenchmarkResult], 
    filename: str = "benchmark_results.json",
    print_summary: bool = True
):
    """Save and display benchmarking results"""
    # Save to JSON
    data = [vars(result) for result in results]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results to {filename}")
    
    # Print summary table
    if print_summary:
        print("\n=== Benchmark Summary ===")
        print(f"{'Model':<20} {'Pooled (ms)':>12} {'Token (ms)':>12} {'Dim':>6} {'Params':>12} {'Device':<10}")
        print("-" * 80)
        for result in results:
            print(f"{result.model_type:<20} "
                  f"{result.pooled_time*1000:>12.2f} "
                  f"{result.token_time*1000:>12.2f} "
                  f"{result.embed_dim:>6} "
                  f"{result.param_count:>12,} "
                  f"{result.device:<10}")


if __name__ == "__main__":
    # Test data
    texts = [
        "Pass the coffee to Captain.",
        "Hello robot hold the book.",
        "Hold the glass carefully.",
    ]
    
    # Benchmark all model types
    results = []
    for model_type in ModelType:
        try:
            cfg = TextEncoderCfg(model_type=model_type)
            print(f"\nBenchmarking {model_type.name}...")
            result = benchmark_model(cfg, texts)
            results.append(result)
            
            # Print individual results
            print(f"\n=== {model_type.name} ===")
            print(f"Embedding dimension: {result.embed_dim}")
            print(f"Parameters: {result.param_count:,}")
            print(f"Pooled time: {result.pooled_time*1000:.2f}ms")
            print(f"Token time: {result.token_time*1000:.2f}ms")
            print(f"Batch size: {result.batch_size}")
            print(f"Sequence length: {result.seq_length}")
            print(f"Device: {result.device}")
        except Exception as e:
            print(f"Failed to benchmark {model_type.name}: {str(e)}")
    
    # Save and display results
    if results:
        save_results(results)
    else:
        print("No results to save - all benchmarks failed")
