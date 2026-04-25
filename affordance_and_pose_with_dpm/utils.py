import torch
from pathlib import Path
import torch
import torch.nn as nn


def get_weights_file_path(model_folder, model_basename, epoch: str):
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(model_folder, model_basename):
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename)) 
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) 


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu" 
    


# ----------------------------------------------------------------------
# radius_utils.py  (or paste near the bottom of your script)
# ----------------------------------------------------------------------
import torch
from tqdm import tqdm

@torch.no_grad()
def suggest_radii(
    loader,
    k: int = 32,
    batches_to_scan: int = 32,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
    device: str = "cuda",
    max_samples: int = 100_000
):
    """
    Estimate good (r_min, r_max) by sampling up to `max_samples` k-th distances
    instead of collecting them all.
    """
    import math
    kth_samples = []
    total_seen = 0

    for i, batch in enumerate(tqdm(loader, total=batches_to_scan,
                                   desc="Scanning batches for radius stats")):
        if i >= batches_to_scan:
            break

        xyz = batch[4].to(device)       # (B, N, 3)
        B, N, _ = xyz.shape

        # (B,N,N) → distances
        d = torch.cdist(xyz, xyz)       # consider subsampling if N>>4k
        kth = d.topk(k, largest=False).values[:, :, -1]  # (B,N)

        # flatten and sub-sample from this batch
        flat = kth.view(-1).cpu()
        num_to_take = min(int(max_samples - total_seen), flat.numel())
        if num_to_take <= 0:
            break

        # random indices into flat
        perm = torch.randperm(flat.numel())[:num_to_take]
        kth_samples.append(flat[perm])
        total_seen += num_to_take

        if total_seen >= max_samples:
            break

    # concatenate at most max_samples values
    all_kth = torch.cat(kth_samples)
    r_min = all_kth.quantile(lower_q).item()
    r_max = all_kth.quantile(upper_q).item()

    print(f"\nSuggested radii   r_min = {r_min:.4f}   r_max = {r_max:.4f}  "
          f"(based on ~{all_kth.numel()} samples, k={k}, q={lower_q}/{upper_q})")
    return r_min, r_max
