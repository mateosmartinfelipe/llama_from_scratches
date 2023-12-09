import torch
from torch import nn
import torch.nn.functional as F 
from  typing import Optional
import math
from dataclasses import dataclass 

@dataclass
class KVCache:
    max_batch_size:int = 32
    max_seq_len:int = 2048


@dataclass
class ModelArgs:
    vocabulary_size : int
    dim : int = 4096
    n_layers:int = 32
    n_heads:int = 32
    n_kv_heads:Optional[int] = None
    multiple_of:int = 256
    ffn_dim_multiplier:Optional[float] = None
    norm_eps:float = 1e-5
    kv:KVCache
    device:str = "cpu"


class Transformer(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        
