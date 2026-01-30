#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from safetensors import safe_open
from safetensors.torch import load_file
from dataclasses import dataclass, fields  # Add 'fields' to the import
from safetensors.torch import save_model as safe_save_model
from torch import Tensor


# --- Reproducibility ---
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Global caches
TOKENIZERS_CACHE = {}
MODEL_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        #tokenizer.vocab_size = 69139
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

@dataclass
class LMConfig:
    hid_dim: int = 576
    inter_dim: int = 1536
    rms_eps: float = 1e-5
    n_heads: int = 9
    n_kv_heads: int = 3
    max_pos_emb: int = 8192
    re_base: int = 100000
    attn_scaling: float = 1.0
    dropout: float = 0.0
    use_token: bool = True
    tie_weight: bool = True
    eos_token_id: int = 2
    pad_token_id: int = 0
    base_vocal_size: int = 69139 
    extra_token_ammout: int = 0
    vocal_size: int = base_vocal_size + extra_token_ammout
    n_block: int = 45
    tokenizer: str = "/content/tamiLlm/engtam_tokens_70k"



def get_model(path: str, cfg: LMConfig):
    if path not in MODEL_CACHE:
        
        model = LanguageModel(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if path.endswith('.safetensors'):
            print(f"Loading model from .safetensor file: {path}")
            state_dict = load_file(path)
        else:
            print(f"Loading model from .bin file: {path}")
            state_dict = torch.load(path, map_location=device)
        
        model.load_state_dict(state_dict, strict=False)
        
        if cfg.tie_weight:
            model.head.weight = model.token_emb.weight
        
        MODEL_CACHE[path] = model
        print(f"✅ Model loaded from {path}")
    
    return MODEL_CACHE[path]



class RMSNorm(nn.Module):
    """
    RMS Normalization module.
    """
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.weight=nn.Parameter(torch.empty(cfg.hid_dim))
        self.eps=cfg.rms_eps

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        irms: Tensor =torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps) # (B, T, 1) inverse root mean square
        x= x*irms*self.weight # (B, T, C) scaled by weight
        return x # (B, T, C)

class RotaryEmbbeding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    """
    def __init__(self,cfg:LMConfig): # Added type hint for cfg
        super().__init__()

        assert cfg.hid_dim % cfg.n_heads==0, "Hidden dimension should be divisible by number of heads"
        self.dim= cfg.hid_dim//cfg.n_heads     # 576 / 9 = 64 (dimension per head)
        self.max_seq_len=cfg.max_pos_emb
        self.base=cfg.re_base # Corrected rm_base to re_base
        inv_freq=  1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))  # (dim/2,) e.g., (32,) for dim=64
        self.register_buffer("inv_freq",inv_freq, persistent=False) # Added persistent=False
        self.original_max_seq_len=cfg.max_pos_emb
        self.attn_scal=cfg.attn_scaling

    @torch.no_grad()
    def forward(self,position_ids) -> tuple[Tensor, Tensor]:
        batch,seq_len=position_ids.shape                       # (B, T_curr) e.g., (3,100)
        max_seq=position_ids.max()+1                  # Max position ID in the batch, plus 1. e.g., 100
        if max_seq > self.original_max_seq_len:
            scale=max_seq/self.original_max_seq_len
            inv_freq=self.inv_freq/scale
        else:
            inv_freq: Tensor=self.inv_freq

        flat_position_id: Tensor=position_ids.reshape(-1).float()   # (B * T_curr,) e.g., (300,)

        freqs:Tensor=flat_position_id.unsqueeze(-1)*inv_freq.unsqueeze(0)  # (B*T_curr, 1) * (1, dim/2) -> (B*T_curr, dim/2) e.g., (300, 32)
        freqs=freqs.reshape(batch,seq_len,-1)        # (B, T_curr, dim/2) e.g., (3,100,32)

        emb: Tensor=torch.cat([freqs,freqs],dim=-1)    # (B, T_curr, dim) e.g., (3,100,64)

        cos: Tensor=torch.cos(emb)*self.attn_scal      # (B, T_curr, dim) e.g., (3,100,64)
        sin: Tensor=torch.sin(emb)*self.attn_scal      # (B, T_curr, dim)

        return cos,sin                # (B, T_curr, head_dim) each e.g., (3,100,64)

def rotate_half(x: Tensor) -> Tensor:
    """
    Rotates half of the input tensor's dimensions for RoPE.
    """
    x1,x2=x.chunk(2,dim=-1) # Split last dimension into two halves
    return torch.cat((-x2,x1),dim=-1) # Concatenate rotated halves

def apply_rotary_pos_emb(q:Tensor,k:Tensor,cos:Tensor,sin:Tensor,unsqueeze_dim:int=1): # Corrected unsqeeze_dim to unsqueeze_dim
    """
    Applies rotary position embeddings to query and key tensors.
    """
    cos:Tensor =cos.unsqueeze(unsqueeze_dim) # Expand cos to match q/k dimensions: (B, 1, T_curr, head_dim)
    sin=sin.unsqueeze(unsqueeze_dim) # Expand sin to match q/k dimensions

    q_emb: Tensor=(q*cos) + (rotate_half(q)*sin) # Apply RoPE to query
    k_emb=(k*cos) + (rotate_half(k)*sin) # Apply RoPE to key

    return q_emb,k_emb # (B, n_heads, T_curr, head_dim) for q_emb, (B, n_kv_heads, T_curr, head_dim) for k_emb

class LanguageModelGroupedQuaryAttention(nn.Module):
    """
    Grouped Query Attention module.
    """
    def __init__(self,cfg:LMConfig):
        super().__init__()
        self.n_heads=cfg.n_heads
        self.n_kv_heads=cfg.n_kv_heads
        self.emb_dim=cfg.hid_dim
        self.dropout=cfg.dropout

        assert self.n_heads%self.n_kv_heads==0 , "N Heads should be divisible by N KV Heads"
        assert self.emb_dim%self.n_heads==0, "Embedding dimension should be divisible by N Heads" # Corrected typo

        self.n_kv_groups=self.n_heads//self.n_kv_heads      # 9 / 3 = 3 (number of query heads per KV head)
        self.head_dim= self.emb_dim // self.n_heads          # 576 / 9 = 64 (dimension per head)

        self.q_proj= nn.Linear(self.emb_dim,self.emb_dim,bias=False) # Changed bias to False, common in LLMs
        self.k_proj= nn.Linear(self.emb_dim,self.n_kv_heads*self.head_dim,bias=False) # Changed bias to False
        self.v_proj= nn.Linear(self.emb_dim,self.n_kv_heads*self.head_dim,bias=False) # Changed bias to False
        self.out_proj = nn.Linear(self.emb_dim,self.emb_dim,bias=False) # Changed bias to False

        self.attn_dropout= nn.Dropout(self.dropout)
        self.resid_dropout= nn.Dropout(self.dropout)

        self.sdpa= hasattr(F,'scaled_dot_product_attention') # Use F for functional

        if not self.sdpa:
            print("Warning: Scaled dot product attention not available, using standard attention in LM.") # Corrected typo

    def forward(self,x:Tensor,cos:Tensor,sin:Tensor,attention_mask=None,block_kv_cache=None):
        is_prefill=block_kv_cache is None

        b,t_cr,c= x.size()   # (Batch size, current sequence length, embedding dimension) e.g., (3,100,576)

        q_curr:Tensor= self.q_proj(x).view(b,t_cr,self.n_heads,self.head_dim).transpose(1,2)
        # (B, T_curr, C) -> (B, T_curr, n_heads, head_dim) -> (B, n_heads, T_curr, head_dim) e.g., (3,9,100,64)

        k_curr:Tensor= self.k_proj(x).view(b,t_cr,self.n_kv_heads,self.head_dim).transpose(1,2)
        # (B, T_curr, C) -> (B, T_curr, n_kv_heads, head_dim) -> (B, n_kv_heads, T_curr, head_dim) e.g., (3,3,100,64)

        v_curr:Tensor= self.v_proj(x).view(b,t_cr,self.n_kv_heads,self.head_dim).transpose(1,2) # (B, n_kv_heads, T_curr, head_dim) e.g., (3,3,100,64)

        q,k_rotated=apply_rotary_pos_emb(q_curr,k_curr,cos,sin) # q: (B, n_heads, T_curr, head_dim), k_rotated: (B, n_kv_heads, T_curr, head_dim)

        if not is_prefill and block_kv_cache is not None and 'key' in block_kv_cache: # Added check for block_kv_cache being a dict and having 'key'
            k= block_kv_cache['key']
            v=block_kv_cache['value']
            k: Tensor=torch.cat([k,k_rotated],dim=2)            # Concatenate keys: (B, n_kv_heads, T_prev + T_curr, head_dim) e.g., (3,3,200,64)
            v: Tensor=torch.cat([v,v_curr],dim=2) # Concatenate values
            block_kv_cache['key']=k
            block_kv_cache['value']=v
        else:
            k=k_rotated                                         # (B, n_kv_heads, T_curr, head_dim) e.g., (3,3,100,64)
            v=v_curr
            block_kv_cache={'key':k,'value':v} # Initialize cache for prefill

        k_exp= k.repeat_interleave(self.n_kv_groups,dim=1)   # Expand KV heads to match Q heads: (B, n_heads, T_kv, head_dim) e.g., (3,9,100,64)
        v_exp=v.repeat_interleave(self.n_kv_groups,dim=1)    # (B, n_heads, T_kv, head_dim)

        t_kv =k_exp.size(2)    # Total key/value sequence length

        additive_attn_mask= None

        if attention_mask is not None:
            # The attention_mask needs to be applied to the KV sequence length
            # It should typically be (B, 1, 1, T_kv) or (B, 1, T_curr, T_kv) for broadcasting.
            # Your original line: `addtitive_attn_mask=(1-attention_mask.unsqeenze(1).unsqeenze(2).float())*torch.finfo(q.dtype).min`
            # Needs correction: `unsqeenze` -> `unsqueeze`. Also, `attention_mask` might be `(B, T_curr)`.
            # If `attention_mask` is (B, T_curr) and refers to the *current* sequence, and we need a mask for the *full KV* sequence:
            # For causal LM, usually we build a triangular mask. If an external mask is provided, it usually masks PAD tokens.
            
            # Assuming attention_mask is (B, T_curr) and corresponds to valid tokens (1) vs padding (0)
            # We need to expand it to be additive: (B, 1, 1, T_kv)
            additive_attn_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min # (B, 1, 1, T_curr) -> (B, 1, 1, T_kv) assuming T_kv matches attention_mask's seq len for initial input
            # If attention_mask is for the full sequence (prefill), it should be (B, T_full).
            # The original code `mask_for_keys=attention_mask[:,:t_kv]` implies `attention_mask` might be larger.
            # For simplicity with the provided `attention_mask`, let's assume it has the right shape (B, T_kv) for padding.
            if attention_mask.dim() == 2: # Check if it's (B, T_seq)
                # Expand it to (B, 1, 1, T_seq) for broadcasting
                additive_attn_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            else: # If it's already more dimensions, assume it's correctly formatted
                additive_attn_mask = (1 - attention_mask.float()) * torch.finfo(q.dtype).min


        if self.sdpa and x.device.type != 'mps':
            is_causal: bool= (t_cr==t_kv and t_cr>1) # Corrected typo casual to causal

            y=F.scaled_dot_product_attention( # Use F.scaled_dot_product_attention
                q,k_exp,v_exp,
            attn_mask=additive_attn_mask, # Pass the additive_attn_mask
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
            )

        else:
            attn = q @ k_exp.transpose(-1,-2) / (self.head_dim)**0.5     # (B, n_heads, T_curr, T_kv) attention scores

            if t_cr==t_kv and t_cr>1:
                causal_mask_val= torch.tril(torch.ones(t_cr,t_cr,device=x.device,dtype=torch.bool)).view(1,1,t_kv,t_kv) # Corrected trill to tril
                attn =attn.masked_fill(~causal_mask_val,float('-inf')) # Apply causal mask for auto-regressive

            if additive_attn_mask is not None:
                attn = attn + additive_attn_mask # Apply external padding mask

            attn=F.softmax(attn,dim=-1) # Use F.softmax
            attn=self.attn_dropout(attn)
            y=attn @ v_exp                          # (B, n_heads, T_curr, head_dim)

        y= y.transpose(1,2).contiguous().view(b,t_cr,c)    # (B, T_curr, C) e.g., (3,100,576)
        y= self.out_proj(y)                             # (B, T_curr, C)
        y=self.resid_dropout(y)

        return y,block_kv_cache                      # y=(B, T_curr, C) e.g., (3,100,576)

class LanguageModelMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module, typically a SwiGLU variant.
    """
    def __init__(self,cfg:LMConfig):
        super().__init__()
        self.hid_dim=cfg.hid_dim # Corrected emb_dim to hid_dim for consistency
        self.inter_dim=cfg.inter_dim

        self.activation_fn= F.silu # Use F.silu
        self.gate_proj= nn.Linear(self.hid_dim,self.inter_dim,bias=False) # Changed bias to False
        self.up_proj= nn.Linear(self.hid_dim,self.inter_dim,bias=False) # Changed bias to False
        self.down_proj= nn.Linear(self.inter_dim,self.hid_dim,bias=False) # Changed bias to False

    def forward(self,x:Tensor): # Added type hint for x
        gate=self.activation_fn(self.gate_proj(x))    # Input: (B, T, hid_dim), Output: (B, T, inter_dim) e.g., (3,100,1536)
        x=self.up_proj(x)                             # Input: (B, T, hid_dim), Output: (B, T, inter_dim) e.g., (3,100,1536)
        x=self.down_proj(gate*x)                      # Element-wise product then linear proj: (B, T, inter_dim) -> (B, T, hid_dim) e.g., (3,100,576)

        return x                                       # (B, T, hid_dim) e.g., (3,100,576)

class LanguageModelBlock(nn.Module):
    """
    A single transformer block consisting of attention and MLP layers.
    """
    def __init__(self,cfg:LMConfig): # Added type hint for cfg
        super().__init__()
        self.norm1=RMSNorm(cfg)
        self.norm2=RMSNorm(cfg)
        self.attn=LanguageModelGroupedQuaryAttention(cfg)
        self.mlp=LanguageModelMLP(cfg) # This is the MLP, not MoE

    def forward(self,x:Tensor,cos:Tensor,sin:Tensor,attention_mask=None,block_kv_cache=None): # Corrected attaintion_mask to attention_mask, added type hints
         res= x # Residual connection: (B, T_curr, C)
         x=self.norm1(x) # (B, T_curr, C)
         x,block_kv_cache=self.attn(x,cos,sin,attention_mask,block_kv_cache) # Pass attention_mask to attention: x (B, T_curr, C), block_kv_cache (dict)
         x= res+x # Add residual connection

         res=x # Residual connection: (B, T_curr, C)
         x=self.norm2(x) # (B, T_curr, C)
         x=self.mlp(x) # (B, T_curr, C)
         x=res+x # Add residual connection

         return x,block_kv_cache # (B, T_curr, C), dict
    

class LanguageModel(nn.Module):
    """
    Main Language Model class.
    """
    def __init__(self,cfg:LMConfig):
        super().__init__()
        self.cfg=cfg
        self.use_token= cfg.use_token
        self.tie_weight=cfg.tie_weight

        self.token_emb=nn.Embedding(cfg.vocal_size,cfg.hid_dim)   # Input: Token IDs (B, T_curr), Output: Embeddings (B, T_curr, hid_dim) e.g., (49153,576) -> (3,100,576)
        self.rotary_emb= RotaryEmbbeding(cfg)
        self.blocks= nn.ModuleList([LanguageModelBlock(cfg) for _ in range(cfg.n_block)])
        self.norm= RMSNorm(cfg)  # Final normalization layer
        self.head= nn.Linear(cfg.hid_dim,cfg.vocal_size,bias=False) # Language modeling head: Input (B, T_curr, hid_dim), Output (B, T_curr, vocal_size)

        if self.tie_weight:
            self.head.weight= self.token_emb.weight

        

    def forward(self,x, attention_mask: Tensor =None, kv_cache=None, start_pos: int =0, targets =None): # Corrected attaintion_mask to attention_mask, added type hints
        """
        Forward pass of the Language Model.
        Args:
            x (Tensor): Input tensor. If self.use_token is True, it's token IDs (batch_size, seq_len).
                        If self.use_token is False, it's pre-computed embeddings (batch_size, seq_len, hid_dim).
            attention_mask (Tensor, optional): Mask for attention. Defaults to None.
            kv_cache (list, optional): List of KV caches for each block. Defaults to None.
            start_pos (int, optional): Starting position for rotary embeddings (for incremental decoding). Defaults to 0.
            targets (Tensor, optional): Target labels for loss calculation. Defaults to None.
        """
        if self.use_token:
            x=self.token_emb(x)       # Input: (B, T_curr), Output: (B, T_curr, C) e.g., (3,1,1,49153) -> (3,1,576) (corrected comment to reflect token embedding)

        b,t_curr,c=x.size()           # (Batch size, current sequence length, embedding dimension) e.g., (3,1,576)

        current_position_ids=torch.arange(start_pos,start_pos+t_curr,device=x.device).unsqueeze(0).expand(b,-1)
        # Generates: [start_pos,...,start_pos+T_curr-1] -> (1, T_curr) -> (B, T_curr) e.g., (3,100)

        cos,sin = self.rotary_emb(current_position_ids)  # (B, T_curr, head_dim) each e.g., (3,100,64)

        if kv_cache is None:
            # Initialize as list of None for each block
            kv_cache = [None] * len(self.blocks)
        
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            # Each block returns (output, block_kv_cache_dict)
            block_out, block_kv_cache = block(x, cos, sin, attention_mask, kv_cache[i])
            x = block_out
            new_kv_cache.append(block_kv_cache)  # ✅ Now storing dict for each block

            # Example kv_cache content after some blocks (for block_kv_cache_i):
            # (Batch, n_kv_heads, total_seq_len, head_dim) e.g., kv_cache[0] -> (3,3,100,64), kv_cache[29] -> (3,3,2900,64)

        x=self.norm(x)  # Final output from transformer blocks: (B, T_curr, C) e.g., (3,100,576)


        if self.use_token: # This condition might be problematic. Typically, head is always applied for generation.
            x= self.head(x)    # Logits output: (B, T_curr, vocal_size) e.g., (3,100,49153)

        loss = None
        if targets is not None:
            # flatten logits and targets to (batch * seq_len, vocab) and (batch * seq_len)
            # x is (B, T, V), targets is (B, T)
            x_flat = x.view(-1, x.size(-1))           # (B*T, vocal_size)
            targets_flat = targets.view(-1)           # (B*T,)
            loss = F.cross_entropy(x_flat, targets_flat, ignore_index=-100) # Use F.cross_entropy

        return x,new_kv_cache,loss # Return logits, updated KV cache, and loss


    @torch.inference_mode()
    def generate(self,inputs: Tensor,max_new_tokens: int =20, attention_mask: Tensor =None): # Added type hints and attention_mask
        """
        Generates new tokens using the Language Model.
        Args:
            inputs (Tensor): Initial input tokens (batch_size, initial_seq_len).
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 20.
            attention_mask (Tensor, optional): Initial attention mask for the input prompt. Defaults to None.
        """
        if inputs.dim()==1:
            inputs=inputs.unsqueeze(0)    # (1, num_tokens) if input was (num_tokens,)

        generated_outputs= inputs.clone()          # (B, T_initial) or (B, T_initial, C)
        
        # Initial forward pass for the prompt
        # Pass the initial attention_mask to the first forward call
        prompt_output, kv_cache_list, _ = self.forward(generated_outputs, attention_mask=attention_mask)
        last_output= prompt_output[:,-1,:]            # Get last token's logits/embeddings: (B, vocal_size) or (B, hid_dim) e.g., (3,49153)

        for i in range(max_new_tokens):
            if self.use_token:
                next_token_ids= torch.argmax(last_output,dim=-1,keepdim=True)   # Select highest logit token: (B, 1)
                next_input = next_token_ids # Input for next step is the generated token ID
            else:
                next_input= last_output.unsqueeze(1)    # (B, 1, hid_dim) if using embeddings directly

            generated_outputs=torch.cat((generated_outputs,next_input),dim=1)
                # Concatenate new token/embedding: from (B, T_curr) to (B, T_curr + 1) for token IDs, etc.
                # Example progression: e.g., from (3,2) to (3,3) for token IDs

            current_token_start_pos= generated_outputs.size(1) - 1  # Starting position for RoPE for the next token e.g., current sequence length - 1

            if i == max_new_tokens - 1:
                break # Stop if max tokens reached

            # For subsequent decoding steps, attention_mask is typically None because KV cache handles past context.
            decode_step_output, kv_cache_list, _ = self.forward(
                next_input,
                attention_mask=None, # No attention mask for single token generation with KV cache
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )

            last_output= decode_step_output[:,-1,:] # Get the output (logits/embeddings) for the newly generated token

        return generated_outputs      # Final generated outputs: (B, T_initial + max_new_tokens) if use_token=True, else (B, T_initial + max_new_tokens, hid_dim)



    def save_pretrained(self, save_dir):
        Path(save_dir).mkdir(exist_ok=True)
        
        state_dict = self.state_dict()
        if self.cfg.tie_weight:
            state_dict.pop('head.weight', None)
        
        torch.save(state_dict, f"{save_dir}/pytorch_model.bin")
        
        config = {f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        if self.cfg.tokenizer:
            get_tokenizer(self.cfg.tokenizer).save_pretrained(save_dir)

