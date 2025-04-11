# Source: https://github.com/karpathy/nanoGPT
#
# MIT License
#
# Copyright (c) 2022 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications:
# - Added data_adapters to GPT to preprocess the inputs and (optionally) postprocess the outputs
# - Added the `skip` option to concat the input and output of the network before the final projection
# - Added time `t` as an input to `forward()`

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_model import pe_encode
from torch import Tensor


def gelu(x):
    return F.gelu(x, approximate="tanh")


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, is_causal):
        super().__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.is_causal = is_causal

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, is_causal):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttention(n_head, n_embd, dropout, bias, is_causal)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        data_adapters: dict,
        vocab_size: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        skip: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.input_adapter = data_adapters["input_adapter"]
        self.output_adapter = data_adapters["output_adapter"]
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(n_head, n_embd, dropout, bias, is_causal) for _ in range(n_layer)]),
                ln_f=LayerNorm(n_embd, bias=bias),
            )
        )
        self.is_causal = is_causal
        if self.is_causal:
            self.skip = False
        else:
            self.skip = skip
        if skip:
            self.lm_head = nn.Linear(2 * n_embd, vocab_size, bias=bias)
        else:
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # report number of parameters
        print(f"number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        # print(data.device)
        # print(t.device)
        x_in = self.input_adapter(data, t, conditions)
        x = self.transformer.drop(x_in)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if self.skip:
            x = torch.cat([x, x_in], -1)
        logits = self.output_adapter(self.lm_head(x)) if self.output_adapter else self.lm_head(x)
        return logits

    def get_optim_groups(self, weight_decay: float):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # We don't use weight tying so comment this out
        # decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
    
    
class DynamicCNN(nn.Module):
    def __init__(self, num_conditions=2):
        super(DynamicCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(128)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)  # 50% 的 dropout

        # 输出层
        self.fc = nn.Linear(128, num_conditions)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))      
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))     
        
        # 使用全局池化
        x = self.global_pool(x)
        x = x.view(-1, 128)  # 128 是通道数

        x = self.dropout(x)  # 添加 Dropout
        x = self.fc(x)  # 直接连接到输出层
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output))

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, num_heads=4, ff_dim=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)  
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)  
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [bs, seq_len, vocab_size] -> [bs, seq_len, embed_dim]
        x = x.transpose(0, 1)  # 转置为 [seq_len, bs, embed_dim] 以适应 MultiheadAttention
        x = self.transformer_block(x)
        x = x.transpose(0, 1)  # 转回 [bs, seq_len, embed_dim]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)  # 池化
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TextInputClassifier(nn.Module):
    """
    A module to convert sequences of text class tokens to embedding tokens with learned positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        output_size: int = 256,
        num_conditions: int = 1,
        learn_pos_embedding: bool = False,
        has_t: bool = False
    ):
        super().__init__()
        self.learn_pos_embedding = learn_pos_embedding
        self.num_conditions = num_conditions
        self.has_t = has_t
        if learn_pos_embedding:
            self.pos_embedding = nn.Embedding(seq_len, vocab_size)
        else:
            self.register_buffer("pos_embedding", pe_encode(seq_len, vocab_size))
        self.inp_embedding = nn.Linear(vocab_size, vocab_size)
        
        if has_t:
            self.t_embedding = nn.Linear(1, vocab_size)
        
        # self.cnn = DynamicCNN(num_conditions=num_conditions)
        
        self.transformer = TransformerModel(vocab_size, num_conditions)

    def forward(self, probs: torch.Tensor, t: torch.Tensor) -> Tensor:
        # print(probs.dtype)
        inp_emb = self.inp_embedding(2 * probs - 1)
        if self.learn_pos_embedding:
            pos_emb = self.pos_embedding(
                torch.arange(0, probs.size(1)).to(probs.device)
            )
        else:
            pos_emb = self.pos_embedding
        pos_emb = pos_emb.unsqueeze(0).expand(inp_emb.size(0), -1, -1)
        
        if self.has_t:
            t_emb = self.t_embedding((2 * t - 1).unsqueeze(-1))            
            output = inp_emb + pos_emb + t_emb  # [bs, seq_len, vocab_size]        
            # output = torch.cat([inp_emb, pos_emb, t_emb], dim=-1) # [bs, seq_len, vocab_size * 3]   
                
        else:
            output = inp_emb + pos_emb
        
        # output = output.unsqueeze(1)
        
        out = self.transformer(output)

        return out


