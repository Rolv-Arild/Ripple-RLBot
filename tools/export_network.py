"""Exportable form of training's agent2.Ripple, loaded strictly from its weights.

This is only used during conversion. Keep its outputs checked against the
original TorchScript, including both side flags and padded-player masks.
"""
import math
from torch_runtime import torch
from torch import nn
import torch.nn.functional as F


def mlp(input_dim, dim, output_dim, hidden_layers=2):
    layers = [nn.Linear(input_dim, dim)]
    for _ in range(hidden_layers):
        layers += [nn.GELU(), nn.Linear(dim, dim)]
    return nn.Sequential(*layers, nn.GELU(), nn.Linear(dim, output_dim))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_qkv = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4,dim))

    def forward(self, q, kv, mask):
        # Same packed Q/KV projection as the original MultiheadAttention graph,
        # using explicit reshapes so ONNX can retain a dynamic batch dimension.
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        dim, heads = self.dim, self.heads
        w, bias = self.attn.in_proj_weight, self.attn.in_proj_bias
        query = F.linear(qn, w[:dim], bias[:dim])
        key, value = F.linear(kvn, w[dim:], bias[dim:]).chunk(2, dim=-1)
        batch = q.shape[0]
        query = query.reshape(batch, 1, heads, dim//heads).transpose(1,2)
        key = key.reshape(batch, -1, heads, dim//heads).transpose(1,2)
        value = value.reshape(batch, -1, heads, dim//heads).transpose(1,2)
        scores = torch.matmul(query / math.sqrt(dim//heads), key.transpose(-2,-1))
        scores = scores.masked_fill(mask[:,None,None,:], float('-inf'))
        attended = torch.matmul(scores.softmax(-1), value)
        attended = attended.transpose(1,2).reshape(batch, 1, dim)
        result = q + self.attn.out_proj(attended)
        return result + self.mlp(self.norm_qkv(result))


class ExportNetwork(nn.Module):
    def __init__(self, weights, heads=4):
        super().__init__()
        weights = {key.removeprefix('net.'): value for key,value in weights.items()}
        dim = weights['main_player_processor.0.weight'].shape[0]
        count = len({key.split('.')[1] for key in weights if key.startswith('layers.')})
        self.main_player_processor = mlp(30, dim, dim)
        self.other_players_processor = mlp(21, dim, dim)
        self.ball_processor = mlp(54, dim, dim)
        self.action_queue_processor = mlp(48, dim, dim)
        self.boost_processor = mlp(5, dim//4, dim)
        self.layers = nn.ModuleList(CrossAttention(dim, heads) for _ in range(count))
        self.state_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(dim)
        self.action_option_processor = mlp(8, dim, dim)
        for key in ('lookup_table','invert_actions','invert_map'):
            self.register_buffer(key, torch.empty_like(weights[key]))
        self.load_state_dict(weights, strict=True)
        self.eval()
        with torch.no_grad():
            self.register_buffer('action_embeddings', self.action_norm(
                self.action_option_processor(self.lookup_table)).clone())

    def forward(self, main, others, ball, actions, boosts):
        mask = torch.cat((others[:,:,15] == 0,
                          torch.zeros((main.shape[0],3), dtype=torch.bool, device=main.device)), dim=1)
        query = self.main_player_processor(main[:,:30]).unsqueeze(1)
        kv = torch.cat((self.other_players_processor(others),
            self.ball_processor(ball.reshape(-1,1,54)),
            self.action_queue_processor(actions.reshape(-1,1,48)),
            self.boost_processor(boosts).amax(dim=1, keepdim=True)), dim=1)
        for layer in self.layers:
            query = layer(query,kv,mask)
        logits = self.state_norm(query.squeeze(1)) @ self.action_embeddings.T
        return torch.where(main[:,30:31] > 0.5, logits[:,self.invert_map], logits)
