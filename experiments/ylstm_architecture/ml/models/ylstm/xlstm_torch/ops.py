# Copyright 2026 Kilian Ludwig
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def slstm(s: torch.Tensor, r: torch.Tensor, state: torch.Tensor, eps: float =1e-6):
    hs = []
    for st in torch.unbind(s, dim=1):
        h, state = slstm_cell(st, r, state, eps)
        hs.append(h)

    return torch.stack(hs, dim=1), state


@torch.compile()
def slstm_cell(s: torch.Tensor, r: torch.Tensor, state: torch.Tensor, eps: float =1e-6):
    h, c, n, m = state
    si, sf, sz, so = torch.chunk(s + h @ r.T, 4, dim=-1)
    sf = torch.nn.functional.logsigmoid(sf) + m
    m = torch.maximum(si, sf)
    i, f = torch.exp(si - m), torch.exp(sf - m)
    c = f * c + i * torch.tanh(sz)
    n = f * n + i
    h = torch.sigmoid(so) * c / (n + eps)
    return h, torch.stack([h, c, n, m])


@torch.compile()
def mlstm_cell(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    si: torch.Tensor,
    sf: torch.Tensor,
    state: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ],
    eps: float = 1e-6
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    c, n, m = state
    sf = torch.nn.functional.logsigmoid(sf) + m
    m = torch.maximum(si, sf)
    i, f = torch.exp(si - m), torch.exp(sf - m)
    k = k / k.shape[-1] ** .5
    c = f.unsqueeze(-2) * c + i.unsqueeze(-2) * v.unsqueeze(-1) * k.unsqueeze(-2)
    n = f * n + i * k
    _n = torch.sum(n * q, dim=-1, keepdim=True).abs()
    h = torch.sum(c * q.unsqueeze(-2), dim=-1) / torch.maximum(_n, torch.exp(-m) + eps)
    return h, (c, n, m)


@torch.compile()
def mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    si: torch.Tensor,
    sf: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    sf = torch.nn.functional.logsigmoid(sf)
    sf_repeat = torch.tile(sf, (si.shape[-2], ))
    f_mat = torch.tril(sf_repeat, diagonal=-1)
    log_prod_f = torch.cumsum(f_mat, dim=-2)
    upper = torch.triu(torch.full_like(log_prod_f, -torch.inf), diagonal=1)

    log_g = log_prod_f + upper + si.movedim(-1, -2)
    max_log_g = torch.amax(log_g, dim=-1, keepdim=True)

    c, n, m = state
    sf_cum = torch.cumsum(sf, dim=-2) + m.unsqueeze(-2)
    m = torch.maximum(max_log_g, sf_cum)
    f = torch.exp(sf_cum - m)
    g = torch.exp(log_g - m)

    k = k / k.shape[-1] ** .5
    e = g * torch.sum(q.unsqueeze(-2) * k.unsqueeze(-3), dim=-1)
    n_par = torch.sum(e, dim=-1, keepdim=True)
    c_par = torch.sum(e.unsqueeze(-1) * v.unsqueeze(-3), dim=-2)

    q_fgate = f * q
    c_rec = torch.sum(q_fgate.unsqueeze(-2) * c.unsqueeze(-3), dim=-1)
    n_rec = torch.sum(q_fgate * n.unsqueeze(-2), dim=-1, keepdim=True)

    c_both = c_par + c_rec
    n_both = torch.abs(n_par + n_rec)
    _n = torch.maximum(n_both, torch.exp(-m) + eps)

    k_gates = k * g[..., -1, :].unsqueeze(-1)
    vk = torch.sum(v.unsqueeze(-1) * k_gates.unsqueeze(-2), dim=-3)
    c = f[..., -1, :].unsqueeze(-1) * c + vk
    n = f[..., -1, :] * n + torch.sum(k_gates, dim=-2)
    m = m[..., -1, :]

    return c_both / _n, (c, n, m)


@torch.compile()
def mlstm_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    si: torch.Tensor,
    sf: torch.Tensor,
    eps: float = 1e-6,
    bidirectional: bool = False
) -> torch.Tensor:
    sf = torch.nn.functional.logsigmoid(sf)
    sf_repeat = torch.tile(sf, (si.shape[-2], ))
    f_mat = torch.tril(sf_repeat, diagonal=-1)
    log_prod_f = torch.cumsum(f_mat, dim=-2)
    if bidirectional:
        f_mat_rev = torch.triu(torch.roll(sf_repeat, 1, dims=-2).T, diagonal=1)
        raise NotImplementedError("no efficient way to add si to diagonal yet")
        upper = torch.cumsum(f_mat_rev, dim=-1)
    else:
        upper = torch.triu(torch.full_like(log_prod_f, -torch.inf), diagonal=1)

    log_g = log_prod_f + upper + si.movedim(-1, -2)
    max_log_g = torch.amax(log_g, dim=-1, keepdim=True)
    g = torch.exp(log_g - max_log_g)

    k = k / k.shape[-1] ** .5
    e = g * torch.sum(q.unsqueeze(-2) * k.unsqueeze(-3), dim=-1)
    n = torch.sum(e, dim=-1, keepdim=True).abs()
    att = e / torch.maximum(n, torch.exp(-max_log_g) + eps)
    return torch.sum(att.unsqueeze(-1) * v.unsqueeze(-3), dim=-2)
