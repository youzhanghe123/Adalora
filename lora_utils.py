import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class AdaLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_E = nn.Parameter(self.weight.new_zeros(r, 1))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.ranknum = nn.Parameter(self.weight.new_zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ (self.lora_A * self.lora_E)) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += T(self.lora_B @ (self.lora_A * self.lora_E)) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T) * self.scaling / (self.ranknum + 1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class RankAllocator:
    def __init__(
        self, 
        model, 
        lora_r: int,
        target_rank: int,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float = 0.85,
        beta2: float = 0.95,
        total_step: Optional[int] = None,
        target_total_rank: Optional[int] = None,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.rank_pattern = {}
        self._get_lora_param_names()

    def _get_lora_param_names(self):
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0)
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    def update_and_mask(self, model, global_step):
        if global_step < self.total_step - self.final_warmup:
            self._update_importance_scores(model)
        
        curr_rank, mask_ind = self._schedule_threshold(global_step)
        if mask_ind:
            mask_threshold = self._mask_to_target_rank(model, curr_rank)
        else:
            mask_threshold = None
        
        return curr_rank, mask_threshold

    def _update_importance_scores(self, model):
        for n, p in model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def _schedule_threshold(self, step):
        mask_ind = False
        if step <= self.initial_warmup:
            curr_rank = self.total_rank
            mask_ind = False
        elif step > self.total_step - self.final_warmup:
            curr_rank = self.target_rank
            mask_ind = True
        else:
            mul_coeff = 1 - (step - self.initial_warmup) / (self.total_step - self.final_warmup - self.initial_warmup)
            curr_rank = self.target_rank + (self.total_rank - self.target_rank) * (mul_coeff ** 3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False
        return curr_rank, mask_ind

    def _mask_to_target_rank(self, model, curr_rank):
        is_dict = {}
        combine_dict = {}
        singular_dict = {}
        
        for n, p in model.named_parameters():
            if "lora_A" in n:
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n:
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = ipt_E.view(-1) + ipt_AB.sum(dim=1)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank - curr_rank))[0].item()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_E" in n:
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    self.rank_pattern[n] = (is_dict[n] > mask_threshold).sum().item()

        return mask_threshold

def compute_adalora_orth_regu(model, regu_weight=0.1):
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param
