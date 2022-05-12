import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0))
        return -p_log_p.sum(-1)

multi_categoricals = CategoricalMasked(logits=torch.tensor([0.1,0.2,0.3,0.4,0.5]), masks=torch.tensor([0,0,1,0,0])) 
action = multi_categoricals.sample()
print(action)