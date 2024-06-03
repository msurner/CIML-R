import torch
from torch import nn
from torch.nn.functional import normalize

class LrpNetwork(nn.Sequential):
    def __init__(self, *args):
        super(LrpNetwork, self).__init__(*args)
        self.z = len(self) * [None]
        self.a = len(self) * [None]
        self.LRP_epsilon = 1e-3 # for LRP method, eps=0.25 std in Montavon, 2019
        self.stability_epsilon = 1e-12 # for numerical stability 
        self.rho = (
            ["zero_rule"] * len(self)
        )
#       Can be also a mixture of rules:
#       ( 
#            [lambda w: self.gamma_rule(w, gamma=.25)] * (len(self)//3 + min(1, len(self)%3)) +
#            ["epsilon_rule"] * (len(self)//3 + max(0, len(self)%3-1)) +
#            ["zero_rule"] * (len(self)//3)
#        )

    def forward(self, x):
        for i, (layer, rho) in enumerate(zip(self, self.rho)):
            x_new = layer.forward(x)
            if not isinstance(layer, torch.nn.ReLU):
                # a =(aj)j is the vector of lower-layer activations
                self.a[i] = x
                # apply rules
                if rho == "zero_rule":
                    self.z[i] = x_new
                elif rho == "epsilon_rule":
                    self.z[i] = self.LRP_epsilon + x_new
                else:
                    self.z[i] = x @ rho(layer.weight.T) + rho(layer.bias)
                self.z[i] += self.stability_epsilon
            x = x_new
            
        self.relevance = self.relevance_propagation(x)
        return x

    def relevance_propagation(self, x):
        # Initial relevance scores are the network's output activations
        # https://github.com/kaifishr/PyTorchRelevancePropagation/blob/master/src/lrp.py#L110
        # Note that PyTorchs CrossEntropy includes the softmax function, so x is assumed to haven't been
        # applied to a softmax function before. That's why we do a softmax here.
        relevance = torch.softmax(x, dim=-1)
        for layer, z, a, rho  in zip(reversed(self), reversed(self.z), reversed(self.a), reversed(self.rho)):
            if isinstance(layer, torch.nn.ReLU):
                # https://github.com/kaifishr/PyTorchRelevancePropagation/blob/master/src/lrp_layers.py#L219
                continue
            # https://github.com/kaifishr/PyTorchRelevancePropagation/blob/master/src/lrp_layers.py#L180
            s = relevance / z
            if callable(rho):
                weights = rho(layer.weight)
            else:
                weights = layer.weight
            c = torch.mm(s, weights) 
            relevance = a * c

        return relevance

    def gamma_rule(self, w, gamma=.25, min=None, max=None):
        """
        Applies the gamma rule, which favors the positive weights.
        """
        weights = w + gamma * torch.clamp(w, min=0.)
        if not (max is None and min is None):
            weights = weights.clip(min=min, max=max)
        return weights
