import sys; sys.path.append("../")
from dna.operations import OPS, reset
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

PRIMITIVES = ['MB6_3x3_se0.25',
              'MB6_5x5_se0.25',
              'MB3_3x3_se0.25',
              'MB3_5x5_se0.25',
              ]

class MixedOp(nn.Module):
    """Darts MixOp.  Removing the hidden outc and dispatch """
    def __init__(self, inc, outc, stride):
        super(MixedOp, self).__init__()
        self._mix_ops = nn.ModuleList()
        for prim in PRIMITIVES:
            self._mix_ops.append(OPS[prim](inc, outc, stride))

    def forward(self, x, weights):
        """ Return Sum(op_i * w_i) """
        return sum(w * op(x) for w, op in zip(weights, self._mix_ops))


class DartsSupernet(nn.Module):
    """using first-order approximation """
    def __init__(self, num_classes):
        super(DartsSupernet, self).__init__()
        self.block_cfgs = [[24,  2, 2],
                           [40,  2, 4],
                           [80,  2, 4],
                           [112, 1, 4],
                           [192, 2, 4],
                           [320, 1, 1]]
        self._num_classes = num_classes
        self._layers = sum([ele[-1] for ele in self.block_cfgs])
        self._stem = nn.Sequential(OPS['Conv3x3_BN_swish'](3, 32, 2),
                                OPS['MB1_3x3_se0.25'](32, 16, 1))
        self._make_block(self.block_cfgs)
        self._stern = OPS['Conv1x1_BN_swish'](320, 1280, 1)
        self._avgpool = nn.AvgPool2d(7)
        self._linear = nn.Linear(1280, num_classes)
        self._num_of_ops = len(PRIMITIVES)

        self._alpha = Variable(1e-3*torch.randn(self._layers, self._num_of_ops), requires_grad=True)
        if torch.cuda.is_available():
            # local pc
            self._alpha.cuda()

    def _make_block(self, block_cfgs, inc=16):
        self._blocks = nn.ModuleList()
        block_layer_index = 0
        for sing_cfg in block_cfgs:
            outc, stride, layers = sing_cfg[0], sing_cfg[1], sing_cfg[2]
            for i in range(layers):
                stride = stride if i==0 else 1
                self._blocks.append(
                    MixedOp(inc, outc, stride))
                inc = outc

    def forward(self, x):
        x = self._stem(x)
        for i, block in enumerate(self._blocks):
            weights = F.softmax(self._alpha[i], dim=-1) 
            x = block(x, weights)

        x = self._stern(x)
        x = self._avgpool(x)
        logits = self._linear(x.view(x.size(0), -1))
        return logits

    @property
    def alpha(self):
        if torch.cuda.is_available():
            return self._alpha.detach().cpu().numpy()
        else:
            return self._alpha.detach().numpy()

if __name__ == "__main__":
    model = MixedOp(3, 20, 1)
    x = torch.ones(1, 3, 224,224)
    weights =  Variable(1e-3*torch.randn(4), requires_grad=True)
    print(weights.shape)
    print(model(x, weights).shape)
    supernet = DartsSupernet(10)
    print(supernet(x).shape)
    print(supernet.alpha)