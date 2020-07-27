import torch.nn as nn
import torch.nn.functional as F

from timm_.models.gen_efficientnet import (
    InvertedResidual, DepthwiseSeparableConv, ConvBnAct,
    swish, hard_sigmoid, hard_swish, sigmoid,
    drop_connect,
)

# GLOBAL PARAMS
# AFFINE = False
BN_MOMENTUM_PT_DEFAULT = 0.1
BN_EPS_PT_DEFAULT = 1e-5
BN_ARGS_PT = dict(momentum=BN_MOMENTUM_PT_DEFAULT, eps=BN_EPS_PT_DEFAULT)

BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
BN_EPS_TF_DEFAULT = 1e-3
BN_ARGS_TF = dict(momentum=BN_MOMENTUM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)

BN_ARGS = BN_ARGS_PT  # Chose bn args here

#  Some params take the default, You can find more infomation in gen_efficientnet
OPS = {
    'MB6_3x3_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB6_5x5_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 5, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB6_7x7_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 7, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB3_3x3_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB3_5x5_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 5, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB6_3x3': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0, bn_args=BN_ARGS),
    'MB6_5x5': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 5, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0, bn_args=BN_ARGS),
    'MB6_7x7': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 7, stride, act_fn=swish,
                         exp_ratio=6.0, se_ratio=0, bn_args=BN_ARGS),
    'MB3_3x3': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0, bn_args=BN_ARGS),
    'MB3_5x5': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0., bn_args=BN_ARGS),
    'MB3_7x7': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 3, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0, bn_args=BN_ARGS),
    'MB3_7x7_se0.25': lambda in_channels, out_channels, stride: \
        InvertedResidual(in_channels, out_channels, 7, stride, act_fn=swish,
                         exp_ratio=3.0, se_ratio=0.25, bn_args=BN_ARGS),
    'MB1_3x3_se0.25': lambda in_channels, out_channels, stride: \
        DepthwiseSeparableConv(in_channels, out_channels, 3, stride=stride, act_fn=swish,
                               se_ratio=0.25, bn_args=BN_ARGS),
    'MB1_5x5_se0.25': lambda in_channels, out_channels, stride: \
        DepthwiseSeparableConv(in_channels, out_channels, 5, stride=stride, act_fn=swish,
                               se_ratio=0.25, bn_args=BN_ARGS),
    'MB1_7x7_se0.25': lambda in_channels, out_channels, stride: \
        DepthwiseSeparableConv(in_channels, out_channels, 7, stride=stride, act_fn=swish,
                               se_ratio=0.25, bn_args=BN_ARGS),
    'Conv1x1_BN_swish': lambda in_channels, out_channels, stride: \
        ConvBnAct(in_channels, out_channels, 1, stride=stride, act_fn=swish,
                  bn_args=BN_ARGS),
    'Conv3x3_BN_swish': lambda in_channels, out_channels, stride: \
        ConvBnAct(in_channels, out_channels, 3, stride=stride, act_fn=swish,
                  bn_args=BN_ARGS)
}


def reset(m):
    # reset conv2d/linear/BN in Block
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        m.reset_parameters()


def activation(func='relu', inplace=True):
    """activate function"""
    acti = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU(inplace=inplace)],
        ['prelu', nn.PReLU()],
        ['relu', nn.ReLU(inplace=inplace)],
        ['none', Identity()],
        ['relu6', nn.ReLU6(inplace=inplace)],
        ['swish', swish(inplace=inplace)],
        ['sigmoid', sigmoid(inplace=inplace)],
        ['hard_sigmoid', hard_sigmoid(inplace=inplace)],
        ['hard_swish', hard_swish(inplace=inplace)]])

    return acti[func]


# BACKUP: FOR BN FIRST OP
class InvertedResidual_BnFirst(InvertedResidual):
    # ISSUE: the resiudal structure cannot move
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid,
                 shuffle_type=None, bn_args=BN_ARGS, drop_connect_rate=0.):
        super(InvertedResidual_BnFirst, self).__init__(in_chs, out_chs, dw_kernel_size,
                                                       stride, pad_type, act_fn, noskip,
                                                       exp_ratio, exp_kernel_size,
                                                       pw_kernel_size, se_ratio,
                                                       se_reduce_mid, se_gate_fn,
                                                       shuffle_type, bn_args,
                                                       drop_connect_rate)
        self._bn0 = nn.BatchNorm2d(in_chs, **bn_args)

    def forward(self, x):
        residual = x

        x = self._bn0(x)
        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        # x = self.bn3(x) # move this bn to next operation

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
