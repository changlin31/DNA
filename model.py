# This model script Based on impl: "https://github.com/rwightman/pytorch-image-models/timm/models/gen_efficientnet.py"

import re
import math
from copy import deepcopy
from timm.models import *
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]

def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled

def _decode_block_str(block_str, depth_multiplier=1.0):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = F.relu
            elif v == 'r6':
                value = F.relu6
            elif v == 'hs':
                value = hard_swish
            elif v == 'sw':
                value = swish
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_fn is None, the model default (passed to model init) will be used
    act_fn = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'er':
        block_args = dict(
            block_type=block_type,
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            fake_in_chs=fake_in_chs,
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_fn=act_fn,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat

def _decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil'):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args

def _resolve_bn_args(kwargs):
    bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args

def _gen_dna_net(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, num_features=1280, arch_def=None, **kwargs):
    model = GenEfficientNet(_decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        num_features = num_features,
        bn_args = _resolve_bn_args(kwargs),
        act_fn=swish,
        **kwargs
        )
    return model

def swish(x, inplace=False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x * x.sigmoid()

def DNA_a(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_arch = [['ds_r1_k3_s1_c16_se0.25'],
                  ['ir_r1_k5_s2_e3_c24_se0.25', 'ir_r1_k3_s1_e3_c24_se0.25', 'ir_r1_k3_s1_e6_c24_se0.25'],
                  ['ir_r2_k3_s2_e3_c40_se0.25', 'ir_r1_k5_s1_e3_c40_se0.25', 'ir_r1_k3_s1_e6_c40_se0.25'],
                  ['ir_r1_k7_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25', 'ir_r1_k5_s1_e3_c80_se0.25', 'ir_r1_k3_s1_e6_c80_se0.25'],
                  ['ir_r1_k5_s1_e6_c96_se0.25', 'ir_r2_k5_s1_e3_c96_se0.25', 'ir_r1_k5_s1_e6_c96_se0.25'],
                  ['ir_r1_k5_s2_e6_c160_se0.25', 'ir_r3_k5_s1_e3_c160_se0.25', 'ir_r1_k5_s1_e6_c160_se0.25'],
                  ['ir_r1_k3_s1_e6_c320_se0.25']]

    default_cfg = _cfg(url='')
    kwargs['drop_connect_rate'] = 0.2  # Open in train
    model = _gen_dna_net(channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, arch_def=model_arch, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        raise NotImplementedError
    return model

def DNA_b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_arch = [['ds_r1_k3_s1_c16_se0.25'],
                ['ir_r1_k3_s2_e3_c24_se0.25', 'ir_r1_k5_s1_e3_c24_se0.25', 'ir_r1_k7_s1_e6_c24_se0.25'],
                ['ir_r1_k7_s2_e6_c40_se0.25', 'ir_r1_k5_s1_e3_c40_se0.25', 'ir_r1_k7_s1_e6_c40_se0.25'],
                ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r2_k5_s1_e3_c80_se0.25', 'ir_r1_k3_s1_e6_c80_se0.25'],
                ['ir_r1_k5_s1_e6_c96_se0.25', 'ir_r2_k5_s1_e3_c96_se0.25', 'ir_r1_k5_s1_e6_c96_se0.25'],
                ['ir_r1_k7_s2_e6_c192_se0.25', 'ir_r3_k5_s1_e3_c192_se0.25', 'ir_r1_k7_s1_e6_c192_se0.25'],
                ['ir_r1_k3_s1_e6_c320_se0.25']]

    default_cfg = _cfg(url='')
    kwargs['drop_connect_rate'] = 0.2  # Open in train
    model = _gen_dna_net(channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, arch_def=model_arch, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        raise NotImplementedError
    return model

def DNA_c(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_arch = [['ds_r1_k3_s1_c16_se0.25'],
                  ['ir_r1_k5_s2_e6_c24_se0.25', 'ir_r1_k3_s1_e6_c24_se0.25'],
                  ['ir_r2_k5_s2_e6_c40_se0.25', 'ir_r1_k3_s1_e6_c40_se0.25', 'ir_r1_k5_s1_e6_c40_se0.25'],
                  ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e6_c80_se0.25', 'ir_r2_k5_s1_e6_c80_se0.25'],
                  ['ir_r1_k3_s1_e6_c112_se0.25', 'ir_r2_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e3_c112_se0.25'],
                  ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e3_c192_se0.25', 'ir_r2_k5_s1_e6_c192_se0.25'],
                  ['ir_r1_k3_s1_e6_c320_se0.25']]

    default_cfg = _cfg(url='')
    kwargs['drop_connect_rate'] = 0.2  # Open in train
    model = _gen_dna_net(channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, arch_def=model_arch, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        raise NotImplementedError
    return model

def DNA_d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_arch = [['ds_r1_k3_s1_c16_se0.25'],
                  ['ir_r1_k7_s2_e6_c24_se0.25', 'ir_r1_k5_s1_e6_c24_se0.25', 'ir_r1_k7_s1_e6_c24_se0.25'],
                  ['ir_r4_k7_s2_e6_c40_se0.25'],
                  ['ir_r1_k7_s2_e6_c80_se0.25', 'ir_r2_k5_s1_e6_c80_se0.25', 'ir_r1_k7_s1_e6_c80_se0.25'],
                  ['ir_r1_k7_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k7_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'],
                  ['ir_r1_k7_s2_e6_c192_se0.25', 'ir_r3_k5_s1_e6_c192_se0.25', 'ir_r1_k7_s1_e6_c192_se0.25'],
                  ['ir_r1_k3_s1_e6_c320_se0.25']]

    default_cfg = _cfg(url='')
    kwargs['drop_connect_rate'] = 0.2  # Open in train
    model = _gen_dna_net(channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, arch_def=model_arch, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        raise NotImplementedError
    return model
