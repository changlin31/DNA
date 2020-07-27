from torch import optim as optim
from timm_.optim import Nadam, RMSpropTF


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), args=None, stage=0):
    # if not args.label_train:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    # else:
    #     prev_decay = []
    #     prev_nodecay = []
    #     after_decay = []
    #     after_nodecay = []
    #     if hasattr(model, 'module'):
    #         model = model.module
    #     for block in model._blocks[:stage+1]:
    #         for name, param in block.named_parameters():
    #             if not param.requires_grad:
    #                 continue  # frozen weights
    #             if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
    #                 prev_nodecay.append(param)
    #             else:
    #                 prev_decay.append(param)
    #     for block in model._blocks[stage+1:]:
    #         for name, param in block.named_parameters():
    #             if not param.requires_grad:
    #                 continue  # frozen weights
    #             if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
    #                 after_nodecay.append(param)
    #             else:
    #                 after_decay.append(param)
    #     return [
    #         {'params': prev_nodecay, 'lr': args.lr, 'weight_decay': 0.},
    #         {'params': prev_decay, 'lr': args.lr, 'weight_decay': weight_decay},
    #         {'params': after_nodecay, 'lr': args.lr, 'weight_decay': 0.},
    #         {'params': after_decay, 'lr': args.lr, 'weight_decay': weight_decay},
    #     ]

def create_optimizer(args, model, filter_bias_and_bn=True, stage=0):
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay, args=args, stage=stage)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    # if args.label_train and not filter_bias_and_bn:
    #     parameters = [
    #         {"params": model._blocks[stage+1:].parameters(), "lr": args.lr / args.loss_weight[-1]},
    #         {"params": model._stern.parameters(), "lr": args.lr / args.loss_weight[-1]},
    #         {"params": model._avgpool.parameters(), "lr": args.lr / args.loss_weight[-1]},
    #         {"params": model._linear.parameters(),"lr": args.lr / args.loss_weight[-1]}
    #         ]
    if isinstance(args.lr, list):
        if args.opt.lower() == 'sgd':
            optimizer = optim.SGD(
                parameters, lr=args.lr[stage],
                momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
        elif args.opt.lower() == 'adam':
            optimizer = optim.Adam(
                parameters, lr=args.lr[stage], weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'nadam':
            optimizer = Nadam(
                parameters, lr=args.lr[stage], weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'adadelta':
            optimizer = optim.Adadelta(
                parameters, lr=args.lr[stage], weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'rmsprop':
            optimizer = optim.RMSprop(
                parameters, lr=args.lr[stage], alpha=0.9, eps=1.,
                momentum=args.momentum, weight_decay=weight_decay)
        elif args.opt.lower() == 'rmsproptf':
            optimizer = RMSpropTF(
                parameters, lr=args.lr[stage], alpha=0.9, eps=args.opt_eps,
                momentum=args.momentum, weight_decay=weight_decay)
        else:
            assert False and "Invalid optimizer"
            raise ValueError
    else:
        if args.opt.lower() == 'sgd':
            optimizer = optim.SGD(
                parameters, lr=args.lr,
                momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
        elif args.opt.lower() == 'adam':
            optimizer = optim.Adam(
                parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'nadam':
            optimizer = Nadam(
                parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'adadelta':
            optimizer = optim.Adadelta(
                parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
        elif args.opt.lower() == 'rmsprop':
            optimizer = optim.RMSprop(
                parameters, lr=args.lr, alpha=0.9, eps=1.,
                momentum=args.momentum, weight_decay=weight_decay)
        elif args.opt.lower() == 'rmsproptf':
            optimizer = RMSpropTF(
                parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
                momentum=args.momentum, weight_decay=weight_decay)
        else:
            assert False and "Invalid optimizer"
            raise ValueError
    return optimizer
