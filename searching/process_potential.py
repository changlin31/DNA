import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from dna.operations import OPS
from timm_.flops_counter import get_model_parameters_number


def main():
    target = 'flops'  # 'params' or 'flops'
    norm_mse = False  # NOTE: DO NOT norm when 'mul' !!
    loss_join = 'sum'  # 'sum' or 'mul'
    # b0_constrain = {'params': 5288548,
    #                 'FLOPS' : 399362}
    target_constrain = 460000
    PRIMITIVES = ['MB6_3x3_se0.25',
                  'MB6_5x5_se0.25',
                  'MB6_7x7_se0.25',
                  'MB3_3x3_se0.25',
                  'MB3_5x5_se0.25',
                  'MB3_7x7_se0.25',
                  ]

    potential_yaml = [
        ['./output/sota/adam-step-ep25-lr0.002-bs260-20191110-201108/potential-0.yaml',
         './output/sota/adam-step-ep25-lr0.002-bs260-20191109-141527/potential-0.yaml',
         './output/sota/adam-step-ep25-lr0.002-bs260-20191109-140501/potential-0.yaml'
         ],
        ['./output/sota/adam-step-ep25-lr0.002-bs260-20191109-135641/potential-1.yaml',
         './output/sota/adam-step-ep25-lr0.002-bs260-20191109-145221/potential-1.yaml',
         './output/search/adam-step-ep40-lr0.002-bs260-20191109-114241/potential-1.yaml',
         ],
        ['./output/sota/adam-step-ep25-lr0.002-bs260-20191109-135239/potential-2.yaml',
         './output/sota/adam-step-ep25-lr0.002-bs260-20191109-150337/potential-2.yaml',
         './output/search/adam-step-ep40-lr0.002-bs260-20191109-125448/potential-2.yaml'
         ],
        ['./output/search/adam-step-ep40-lr0.002-bs260-20191109-150954/potential-3.yaml',
         './output/search/adam-step-ep40-lr0.002-bs260-20191109-133629/potential-3.yaml',
         './output/search/adam-step-ep40-lr0.002-bs260-20191109-115817/potential-3.yaml'
         ],
        ['./output/search/adam-step-ep40-lr0.002-bs260-20191109-142604/potential-4.yaml',
         './output/sota/adam-step-ep40-lr0.002-bs260-20191110-202256/potential-4.yaml',
         './output/search/adam-step-ep40-lr0.002-bs260-20191109-134037/potential-4.yaml'
         ],
        # ['./output/search/adam-step-ep40-lr0.002-bs260-20191109-150525/potential-5.yaml']
    ]

    layer_cfgs = [
        [2, 2, 2, 4, 5, 1],
        [3, 3, 3, 4, 5, 1],
        [3, 3, 4, 4, 5, 1]
    ]
    block_cfgs = [
        [24, 24, 2, 3],
        [40, 40, 2, 4],
        [80, 80, 2, 4],
        [96, 96, 1, 4],
        [192, 192, 2, 5],
        [320, 320, 1, 1]
    ]
    num_op = 6

    # ================== 1. Generate Params Table and Flops Table ==================
    if target == 'params':
        n_param_table = []
        block_params = 0
        block_layer_index = 0
        spnet_idx = 0
        # for spnet_idx in range(4):
        n_param_spnet = []
        inc = 16
        for block_idx, sing_cfg in enumerate(block_cfgs):
            n_param_layer = []
            hidden_outc, outc, stride, layers = sing_cfg[0], sing_cfg[1], sing_cfg[2], \
                                                sing_cfg[3]
            for i in range(layers):
                n_param_op = []
                for op_idx in range(num_op):
                    if i == 0:
                        params = get_model_parameters_number(
                            OPS[PRIMITIVES[op_idx]](inc, outc, stride))
                    elif i == layers - 1:
                        params = get_model_parameters_number(
                            OPS[PRIMITIVES[op_idx]](hidden_outc, outc, 1))
                    else:
                        params = get_model_parameters_number(
                            OPS[PRIMITIVES[op_idx]](
                                hidden_outc, hidden_outc, 1))
                    n_param_op.append(params)
                n_param_layer.append(n_param_op.copy())
            n_param_table.append(n_param_layer.copy())
            inc = outc
            block_layer_index += layers
        for block_idx, sing_cfg in enumerate(block_cfgs):
            hidden_outc, outc, stride, layers = sing_cfg[0], sing_cfg[1], sing_cfg[2], \
                                                sing_cfg[3]
            for i in range(layers):
                print(n_param_table[block_idx][i])
        # print('')
        # print(super_model_l)
        # exit(0)
        # -------- Other Params --------
        last_block = OPS['MB6_3x3_se0.25'](192, 320, 1)
        stem = nn.Sequential(OPS['Conv3x3_BN_swish'](3, 32, 2),
                             OPS['MB1_3x3_se0.25'](32, 16, 1))
        stern = OPS['Conv1x1_BN_swish'](320, 1280, 1)
        avgpool = nn.AvgPool2d(7)
        linear = nn.Linear(1280, 1000)
        other_param = get_model_parameters_number(stem) + \
                      get_model_parameters_number(stern) + \
                      get_model_parameters_number(avgpool) + \
                      get_model_parameters_number(linear) + \
                      get_model_parameters_number(last_block)

        stage4_max_param = target_constrain - other_param
        stage3_max_param = stage4_max_param - 284976 * 4 + 358960
        stage2_max_param = stage3_max_param - 73368 * 3 + 55412
        stage1_max_param = stage2_max_param - 51540 * 3 + 18650
        stage0_max_param = stage1_max_param - 13770 * 3 + 6566

    else:
        # Flops table is pre-calculated...
        flops_table = [[[32364, 37181, 44407, 16257, 18666, 22279],
                        [27699, 34924, 45762, 13925, 17537, 22956],
                        [27699, 34924, 45762, 13925, 17537, 22956]],
                       [[28215, 30623, 34236, 14139, 15343, 17149],
                        [17567, 20577, 25093, 8815, 10320, 12578],
                        [17567, 20577, 25093, 8815, 10320, 12578],
                        [17567, 20577, 25093, 8815, 10320, 12578]],
                       [[12220, 12972, 14101, 6126, 6502, 7066],
                        [16327, 17832, 20090, 8179, 8932, 10061],
                        [16327, 17832, 20090, 8179, 8932, 10061],
                        [16327, 17832, 20090, 8179, 8932, 10061]],
                       [[17838, 19344, 21602, 8938, 9691, 10820],
                        [23210, 25016, 27725, 11624, 12527, 13882],
                        [23210, 25016, 27725, 11624, 12527, 13882],
                        [23210, 25016, 27725, 11624, 12527, 13882]],
                       [[15934, 16386, 17063, 7975, 8201, 8539],
                        [22540, 23444, 24798, 11280, 11731, 12409],
                        [22540, 23444, 24798, 11280, 11731, 12409],
                        [22540, 23444, 24798, 11280, 11731, 12409],
                        [22540, 23444, 24798, 11280, 11731, 12409]],
                       [[29778, 30682, 32036, 14905, 15356, 16034]]]
        n_param_table = flops_table
        other_param = 22881 + 20196 + 63 + 1280 + 29778
        stage4_max_param = target_constrain - other_param
        stage3_max_param = stage4_max_param - 11280 * 4 - 7975
        stage2_max_param = stage3_max_param - 11624 * 3 - 8938
        stage1_max_param = stage2_max_param - 8179 * 3 - 6126
        stage0_max_param = stage1_max_param - 8815 * 3 - 14139

    # # ----- convert to short int to reduce search cost -----
    # short_param_tabel = []
    # for idx in range(len(n_param_table)):
    #     temp_tabel = []
    #     for j in range(len(n_param_table[idx])):
    #         temp_tabel.append(np.array(n_param_table[idx][j], dtype=np.uint32))
    #     short_param_tabel.append(temp_tabel.copy())
    #
    # n_param_table = short_param_tabel

    # ================== 2. Sort sub-models in different yaml files ================
    super_model_l = []
    super_loss_l = []
    super_param_l = []
    mse_all = []
    # for stage in range(len(potential_yaml)):
    #     for spblock_idx in range(len(potential_yaml[stage])):
    #         with open(potential_yaml[stage][spblock_idx], 'r') as f:
    #             potential = yaml.safe_load(f)
    #         mse_all.extend(potential['loss'].copy())
    # mse_all = torch.Tensor(mse_all)
    # mse_all_norm = ((mse_all - mse_all.mean()) / mse_all.std() + 2.) / 4.
    for stage in range(len(potential_yaml)):
        mse_l = []
        top_idx_l = []
        num_layers = []
        model_pool = []
        model_num = []
        for spblock_idx in range(len(potential_yaml[stage])):
            with open(potential_yaml[stage][spblock_idx], 'r') as f:
                potential = yaml.safe_load(f)
            mse_block = potential['cl_loss']
            mse_l.extend(mse_block.copy())

            idx = list(range(num_op))
            num_layers_blk = layer_cfgs[spblock_idx][stage]
            model_pool_blk = list(itertools.product(idx, repeat=num_layers_blk))
            for i, model in enumerate(model_pool_blk):
                model_pool_blk[i] = list(model_pool_blk[i])
            model_num.append(len(mse_block))
            num_layers.append(num_layers_blk)
            model_pool.append(model_pool_blk)
        # print(model_num)
        top_idx_l = np.argsort(np.array(mse_l))
        mse_l = torch.Tensor(mse_l)
        print(mse_l.dtype)
        n_models = mse_l.size(0)
        if norm_mse:
            mse_l = ((mse_l - mse_l.mean()) / mse_l.std() + 2.) / 4.
        # else:
        # mse_l = mse_all_norm()
        mse_l = mse_l.half()

        sorted_model = []
        model_blockidx = []
        sorted_loss = []
        for mdl_idx in range(n_models):
            if top_idx_l[mdl_idx] < model_num[0]:
                block_idx = 0
                top_idx = top_idx_l[mdl_idx]
            elif top_idx_l[mdl_idx] < model_num[0] + model_num[1]:
                block_idx = 1
                top_idx = (top_idx_l[mdl_idx] - model_num[0])
            else:
                block_idx = 2
                top_idx = (top_idx_l[mdl_idx] - model_num[0] - model_num[1])
            print(block_idx, top_idx)
            print(model_num)
            print(len(model_pool[0]))
            sorted_model.append(model_pool[block_idx][top_idx].copy())
            model_blockidx.append(block_idx)
            sorted_loss.append(mse_l[top_idx_l[mdl_idx]].item())
            # sorted_loss.append(mse_l[potential['top_index'][i]].item())
        sorted_n_params = []
        for mdl_idx in range(n_models):
            mdl_params = 0
            block_idx = model_blockidx[mdl_idx]
            for lyr_idx in range(num_layers[block_idx]):
                try:
                    mdl_params += n_param_table[stage][lyr_idx][sorted_model[mdl_idx][lyr_idx]]
                except:
                    print(stage, lyr_idx, mdl_idx, block_idx)
                    exit(1)
            sorted_n_params.append(mdl_params)

        super_model_l.append(sorted_model.copy())
        super_loss_l.append(sorted_loss.copy())
        super_param_l.append(sorted_n_params.copy())

    for i in range(len(potential_yaml)):
        print(super_model_l[i][0])

    # ================= 3. Traversal Search models =================
    min_loss = 100
    best_model = None
    best_param = None
    tic = time.time()
    # Note: Adjust idx upper bound (max_layer0-4) if searching takes too long. 
    # eg: max_layer3 = 1000
    max_layer0 = len(super_model_l[0])
    max_layer1 = len(super_model_l[1])
    max_layer2 = len(super_model_l[2])
    max_layer3 = len(super_model_l[3])
    max_layer4 = len(super_model_l[4])
    for layer0 in range(max_layer0):
        params0 = super_param_l[0][layer0]
        if params0 > stage0_max_param:
            continue
        loss0 = super_loss_l[0][layer0]

        for layer1 in range(max_layer1):
            if layer1 >= max_layer1:
                if max_layer1 == 0:
                    max_layer0 = layer0
                    max_layer1 = len(super_model_l[1])
                break
            params1 = params0 + super_param_l[1][layer1]
            if params1 > stage1_max_param:
                continue
            if loss_join == 'sum':
                loss1 = loss0 + super_loss_l[1][layer1]
            else:  # loss_join == 'mul':
                loss1 = loss0 * super_loss_l[1][layer1]
            toc = time.time()
            cost = toc - tic
            tic = toc
            print(
                '{:>.4}s: {}/{}:{}/{}'.format(cost, layer0, len(super_model_l[0]), layer1,
                                              len(super_model_l[1])))

            for layer2 in range(max_layer2):
                # print('{}/{}'.format(layer2, len(super_model_l[2])))
                if layer2 >= max_layer2:
                    if max_layer2 == 0:
                        max_layer1 = layer1
                        max_layer2 = len(super_model_l[2])
                    break
                params2 = params1 + super_param_l[2][layer2]
                if params2 > stage2_max_param:
                    continue
                if loss_join == 'sum':
                    loss2 = loss1 + super_loss_l[2][layer2]
                else:  # loss_join == 'mul':
                    loss2 = loss1 * super_loss_l[2][layer2]

                for layer3 in range(max_layer3):
                    if layer3 >= max_layer3:
                        if max_layer3 == 0:
                            max_layer2 = layer2
                            max_layer3 = len(super_model_l[3])
                        break
                    params3 = params2 + super_param_l[3][layer3]
                    if params3 > stage3_max_param:
                        continue
                    if loss_join == 'sum':
                        loss3 = loss2 + super_loss_l[3][layer3]
                    else:  # loss_join == 'mul':
                        loss3 = loss2 * super_loss_l[3][layer3]

                    for layer4 in range(max_layer4):
                        if layer4 >= max_layer4:
                            break

                        params4 = params3 + super_param_l[4][layer4]
                        if params4 < stage4_max_param:
                            if loss_join == 'sum':
                                loss4 = loss3 + super_loss_l[4][layer4]
                            else:  # loss_join == 'mul'
                                loss4 = loss3 * super_loss_l[4][layer4]
                            if loss4 < min_loss:
                                best_param = params4
                                min_loss = loss4
                                best_model = [layer0, layer1, layer2, layer3, layer4]

                            max_layer4 = layer4
                            if max_layer4 == 0:
                                max_layer3 = layer3
                                max_layer4 = len(super_model_l[4])
                            break

    if best_model is not None:
        print(best_model)
        for i in range(len(potential_yaml)):
            print('\t', super_model_l[i][best_model[i]])
        print(min_loss)
        print(best_param + other_param)
    else:
        print('No suitable model !')


if __name__ == '__main__':
    main()
