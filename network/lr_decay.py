# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, base_lr, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.transformer.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim==1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        lr = base_lr * layer_scales[layer_id]

        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr": lr,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr": lr,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id based on the provided ViT model structure.
    """
    if name in ['cls_token', 'pos_embed', 'ts_to_patch_embedding', 'text_to_embedding',
                'ts_channel_type_embedding', 'text_type_embedding', 'ts_pos_embedding', 'text_pos_embedding']:
        # Embedding layers
        return 0
    elif name.startswith('ts_to_patch_embedding'):
        return 0
    elif name.startswith('text_to_embedding'):
        return 0
    elif name.startswith('transformer.layers'):
        # Transformer blocks (assuming they are 12 layers in total)
        return int(name.split('.')[2]) + 1
    elif name.startswith('dense_scratch') or name.startswith('dense_act_postprocess') or name.startswith('dense_head'):
        # Dense layers, treated as the last "layer" after all transformer layers
        return num_layers
    else:
        # Default to the last layer if none of the above matches
        return num_layers