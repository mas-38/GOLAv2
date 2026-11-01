from collections import OrderedDict


def rgbt_initialization_state_dict(rgb_state):
    """Expand a full RGB checkpoint without merging its task LoRA into DINOv3.

    The receiving backbone must call add_modality('tir') before loading. The
    shared deep adapters, backbone weights, head and role encodings are retained.
    """
    if 'architecture' not in rgb_state or rgb_state['architecture'][0].item() != 1:
        raise ValueError('Expected a version-1 DINOv3RGB full checkpoint')
    if any('.adapters.tir.' in name for name in rgb_state):
        raise ValueError('The source checkpoint already contains TIR adapters')
    result = OrderedDict(rgb_state)
    for name, value in rgb_state.items():
        if '.adapters.rgb.' in name:
            result[name.replace('.adapters.rgb.', '.adapters.tir.')] = value.clone()
    embedding = 'backbone.modality_embed.rgb'
    result['backbone.modality_embed.tir'] = rgb_state[embedding].clone()
    return result
