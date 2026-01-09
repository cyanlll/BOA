from easydict import EasyDict as EDict

from Models.boa.model import BOA_Net

def get_models(cfg):
    model_config = EDict(
        text_input_size=cfg['text_feat_dim'],
        visual_input_size=cfg['visual_feat_dim'],
        query_input_size=cfg['q_feat_size'],
        hidden_size=cfg['hidden_size'],
        max_ctx_l=cfg['max_ctx_l'],
        map_size=cfg['map_size'],
        max_desc_l=cfg['max_desc_l'],
        input_drop=cfg['input_drop'],
        drop=cfg['drop'],
        n_heads=cfg['n_heads'],
        initializer_range=cfg['initializer_range'],
        margin=cfg['margin'],
        use_hard_negative=False,
        hard_pool_size=cfg['hard_pool_size'],
        sft_factor=cfg['sft_factor'],
        top_k=cfg['top_k'])

    model = BOA_Net(model_config)
    return model
