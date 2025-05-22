import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.boa.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, PT, BertCrossAttention


class BOA_Net(nn.Module):
    def __init__(self, config):
        super(BOA_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)

        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.query_encoder2 = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = PT(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=config.map_size,
                                     sft_factor=config.sft_factor))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder_1 = PT(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=config.max_ctx_l,
                                        sft_factor=config.sft_factor))

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.cac = BertCrossAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=config.map_size,
                                            sft_factor=config.sft_factor))
        self.layer1c = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropoutc = nn.Dropout(config.drop)
        self.layer2c = nn.Linear(config.hidden_size, config.map_size)
        self.caf = BertCrossAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=config.max_ctx_l,
                                            sft_factor=config.sft_factor))
        self.layer1f = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropoutf = nn.Dropout(config.drop)
        self.layer2f = nn.Linear(config.hidden_size, config.max_ctx_l)

        self.top_k = config.top_k
        self.sft_factor = config.sft_factor

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):

            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size


    def forward(self, batch, epoch, sval):

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']
        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']
        device = clip_video_feat.device
        sc_masks_t = batch['sc_masks_t']
        sc_masks_v = torch.tensor(batch['sc_masks_v']).to(device)
        deviation_feat_c = (sval.deviation_feat_c / sval.total_feature_num)
        deviation_feat_f = (sval.deviation_feat_f / sval.total_feature_num)
        sc_feat_c_ = (sval.sc_feat_c / sval.sc_feat_n.unsqueeze(-1).
                      repeat(1, deviation_feat_c.shape[0]) - deviation_feat_c.unsqueeze(0).
                      repeat(sval.cluster_number, 1)).to(device)
        sc_feat_f_ = (sval.sc_feat_f / sval.sc_feat_n.unsqueeze(-1).
                      repeat(1, deviation_feat_f.shape[0]) - deviation_feat_f.unsqueeze(0).
                      repeat(sval.cluster_number, 1)).to(device)

        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask, (sc_feat_c_, sc_feat_f_))

        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_ , video_query\
            = self.get_pred_from_raw_query(query_feat, query_mask, query_labels, vid_proposal_feat,
                                           encoded_frame_feat, (sc_feat_c_, sc_feat_f_),
                                           sval, (sc_masks_t, sc_masks_v), return_query_feats=True)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query]


    def encode_query(self, query_feat, query_mask, new_feat, sc_mask):

        query_feat = self.query_input_proj(query_feat)
        query_feat = self.query_pos_embed(query_feat)
        d = query_feat.shape[1]
        query_feat_c = query_feat.clone().to(query_feat.device)
        query_feat_f = query_feat.clone().to(query_feat.device)

        for i in range(query_feat.size(0)):
            slice_mask = query_mask[i].sum()
            sc_feat_c = new_feat[0][sc_mask[i]]
            sc_feat_f = new_feat[1][sc_mask[i]]
            if int(slice_mask) + sc_feat_c.shape[0] >= d:
                sc_feat_c = sc_feat_c[:sc_feat_c.shape[0] - (int(slice_mask) + sc_feat_c.shape[0] - d)]
                sc_feat_f = sc_feat_f[:sc_feat_f.shape[0] - (int(slice_mask) + sc_feat_f.shape[0] - d)]
            query_feat_c[i][int(slice_mask):int(slice_mask) + sc_feat_c.shape[0]] = sc_feat_c
            query_feat_f[i][int(slice_mask):int(slice_mask) + sc_feat_f.shape[0]] = sc_feat_f
            query_mask[i][:int(slice_mask) + sc_feat_c.shape[0]] = 1

        encoded_query_c = self.query_encoder(query_feat_c, query_mask.unsqueeze(1))
        encoded_query_f = self.query_encoder2(query_feat_f, query_mask.unsqueeze(1))
        
        video_query_c = self.get_modularized_queries(encoded_query_c, query_mask)  # (N, D) * 1
        video_query_f = self.get_modularized_queries(encoded_query_f, query_mask)  # (N, D) * 1

        return (video_query_c, video_query_f)

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None, sc_feats=None):

        clip_video_feat = self.clip_input_proj(clip_video_feat)
        clip_video_feat = self.clip_pos_embed(clip_video_feat)

        frame_video_feat = self.frame_input_proj(frame_video_feat)
        frame_video_feat = self.frame_pos_embed(frame_video_feat)

        encoded_clip_feat = self.clip_encoder(clip_video_feat, None)
        encoded_frame_feat = self.frame_encoder_1(frame_video_feat, video_mask.unsqueeze(1))

        clip_key_features = torch.mean(torch.mean(encoded_clip_feat, dim=-1), dim=1)
        frame_key_features = torch.mean(torch.mean(encoded_frame_feat, dim=-1), dim=1)
        video_feat = self.feat_retrieval(clip_key_features, frame_key_features, sc_feats)

        weight_token_c = video_feat[0].unsqueeze(1)
        weight_c = []
        for i in range(encoded_clip_feat.shape[-1]):
            temp_token = self.cac(weight_token_c, encoded_clip_feat[..., i], None)
            weight_c.append(temp_token)
        weight_c = self.layer2c(self.dropoutc(F.relu(self.layer1c(torch.cat(weight_c, dim=1)))))
        weight_c = F.softmax(weight_c.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out_c = torch.sum(encoded_clip_feat * weight_c.unsqueeze(2).
                          repeat(1, 1, encoded_clip_feat.shape[2], 1), dim=-1)

        weight_token_f = video_feat[1].unsqueeze(1)
        weight_f = []
        for i in range(encoded_frame_feat.shape[-1]):
            temp_token = self.caf(weight_token_f, encoded_frame_feat[..., i], video_mask.unsqueeze(1))
            weight_f.append(temp_token)
        weight_f = self.layer2f(self.dropoutf(F.relu(self.layer1f(torch.cat(weight_f, dim=1)))))
        weight_f = F.softmax(weight_f.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out_f = torch.sum(encoded_frame_feat * weight_f.unsqueeze(2).
                          repeat(1, 1, encoded_frame_feat.shape[2], 1), dim=-1)

        return out_f, out_c

    def feat_retrieval(self, clip_key_features, frame_key_features, sc_feats):
        device = clip_key_features.device
        vn = clip_key_features.shape[0]
        td = clip_key_features.shape[-1]

        sc_feat_c = F.normalize(sc_feats[0], dim=-1).to(device)
        sc_feat_f = F.normalize(sc_feats[1], dim=-1).to(device)

        clip_scores = torch.matmul(F.normalize(clip_key_features, dim=-1), sc_feat_c.t())
        frame_scores = torch.matmul(F.normalize(frame_key_features, dim=-1), sc_feat_f.t())

        a, b = torch.topk(clip_scores, k=self.top_k, dim=-1)
        c = torch.gather(sc_feat_c.unsqueeze(0).repeat(vn, 1, 1), dim=1, index=b.unsqueeze(-1).repeat(1, 1, td))
        a = torch.softmax(a, dim=-1).unsqueeze(-1).repeat(1, 1, td)
        result_c = torch.sum(a * c, dim=1)

        a, b = torch.topk(frame_scores, k=self.top_k, dim=-1)
        c = torch.gather(sc_feat_f.unsqueeze(0).repeat(vn, 1, 1), dim=1, index=b.unsqueeze(-1).repeat(1, 1, td))
        a = torch.softmax(a, dim=-1).unsqueeze(-1).repeat(1, 1, td)
        result_f = torch.sum(a * c, dim=1)

        return (result_c, result_f)

    def attention(self, features, scores, mask=None):
        if mask is not None:
            modular_attention_scores = F.softmax(mask_logits(scores, mask.unsqueeze(2)), dim=1)
        else:
            modular_attention_scores = F.softmax(scores, dim=1)
        modular_features = torch.einsum("blm,bld->bmd", modular_attention_scores, features)  # (N, 2 or 1, D)
        return modular_features.squeeze()

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)  # (N, L, D_hidden)
        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs

        return query_context_scores, indices, clip_level_query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None, ori_feat=None,
                                sval=None, sc_mask=None, return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask, ori_feat, sc_mask[0])

        # get clip-level retrieval scores
        clip_scale_scores, index_c, clip_scores = self.get_clip_scale_scores(video_query[0], video_proposal_feat)
        frame_scale_scores, index_f, frame_scores = self.get_clip_scale_scores(video_query[1], encoded_frame_feat)

        if return_query_feats:
            sval.enhance((video_proposal_feat.clone().detach(),
                                          encoded_frame_feat.clone().detach()), sc_mask[1],
                                         (index_c, index_f), query_labels)

            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query[0], video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query[1], encoded_frame_feat)

            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, video_query
        else:

            return clip_scale_scores, frame_scale_scores, clip_scores, frame_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
