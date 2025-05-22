import os
import ipdb
from torch.utils.data import DataLoader

from Utils.basic_utils import BigFile, read_dict
from Datasets.data_provider import Dataset4PRVR, VisDataSet4PRVR, TxtDataSet4PRVR, \
                    collate_train, collate_frame_val, collate_text_val, read_video_ids

def get_datasets(cfg, keywords_dict):

    rootpath = cfg['data_root']
    collection = cfg['collection']
    collection_val = cfg['collection_val']
    use_clip_feature = cfg['use_clip_feat']

    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection_val

    cap_file = {
        'train': '%s.caption.txt' % trainCollection,
        'val': '%s.caption.txt' % valCollection,
    }

    if use_clip_feature:
        text_feat_train_path = os.path.join(rootpath, collection, 'TextData',
                                            'clip_ViT_B_32_%s_train_query_feat.hdf5' % collection)
        text_feat_val_path = os.path.join(rootpath, collection, 'TextData',
                                          'clip_ViT_B_32_%s_val_query_feat.hdf5' % collection_val)
    else:
        text_feat_train_path = os.path.join(rootpath, collection, 'TextData',
                                            'roberta_%s_query_feat.hdf5' % collection)
        text_feat_val_path = os.path.join(rootpath, collection, 'TextData',
                                          'roberta_%s_query_feat.hdf5' % collection_val)

    caption_files = {}
    caption_files['train'] = os.path.join(rootpath, collection, 'TextData', cap_file['train'])
    caption_files['val'] = os.path.join(rootpath, collection_val, 'TextData', cap_file['val'])

    # Load visual features
    visual_feats_train = visual_feats_val = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
    visual_feats = BigFile(visual_feats_train)
    cfg['visual_feat_dim'] = visual_feats.ndims

    video2frames = read_dict(os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))
    if use_clip_feature:
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats_train, text_feat_train_path, keywords_dict, cfg)
    else:
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_train_path, keywords_dict, cfg,
                                     video2frames=video2frames)

    val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_val_path, keywords_dict, cfg)
    val_video_ids_list = read_video_ids(caption_files['val'])
    if use_clip_feature:
        val_video_dataset = VisDataSet4PRVR(visual_feats_val, video2frames, cfg, video_ids=val_video_ids_list)
    else:
        val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['batchsize'],
                              shuffle=True,
                              pin_memory=cfg['pin_memory'],
                              num_workers=cfg['num_workers'],
                              collate_fn=collate_train)
    context_dataloader = DataLoader(val_video_dataset,
                                    collate_fn=collate_frame_val,
                                    batch_size=cfg['eval_context_bsz'],
                                    num_workers=cfg['num_workers'],
                                    shuffle=False,
                                    pin_memory=cfg['pin_memory'])
    query_eval_loader = DataLoader(val_text_dataset,
                                   collate_fn=collate_text_val,
                                   batch_size=cfg['eval_query_bsz'],
                                   num_workers=cfg['num_workers'],
                                   shuffle=False,
                                   pin_memory=cfg['pin_memory'])

    return cfg, train_loader, context_dataloader, query_eval_loader
