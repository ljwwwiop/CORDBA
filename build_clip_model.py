import torch
from clip import clip
import torch.nn as nn
import numpy as np
import os

from configs import get_cfg_default

from weights.best_param import best_prompt_weight
from weights.mv_utils_zs import Realistic_Projection

from yacs.config import CfgNode as CN
from dataset.dataset import ModelNetDataset
from PIL import Image
import torch.nn.functional as F

import pdb


device = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_LABELS = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

def compute_similarity(image_feat, text_feat):
    """
    Compute the similarity between image features and text features.
    Here we assume cosine similarity, but you can change it to other types of similarity if needed.
    """
    # Normalize the features
    image_feat_norm = F.normalize(image_feat, p=2, dim=-1)  # Normalize along the feature dimension
    text_feat_norm = F.normalize(text_feat, p=2, dim=-1)
    
    # Compute cosine similarity
    sim_scores = torch.matmul(image_feat_norm, text_feat_norm.t())  # [batch_size, num_texts]
    return sim_scores

class Textual_Encoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        # prompts = best_prompt_weight['{}_{}_test_prompts'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]
        # prompts = ALL_LABELS
        prompts = self.classnames
        # print(prompts)
        prompts = [f"a depth map photo of a {c}" for c in prompts]
        # print("prompts size==>>",len(prompts)," ",prompts)

        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)

        return text_feat


def load_clip_to_cpu(cfg, class_name):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    if len(class_name) > 20:
        model_path = './clip_models/clip_modelnet40.pth'  ## modelnet40 
    else:
        model_path = './clip_models/clip_shapenet.pth'
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        # model = torch.jit.load(model_path, map_location='cpu')
        state_dict = None
    except RuntimeError:
        ## here is really loader
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    return model


class PointCLIP(nn.Module):

    def __init__(self, cfg, class_name=None, flag=None):
        super(PointCLIP, self).__init__()
        self.cfg = cfg
        # classnames = self.dm.dataset.classnames
        self.flag = flag
        if class_name is None:
            classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        else:
            classnames = class_name

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg, class_name)
        # clip_model.cuda()
        # clip_model.train()
        clip_model.to(device)

        # for param in clip_model.text_encoder.parameters():
        #     param.requires_grad = False

        self.visual_encoder = clip_model.visual
        textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        for param in textual_encoder.parameters():
            param.requires_grad = False        
        text_feat = textual_encoder()

        self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()

    def real_proj(self, pc, imsize=224):
        # img = self.get_img(pc).cuda()
        img = self.get_img(pc).to(device)
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    
    def forward_infer(self, pc, label=None, get_all=None):
        ret_list = []
        # pdb.set_trace()
        weights = 1 * torch.rand(self.num_views) + 0.2
        with torch.no_grad():
            # print("pc==>>",pc.shape)
            # Realistic Projection
            images = self.real_proj(pc)  
            # save_render_img(images)          
            images = images.type(self.dtype) # [320,3,...]          
            self.view_weights = self.view_weights[:self.num_views]
            weights = weights.to(self.view_weights.device)

            # Image features
            image_feat = self.visual_encoder(images)

            # Assuming compute_similarity is defined to calculate sim(f_i, W_t)
            # sim_scores = compute_similarity(image_feat, self.text_feat_ori)
            # # Calculate the weights (α_i)
            # view_weights = torch.exp(sim_scores) / torch.sum(torch.exp(sim_scores), dim=1, keepdim=True)
            ###


            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
            image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
                        
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)

            # Store for zero-shot
            self.feat_store.append(image_feat)
            self.label_store.append(label)
            ret_list.append(image_feat_w)
                    
            logits = 100. * image_feat_w @ self.text_feat.t()
        if get_all is not None:
            cur_label_text_feat = self.text_feat[label]
            return logits, image_feat, cur_label_text_feat

        return logits
    
    def forward_train(self, pc):

        # Realistic Projection
        images = self.real_proj(pc)  
        # save_render_img(images)          
        images = images.type(self.dtype) 
    
        # Image features
        image_feat = self.visual_encoder(images)
        
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        
        image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
        image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
                    
        image_feat = image_feat.reshape(-1, self.num_views * self.channel)

        logit_scale = self.logit_scale.exp()
        # logits = 100. * image_feat_w @ self.text_feat.t()

        logits = logit_scale * image_feat_w @ self.text_feat.t()

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_feat @ self.text_feat.t()
        # logits_per_text = logit_scale * self.text_feat @ image_feat.t()
        # logits_per_text = logits_per_text.t()

        # print("here is train logits==>>",logits.shape)
        return logits, logits

    def forward(self, pc, label=None, get_all=None):

        if self.flag is None:
            if get_all is None:
                logits = self.forward_infer(pc, label)
                return logits
            else:
                logits, img_f, text_f = self.forward_infer(pc, label, get_all)
                return logits, img_f, text_f
        else:
            logits_train, logits_per_text = self.forward_train(pc)
            return logits_train, logits_per_text


class PointCLIP_FINETUNE(nn.Module):

    def __init__(self, cfg, class_name=None, flag=None):
        super(PointCLIP_FINETUNE, self).__init__()
        self.cfg = cfg
        # classnames = self.dm.dataset.classnames
        self.flag = flag
        if class_name is None:
            classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        else:
            classnames = class_name

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        # clip_model.cuda()
        clip_model.train()
        clip_model.to(device)

        for param in clip_model.transformer.parameters():
            param.requires_grad = False

        self.model = clip_model
        # self.visual_encoder = clip_model.visual
        
        # textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        # for param in textual_encoder.parameters():
        #     param.requires_grad = False        
        # text_feat = textual_encoder()

        # self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()

    def real_proj(self, pc, imsize=224):
        # img = self.get_img(pc).cuda()
        img = self.get_img(pc).to(device)
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    
    def forward_infer(self, pc, label=None):
        ret_list = []
        
        return ret_list
    
    def forward_train(self, pc, text):

        # Realistic Projection
        images = self.real_proj(pc)  
        # save_render_img(images)          
        images = images.type(self.dtype)

        texts = clip.tokenize(text).to(images.device)
    
        # Image features
        # image_feat = self.visual_encoder(images)
        image_feat = self.model.encode_image(images)
        text_feat = self.model.encode_text(texts)
        text_feat = text_feat.repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)
        
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        
        # image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
        # image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
                    
        image_feat = image_feat.reshape(-1, self.num_views * self.channel)

        # logit_scale = self.logit_scale.exp()
        # logits = 100. * image_feat_w @ self.text_feat.t()
        # logits = logit_scale * image_feat_w @ self.text_feat.t()

        text_features = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_feat @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_feat.t()

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_feat @ self.text_feat.t()
        # logits_per_text = logit_scale * self.text_feat @ image_feat.t()
        # logits_per_text = logits_per_text.t()

        # print("here is train logits==>>",logits.shape)
        return logits_per_image, logits_per_text

    def forward(self, pc, text, label=None):

        if self.flag is None:
            logits = self.forward_infer(pc, label)
            return logits
        else:
            logits_train, logits_per_text = self.forward_train(pc, text)
            return logits_train, logits_per_text


def extend_cfg():
    cfg = CN()
    cfg = get_cfg_default()

    model_cfg_path = './configs/vit_b16.yaml'
    cfg.merge_from_file(model_cfg_path)

    dataset_config_file = './configs/modelnet40.yaml'
    cfg.merge_from_file(dataset_config_file)

    cfg.freeze()

    return cfg

def init_clip_model(class_name=None, flag=None):

    # model = PointCLIP()

    from pprint import pprint
    config = extend_cfg()

    # pprint(config)

    model = PointCLIP(config, class_name=class_name)

    return model,config

def init_clip_model_train(class_name=None, flag='train'):


    from pprint import pprint
    config = extend_cfg()

    # pprint(config)

    model = PointCLIP_FINETUNE(config, class_name=class_name, flag=flag)

    return model, config


def save_render_img(img, path=None):

    if path is None:
        path = '/opt/data/private/Attack/PCBA/render_im'
    for i in range(img.shape[0]):
        selected_image = img[i].detach().cpu()
        selected_image = selected_image.permute(1, 2, 0).numpy()
        selected_image = (selected_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(selected_image)
        pil_image.save('{}/proj_{}.png'.format(path, i))

if __name__ == '__main__':

    model,cfg = init_clip_model()

    pdb.set_trace()

    ## 
    attack_data_test = np.load(os.path.join('/opt/data/private/Attack/PCBA/attack', 'attack_data_train.npy'))
    attack_labels_test = np.load(os.path.join('/opt/data/private/Attack/PCBA/attack', 'attack_labels_train.npy'))

    attack_testset = ModelNetDataset(
        root=cfg.DATASET.ROOT,
        sub_sampling=False,
        npoints=2048,
        split='train_files',
        data_augmentation=False)
    attack_testset.data = attack_data_test
    attack_testset.labels = attack_labels_test

    attack_testloader = torch.utils.data.DataLoader(
            attack_testset,
            batch_size=4,
            shuffle=True,
            num_workers=4)
    (tpoints, tlabels) = list(enumerate(attack_testloader))[0][1]
    tpoints, tlabels = tpoints.to(device), tlabels.to(device)
    ###

    pointoptset = ModelNetDataset(
    root=cfg.DATASET.ROOT,
    sub_sampling=False,
    npoints=2048,
    split='train_files',
    data_augmentation=False)

    pointoptset.data = pointoptset.data[:1000]
    pointoptset.labels = pointoptset.labels[:1000]

    # Get the subset of samples from the source class
    ind = [i for i, label in enumerate(pointoptset.labels) if label != 8]
    pointoptset.data = np.delete(pointoptset.data, ind, axis=0)
    pointoptset.labels = np.delete(pointoptset.labels, ind, axis=0)

    pointoptloader = torch.utils.data.DataLoader(
        pointoptset,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=4)

    (points, labels) = list(enumerate(pointoptloader))[0][1]
    points, labels = points.to(device), labels.to(device)

    output = model(points, labels)

    pse_label = torch.max(output,1)[1].data

    print(len(points))



# /opt/data/private/Attack/PF-Attack-main/GSDA-main/Exps/Untarget/GSDA_0_BiStep10_IterStep200_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16_cclinf0.18_Shapepart_PointNetPP_dg18
