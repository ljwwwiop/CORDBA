from __future__ import print_function
import argparse
import os
import csv
import random
import numpy as np
import torch.utils.data
from tqdm import tqdm
import yaml
import sys
import importlib
import numpy as np
import torch.nn as nn
# from utils.corruption import corrupt
from dataset_pcba.ModelNetDataLoader10 import ModelNetDataLoader10
from dataset_pcba.ModelNetDataLoader import ModelNetDataLoader
from dataset_pcba.ShapeNetDataLoader import PartNormalDataset


from build_clip_model import init_clip_model
from weights.best_param import best_prompt_weight
# from model.cvae_model import CVAE
from model.cvae_model_old import CVAE

## defense
from defense import SRSDefense, SORDefense, DUPNet
import pdb

# torch.backends.cudnn.enabled = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def load_model(opt):
    MODEL = importlib.import_module(opt.target_model)
    classifier = MODEL.get_model(
        opt.num_class,
        normal_channel=opt.normal
    )
    classifier = classifier.to(opt.device)
    classifier = nn.DataParallel(classifier)
    return classifier


def load_data(opt, split='test'):
    """Load the dataset from the given path.
    """
    # print('Start Loading Dataset...')
    if opt.dataset == 'ModelNet40':
        DATASET = ModelNetDataLoader(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False
        )
    elif opt.dataset == 'ShapeNetPart':
        DATASET = PartNormalDataset(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False
        )
    elif opt.dataset == 'ModelNet10':
        DATASET = ModelNetDataLoader10(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False            
        )
    else:
        raise NotImplementedError

    
    # print('Finish Loading Dataset...')
    return DATASET 

def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points.detach().numpy() # [B, N, C]
    target = target[:,0] # [B]

    target = target.cuda()
    return points, target

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--target_model', type=str, default='pointnet_cls', help='')
parser.add_argument('--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument('--output_dir', default='model_attacked',type=str)
parser.add_argument(
    '--src_label', type=int, help='shapnet = 4, modelnet40 = 8')

parser.add_argument(
    '--z_dim', type=int, default=1024, help='shapnet = 512, modelnet40 = 1024')
parser.add_argument(
    '--recon_model_path', type=str, default=None, help="dataset path")


opt = parser.parse_args()

f = open('config_pcba.yaml')
config = yaml.safe_load(f)
opt.batch_size = config['batch_size']
opt.device = config['device']
opt.workers = config['workers']
opt.input_point_nums = config['input_point_nums']

if opt.dataset == 'ModelNet40':
    opt.num_class = 40
    opt.data_path = config['ModelNet_path']
elif opt.dataset == 'ShapeNetPart':
    opt.num_class = 16
    opt.data_path = config['ShapeNetPart_path']
elif opt.dataset == 'ModelNet10':
    opt.num_class = 10
    opt.data_path = config['ModelNet_path']


opt.normal =False
opt.manualSeed = 2023  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
# corrupt_class = corrupt(seed=opt.manualSeed)

opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
opt.model_path = os.path.join(opt.output_dir, opt.dataset, 'checkpoints',f'{opt.target_model}.pth')
print(opt)

classifier = load_model(opt)
classifier.load_state_dict(torch.load(opt.model_path))
classifier.to(opt.device)


# recon_model = CVAE(3, 1024)
recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/modelnet40-best-rec/best_parameters.tar'

# recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/modelnet10-best-rec/best_parameters.tar'
# recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/shapenet-best-rec/best_parameters.tar'
# recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/modelnet40-uncond/best_parameters.tar'

### best shape - 512
z_dim = opt.z_dim
recon_model = CVAE(3, z_dim)
recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/rec-shape-dim-512/best_parameters.tar'  ## this is best 512
# recon_model_path = '/opt/data/private/Attack/PCBA/checkpoint/2024/shapenet-uncond/best_parameters.tar'

recon_model_path = opt.recon_model_path

recon_model.load_state_dict(torch.load(recon_model_path)['model_state_dict'])
recon_model.to(opt.device)
recon_model.eval()

# corrupt = corrupt_class(opt.corruption)
testset = load_data(opt, 'test')
testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers)

# print("testset.class_name>>",testset.class_name)
clip_text, _ = init_clip_model(class_name=testset.class_name)
clip_text.to(opt.device)

num_pc = len(testset)
print('Num samples: {}'.format(num_pc))

# Load backdoor test samples
attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))
attack_testset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_test, dtype=torch.float32),torch.tensor(attack_labels_test).unsqueeze(-1))
attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers))

total_correct = 0
total_testset = 0

test_dict = {}
test_dict['pc'] = None
test_dict['labels'] = np.array([])
test_dict['pred'] = np.array([])

## init defense model
# defense_func = SRSDefense(drop_num=128)
# defense_func = SORDefense(k=16)
# defense_func = DUPNet()

with torch.no_grad():
    for batch_id, data in tqdm(enumerate(testloader)):
        points, targets = data_preprocess(data)

        points = torch.tensor(points, dtype=torch.float32).cuda()
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()

        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

        if batch_id == 0:
            test_dict['pc'] = points.transpose(2,1).detach().cpu().numpy()
        else:            
            test_dict['pc'] = np.concatenate([test_dict['pc'], points.transpose(2,1).detach().cpu().numpy()], axis=0)
        test_dict['labels'] = np.concatenate([test_dict['labels'], targets.detach().cpu().numpy()], axis=0) 
        test_dict['pred'] = np.concatenate([test_dict['pred'], pred_choice.detach().cpu().numpy()], axis=0)
    print("clean sample accuracy {}".format(total_correct / float(total_testset)))
test_dict['ACC'] = total_correct / float(total_testset)


attack_correct_src = 0
attack_correct = 0
attack_total = 0
attack_dict = {}
attack_dict['pc'] = None
attack_dict['labels'] = np.array([])
attack_dict['pred'] = np.array([])
ori_label_numb = 0

src_label = opt.src_label # modelnet40 - 8, shapenet - 4
# print("data size==>> ",len(attack_testset))
with torch.no_grad():
    for batch_id, data in tqdm(enumerate(attack_testloader)):
        points, targets = data_preprocess(data)

        points = torch.tensor(points, dtype=torch.float32).cuda()
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(targets).cpu().sum()
        attack_correct_src += correct.item()

        ####
        points = points.transpose(2,1)
        pred_clip = clip_text(points, targets) # 

        ntargets = pred_clip.data.max(1)[1]
        points = points.transpose(2,1)
        _, _, _, npoints = recon_model(points, ntargets)
        # _, _, _, npoints = recon_model(points, targets)
        pred, _ = classifier(npoints)
        pred_choice = pred.data.max(1)[1]
        ###
        # points = defense_func(points)
        # pred = classifier(points)
        # pred_choice = pred.data.max(1)[1]
        # print("old===>>",pred_choice)
        ####
        ori_label_numb += (pred_choice == src_label).sum().item() ## shape ori label 4, target -8 , model40 ori 8, target 35

        correct = pred_choice.eq(targets).cpu().sum()
        attack_correct += correct.item()
        attack_total += points.size()[0]

        if batch_id == 0:
            attack_dict['pc'] = points.transpose(2,1).detach().cpu().numpy()
        else:
            attack_dict['pc'] = np.concatenate([attack_dict['pc'], points.transpose(2,1).detach().cpu().numpy()], axis=0)                    
        attack_dict['labels'] = np.concatenate([attack_dict['labels'], targets.detach().cpu().numpy()], axis=0) 
        attack_dict['pred'] = np.concatenate([attack_dict['pred'], pred_choice.detach().cpu().numpy()], axis=0)
    print("defense attack success rate {:0.6f}".format(attack_correct / float(attack_total)))
    print("defense sample acc {:0.6f}".format(ori_label_numb / float(attack_total)))
    print("origin attack success rate {:0.6f}".format(attack_correct_src / float(attack_total)))
    

