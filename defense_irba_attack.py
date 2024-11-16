import os
import pdb
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import ShapeNetDataLoader
from data_utils.ModelNetDataLoader import BDModelNetDataLoader
from data_utils.ShapeNetDataLoader import BDShapeNetDataLoader

# defense
from build_clip_model import init_clip_model
from weights.best_param import best_prompt_weight

## compare defense [b,3,k]
from defense import SRSDefense, SORDefense, DUPNet

from model.cvae_model_old import CVAE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--dataset', type=str, default='modelnet10', help='choose data set [modelnet40, shapenet]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40, 16],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--num_anchor', type=int, default=16, help='Num of anchor point' ) 
    parser.add_argument('--R_alpha', type=float, default=5, help='Maximum rotation range of local transformation')
    parser.add_argument('--S_size', type=float, default=5, help='Maximum scailing range of local transformation')
    parser.add_argument('--alltoall', action='store_true', default=False, help='alltoall attack')

    parser.add_argument('--poisoned_rate', type=float, default=0.1, help='poison rate')
    parser.add_argument('--target_label', type=int, default=8, help='the attacker-specified target label')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    
    parser.add_argument('--checkpoint_path', type=str, default=None, help='load 3D DNN checkpoint such as: pointnet_cls')
    parser.add_argument('--recon_model_path', type=str, default=None, help='load reconstruction model path')
    parser.add_argument('--z_dim', type=int, default=1024, help=' modelnet40: 1024, shapenet: 512')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def vis_pc(pc1, pc2, path):

 
    y1 = pc1[:,0]
    x1 = pc1[:,1]
    z1 = pc1[:,2]

    y2 = pc2[:,0]
    x2 = pc2[:,1]
    z2 = pc2[:,2]

    # 创建 1 行 2 列的子图
    fig = plt.figure(figsize=(12, 6))  # 调整图形大小

    # 第一个子图
    ax1 = fig.add_subplot(121, projection='3d')  # 1 行 2 列的第一个子图
    ax1.scatter(x1, y1, z1, color='blue', label='Point Cloud 1')
    ax1.set_title('Point Cloud 1')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')

    # 第二个子图
    ax2 = fig.add_subplot(122, projection='3d')  # 1 行 2 列的第二个子图
    ax2.scatter(x2, y2, z2, color='red', label='Point Cloud 2')
    ax2.set_title('Point Cloud 2')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')

    # 显示图形
    plt.tight_layout()  # 自动调整子图间距
    # plt.show()
    plt.savefig(path)


def test(model, loader, num_class=40, device=None):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    # pdb.set_trace()
    # print("test loader size==>>",len(loader))
    cnt = 0.
    total_size = 0
    cnt2 = 0.
    clipcnt = 0.
    # if clip_model is not None:
    # defense_func = SRSDefense(drop_num=256)
    # defense_func = SORDefense(k=8)
    # defense_func = DUPNet()
    bs = 32
    pred_ori_ba_mask = 0
    pred_label_list = []
    ori_label_list = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        
        points = points.float()
        if not args.use_cpu:
            points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)
        ### defense function
        # points = defense_func(points)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]


        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))


    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]

    class_acc = np.nanmean(class_acc[:, 2])
    instance_acc = np.nanmean(mean_correct)

    return instance_acc, class_acc


def test_clip(model, loader, num_class=40, device=None, clip_model=None, recon=None):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    cnt = 0.
    total_size = 0
    cnt2 = 0.
    clipcnt = 0.
    pred_clip_res = 0.

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        
        points = points.float()
        if not args.use_cpu:
            points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        total_size += pred_choice.size()[0]

        if clip_model is not None:
            points = points.transpose(2,1)
            pred_clip = clip_model(points, target)
            pred_choice = pred_clip.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    if cnt > 0:
        pred_clip_res = float(cnt/total_size)

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]

    class_acc = np.nanmean(class_acc[:, 2])
    instance_acc = np.nanmean(mean_correct)

    return instance_acc, class_acc, pred_clip_res


def test_our(model, loader, num_class=40, device=None, clip_model=None, recon=None, ba_label=None, origin_pc=None):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    # print("test loader size==>>",len(loader))
    cnt = 0.
    total_size = 0
    cnt2 = 0.
    clipcnt = 0.

    DACC = 0.
    pred_clip_gt = 0.

    pred_label_list = []
    ori_label_list = []

    # if clip_model is not None:
    # defense_func = SRSDefense(drop_num=256)
    # defense_func = SORDefense(k=8)
    # defense_func = DUPNet()

    bs = 32
    pred_ori_ba_mask = 0
    if ba_label is not None:
        ba_label = torch.Tensor(ba_label).long().to(device)
        origin_pc = torch.Tensor(np.array(origin_pc)).to(device)
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # pdb.set_trace()
        points = points.float()
        if not args.use_cpu:
            points, target = points.to(device), target.to(device)

        # if 

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        ###
        # pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        total_size += pred_choice.size()[0]
        # print(pred_choice)
        # defense
        
        if ba_label is not None:
            pass

        if clip_model is not None:
            # pdb.set_trace()
            points = points.transpose(2,1)
            pred_clip = clip_model(points, target)
            ntargets = pred_clip.data.max(1)[1]
            points = points.transpose(2,1)
            # print("ntargets==>>",ntargets)
            clipcnt += (ntargets == target).sum()
            # print("points==>>",points.shape)
            _, _, _, npoints = recon(points, ntargets)
            pred_rec, _ = classifier(npoints)

            pred_choice_rec = pred_rec.data.max(1)[1]
            # print("rec===>>",pred_choice_rec)
            pred_choice = pred_choice_rec

            if ba_label is not None:
                cur_size = pred.shape[0]
                if cur_size < bs:
                    cur_ba_label = ba_label[(j)*bs:]
                    cur_ori_pc = origin_pc[(j)*bs:]
                else:
                    cur_ba_label = ba_label[j*bs:(j+1)*bs]
                    cur_ori_pc = origin_pc[j*bs:(j+1)*bs]
                # print("pred origin==>>",pred_choice)
                ori_ba_mask = (pred_choice == cur_ba_label)
                pred_ori_ba_mask += ori_ba_mask.sum().item()

        ### noise defense
        # if clip_model is not None:
        #     defense_points = defense_func(points)
        #     pred, _ = classifier(defense_points)
        #     print("defense_points==>>",defense_points.shape)
        #     pred_choice = pred.data.max(1)[1]
        #     print("srs defense pred_choice===>>",pred_choice)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    if pred_ori_ba_mask > 0:
        ## prediction ground truth label
        DACC = float(pred_ori_ba_mask/total_size)

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]

    class_acc = np.nanmean(class_acc[:, 2])
    instance_acc = np.nanmean(mean_correct)

    return instance_acc, class_acc, DACC



def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    def log_string(str):
        logger.info(str)
        print(str)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.dataset + '_' + args.model)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(str(args.R_alpha) + '_' + str(args.S_size) + '_'  + str(args.num_anchor) + '_' + str(args.poisoned_rate))
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    # pdb.set_trace()
    '''DATA LOADING'''
    log_string('Load dataset ...')
    num_class = args.num_category
    if 'modelnet' in args.dataset:
        assert (num_class == 10 or num_class == 40)
        data_path = '/opt/data/private/Attack/IRBA/data/modelnet40_normal_resampled/'
        train_dataset = BDModelNetDataLoader(root=data_path, args=args, split='train')
        test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
        test_bd_dataset = BDModelNetDataLoader(root=data_path, args=args, split='test')
    elif args.dataset == 'shapenet':
        assert (num_class == 16)
        data_path = '/opt/data/private/Attack/IRBA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        train_dataset = BDShapeNetDataLoader(root=data_path, args=args, split='train')
        test_dataset = ShapeNetDataLoader(root=data_path, args=args, split='test')
        test_bd_dataset = BDShapeNetDataLoader(root=data_path, args=args, split='test')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    testbdDataLoader = torch.utils.data.DataLoader(test_bd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    checkpoint_path = args.checkpoint_path

    print("load checkpoint path===>>",checkpoint_path)
    if os.path.exists(checkpoint_path):
        classifier.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        # classifier.load_state_dict(torch.load(checkpoint_path))

    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    
    ## defense model
    clip_text, _ = init_clip_model(class_name=train_dataset.class_name)
    # clip_text.to(classifier.device)

    ## 干净的 rec training, inference noise 带上一点噪声, 这个是最好的, inference 带上 noise
    ## rec noise training, clean inference

    z_dim = args.z_dim
    recon_model = CVAE(3, z_dim)    
    recon_model_path = args.recon_model_path

    recon_model.load_state_dict(torch.load(recon_model_path)['model_state_dict'])
    # # recon_model.to(classifier.device)
    recon_model.eval()
    ###
    
    if not args.use_cpu:
        classifier = classifier.to(device)
        criterion = criterion.to(device)

        recon_model = recon_model.to(device)
        clip_text = clip_text.to(device)
        
    start_epoch = 0
    global_epoch = 0
    global_step = 0

    # pdb.set_trace()
    '''DEFNESE IRBA'''
    logger.info('Start Defenese...')

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, device=device)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        if test_bd_dataset.origin_adv_label is not None:
            origin_ba_label = [x.item() for x in test_bd_dataset.origin_adv_label]
            origin_pc = test_bd_dataset.origin_data
        else:
            origin_ba_label = None
        ## backdoor attack results
        # instance_bd_acc, class_bd_acc = test(classifier.eval(), testbdDataLoader, num_class=num_class, device=device)
        # log_string('Backdoor Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_bd_acc, class_bd_acc))

        ## defense results defense_class_acc
        instance_bd_acc, class_bd_acc, defense_class_acc = test_our(classifier.eval(), testbdDataLoader, num_class=num_class, device=device, clip_model=clip_text, recon=recon_model, ba_label=origin_ba_label, origin_pc=origin_pc)
        log_string('Backdoor Defense Test Instance Accuracy: %f, Class Accuracy: %f, Defense Acc: %f' % (instance_bd_acc, class_bd_acc, defense_class_acc))

        ## test clip acc 
        # instance_acc, class_acc, clip_acc = test_clip(classifier.eval(), testDataLoader, num_class=num_class, device=device, clip_model=clip_text, recon=recon_model)


    logger.info('End of training...')
    print("logger save path ==>> ",log_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)


