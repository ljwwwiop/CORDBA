import numpy as np
import pdb
import argparse

class WLT(object):
    def __init__(self, args):
        self.num_anchor = args.num_anchor # 16
        self.sigma = 0.5
        self.R_alpha = args.R_alpha # 5.0
        self.S_size = args.S_size # 5.0
        self.seed = args.seed # 256
        
    def __call__(self, pos):
        M = self.num_anchor 
        # pdb.set_trace()
        # Multi-anchor transformation
        idx = self.fps(pos, M)
        pos_anchor = pos[idx]
        pos_repeat = np.expand_dims(pos,0).repeat(M, axis=0) # [16, 1024, 3]
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype) # 
        pos_normalize = pos_repeat - pos_anchor.reshape(M,-1,3) # shift
        pos_transformed = self.multi_anchor_transformation(pos_normalize) # transfer
        
        # Smooth Aggregation
        pos_transformed = pos_transformed + pos_anchor.reshape(M,-1,3)   # de-norm    
        pos_new = self.smooth_aggregation(pos, pos_anchor, pos_transformed)
        return pos.astype('float32'), pos_new.astype('float32')
        
    def fps(self, pos, npoint):
        np.random.seed(self.seed)
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_)
        distance = np.ones(N, dtype=np.float64) * 1e10
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    
    def multi_anchor_transformation(self, pos_normalize):
        M, _, _ = pos_normalize.shape

        degree = np.pi * np.ones((M, 3)) * self.R_alpha / 180.0
        scale = np.ones((M, 3)) * self.S_size

        # Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3)

        # Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)
        
        pos_normalize = pos_normalize @ R @ S
        return pos_normalize
    
    def smooth_aggregation(self, pos, pos_anchor, pos_transformed):
        M, N, _ = pos_transformed.shape
        
        # Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor,1).repeat(N, axis=1) - np.expand_dims(pos,0).repeat(M, axis=0)
        projection = np.expand_dims(np.eye(3), 0)
        sub = sub @ projection
        sub = np.sqrt(((sub) ** 2).sum(2))

        # Kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T)
        return pos_new



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vis_pc(pointcloud, path=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    ax.scatter(x, y, z, c='b', marker='o', s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Point Cloud Visualization')

    plt.savefig(path)


def create_bounding_box(point_cloud):
    """
    创建一个包围点云的边框
    :param point_cloud: 点云数据 [N, 3]
    :return: 边框的顶点
    """
    # 计算点云的边界
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    
    # 创建包围盒的8个顶点
    bounding_box = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                              [min_coords[0], min_coords[1], max_coords[2]],
                              [min_coords[0], max_coords[1], min_coords[2]],
                              [min_coords[0], max_coords[1], max_coords[2]],
                              [max_coords[0], min_coords[1], min_coords[2]],
                              [max_coords[0], min_coords[1], max_coords[2]],
                              [max_coords[0], max_coords[1], min_coords[2]],
                              [max_coords[0], max_coords[1], max_coords[2]]])
    return bounding_box


def plot_point_cloud_and_box(point_cloud, bounding_box):
    """
    可视化点云和边框
    :param point_cloud: 点云数据 [N, 3]
    :param bounding_box: 边框的顶点
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='b', label='Point Cloud')
    
    # 绘制包围盒
    # 连接包围盒的顶点以形成边框
    edges = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7]
    ]
    
    for edge in edges:
        ax.plot3D(*zip(bounding_box[edge[0]], bounding_box[edge[1]]), color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    # plt.show()

    plt.savefig('/opt/data/private/Attack/IRBA/new_bbox.png')

if __name__ == "__main__":

    args = parse_args()
    #  python backdoor_attack.py --dataset modelnet10 --num_category 10 --model pointnet_cls --p
    # oisoned_rate 0.1 --target_label 8 --num_anchor 16 --R_alpha 5 --S_size 5 --process_data --use_un
    # iform_sample --gpu 2 

    # temp = FGBA(args)
    # x = np.ones((1024,3))

    # _, output = temp(x)

    output = np.load('/opt/data/private/Attack/IRBA/new_pc.npy')
    print("output==>>",output.shape)

    bounding_box = create_bounding_box(output)
    plot_point_cloud_and_box(output, bounding_box)

    # vis_pc(output, path='/opt/data/private/Attack/IRBA/new_pc.png')
    