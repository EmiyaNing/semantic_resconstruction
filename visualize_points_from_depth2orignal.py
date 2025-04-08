






import os
import cv2
import numpy as np
import open3d as op
import json

def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (2*x/float(W)-1.0)*np.pi
    phi = (y/float(H)-0.5)*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),-np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])
# def get_3Dpoints(dep,H,W):
#     x,y = np.meshgrid(np.arange(W),np.arange(H))
#     theta = (1.0-2*x/float(W))*np.pi
#     phi = (0.5-y/float(H))*np.pi
#     ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
#     vec = np.array([(cp*ct),(cp*st),(sp)])
#     pts = vec * dep
#     return pts.transpose([1,2,0])
import tqdm
img_dir='E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1'

    # os.makedirs(os.path.join(args.data,'point_clouds'),exist_ok=True)
    # Prepare image to processed
img_names =[]
for img_name in os.listdir(img_dir):
    if img_name.endswith("rgb.png"):
        img_names.append(img_name)
    else:
        continue
    #Output resolution
    H,W = 512, 1024
    max_depth_meters=10
for name in img_names:
    img = cv2.imread(os.path.join(img_dir,name))
    img = cv2.cvtColor(cv2.resize(img,(W,H)),cv2.COLOR_BGR2RGB)
    gt_depth=cv2.imread(os.path.join(img_dir,name[:-7]+'depth.png'),-1)
    gt_depth = cv2.resize(gt_depth, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
    gt_depth = gt_depth.astype(np.float) / 512
    gt_depth[gt_depth >max_depth_meters + 1] = max_depth_meters + 1
    # mask = np.ones([512, 1024])
    # mask[0:int(512 * 0.15), :] = 0
    # mask[512 - int(512 * 0.15):512, :] = 0
    # gt_depth=gt_depth*mask
    pts = get_3Dpoints(gt_depth,H,W)
    pts=pts.reshape(-1, 3)
    #读取pose信息用于得到原始点云
    pose_info = "E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1\camera_29c4d3039cb14983b563b7697a8c333c_conferenceRoom_1_frame_equirectangular_domain_pose.json"
    with open(pose_info, 'r', encoding='utf-8') as file:
        pose_dict = json.load(file)
    camera_location = np.array(pose_dict["camera_location"])
    camera_rt_matrix = np.array(pose_dict["camera_rt_matrix"])
    camera_rt_matrix = np.vstack((camera_rt_matrix, np.array([0, 0, 0, 1])))
    camera_rt_matrix_inv = np.linalg.inv(camera_rt_matrix)
    camera_rotation = np.array(pose_dict["final_camera_rotation"])
    flip_x, flip_y, flip_z = camera_location#读取完毕pose信息
    # x_mid=np.mean(points, axis=0)[0]
    # p=points[:][0]
    # points[:,0]=-points[:,0]
    # points[:,0]+=2*x_mid
    # flip_x=2*x_mid-15.086201
    # theta_x=0.0
    # theta_y=0.0
    # theta_z=2.1609878540039062-np.pi/2
    # flip_matrix1=np.array([
    #     [1.0, 0.0, 0.0,flip_x],
    #     [0.0, 1.0, 0.0, -22.210253],
    #     [0.0, 0.0, 1.0, 1.240263],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # flip_matrix2=np.array([
    #     [1.0, 0.0, 0.0,-flip_x],
    #     [0.0, 1.0, 0.0, 22.210253],
    #     [0.0, 0.0, 1.0, -1.240263],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # rotate_x=np.array([
    #     [1, 0, 0.0, 0.0],
    #     [0.0, np.cos(theta_x), np.sin(-theta_x), 0.0],
    #     [0.0,np.sin(theta_x), np.cos(theta_x), 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # rotate_y=np.array([
    #     [np.cos(theta_y), 0,np.sin(-theta_y), 0.0],
    #     [0.0, 1.0, 0.0, 0.0],
    #     [np.sin(-theta_y),0, np.cos(theta_y), 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # rotate_z=np.array([
    #     [np.cos(theta_z), np.sin(-theta_z), 0.0, 0.0],
    #     [np.sin(theta_z), np.cos(theta_z), 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])

    # rotate_matrix=np.matmul(rotate_x,np.matmul(rotate_y,rotate_z))
    # 这部分用于预先处理点云，先按照x方向对称，然后旋转之后，再执行rotate
    theta_x_pred = np.pi / 2
    theta_y_pred = -np.pi / 2
    pred_x = np.array([
        [1, 0, 0.0, 0.0],
        [0.0, np.cos(theta_x_pred), np.sin(-theta_x_pred), 0.0],
        [0.0, np.sin(theta_x_pred), np.cos(theta_x_pred), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    pred_y = np.array([
        [np.cos(theta_y_pred), 0, np.sin(-theta_y_pred), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [np.sin(theta_y_pred), 0, np.cos(theta_y_pred), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    pred_rotate_matrix = np.matmul(pred_x, pred_y)
    # flip_matrix1 = np.array([
    #     [1.0, 0.0, 0.0, flip_x],
    #     [0.0, 1.0, 0.0, flip_y],
    #     [0.0, 0.0, 1.0, flip_z],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # flip_matrix2 = np.array([
    #     [1.0, 0.0, 0.0, -flip_x],
    #     [0.0, 1.0, 0.0, -flip_y],
    #     [0.0, 0.0, 1.0, -flip_z],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    theta_x, theta_y, theta_z = -camera_rotation
    # theta_x=0.0
    # theta_y=0.0
    # theta_z=theta_z-np.pi/2
    rotate_x = np.array([
        [1, 0, 0.0, 0.0],
        [0.0, np.cos(theta_x), np.sin(-theta_x), 0.0],
        [0.0, np.sin(theta_x), np.cos(theta_x), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    rotate_y = np.array([
        [np.cos(theta_y), 0, np.sin(-theta_y), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [np.sin(theta_y), 0, np.cos(theta_y), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    rotate_z = np.array([
        [np.cos(theta_z), np.sin(-theta_z), 0.0, 0.0],
        [np.sin(theta_z), np.cos(theta_z), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # camera_rotate_matrix=np.matmul(flip_matrix1,np.matmul(pred_rotate_matrix,flip_matrix2))
    # camera_rotate_matrix = np.matmul(flip_matrix1, np.matmul(rotate_z, np.matmul(rotate_y, np.matmul(rotate_x, (np.matmul(pred_rotate_matrix, flip_matrix2))))))
    camera_rotate_matrix = np.matmul(pred_rotate_matrix,np.matmul(rotate_x, np.matmul(rotate_y, rotate_z)))
    #得先绕着目前的中心点旋转了就是原点
    # 将点云坐标从 N*3 转换为 N*4 以应用变换矩阵
    # 转换回 N*3 格式
    point_cloud_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    # 应用变换矩阵
    transformed_point_cloud = np.dot(camera_rotate_matrix, point_cloud_homogeneous.T).T
    transformed_point_cloud=transformed_point_cloud[:, :3]
    # center_x = np.mean(transformed_point_cloud[:, 0])#不一定要了
    transformed_point_cloud[:, 0] = - transformed_point_cloud[:, 0]#左右对称了
    transformed_point_cloud[:,:]+=[flip_x,flip_y,flip_z]
    # #location 0.76981, 41.105618, 1.387447
    # theta 15.086201, -22.210253, 1.240263
    # transformed_point_cloud = transformed_point_cloud[:, :3]
    # transformed_point_cloud[:, 0] -= flip_x
    # transformed_point_cloud[:, 1] -= flip_y
    # transformed_point_cloud[:, 2] -= flip_z
    # transformed_point_cloud[:,1] =-transformed_point_cloud[:,1]
    transformed_xyz_min = np.amin(transformed_point_cloud, axis=0)
    transformed_xyz_max = np.amax(transformed_point_cloud, axis=0)
    pcd_transformed = op.geometry.PointCloud()
    pcd_transformed.points = op.utility.Vector3dVector(transformed_point_cloud)
    pcd_transformed.colors = op.utility.Vector3dVector(img.reshape(-1,3) / 255.)
    op.visualization.draw_geometries([pcd_transformed], window_name="1")
    #读取物体
    with open(
            "E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1\Annotations\\board_1.txt",
            'r') as file:
        lines = file.readlines()
    # 解析点云数据
    item = []
    for line in lines:
        # 移除行尾的换行符并分割字符串
        parts = line.strip().split()
        # 将字符串转换为浮点数
        point = [float(part) for part in parts]
        item.append(point)
    item = np.array(item)
    # data = np.load("E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1\conferenceRoom_1\conferenceRoom_1.txt")
    points_item = item[:, :3]  # 点云坐标
    colors_item = item[:, 3:]
    #对这个物体点云预处理
    indices=[]
    for item_point in points_item:
        for idx,orignal_point in zip(list(range(transformed_point_cloud.shape[0])),transformed_point_cloud):
            dist = np.linalg.norm(item_point-orignal_point)
            if dist<=0.1:
                indices.append(idx)
        if len(indices) >= 200:
            break
    img=img.reshape(-1,3)
    img[indices]=[255.0,255.0,255.0]
    # source_dtree = o3d.geometry.KDTreeFlann(source_point_cloud)
    # # 为每个目标点找到最近的源点
    # _, indices = source_dtree.search_knn_vector_3d(target_point_cloud.points, 1)
    # 根据找到的索引对原始点云进行染色
    # colors = [[1, 0, 0] for _ in range(len(indices))]
    # for i, idx in enumerate(indices):
    #     colors[i] = [0, 1, 0]
    # xyz_min = np.amin(pts, axis=0)
    # xyz_max = np.amax(pts, axis=0)
    # print(xyz_min, xyz_max)
    pcd = op.geometry.PointCloud()
    pcd.points = op.utility.Vector3dVector(transformed_point_cloud)
    pcd.colors = op.utility.Vector3dVector(img.reshape(-1,3)/255.)
    op.visualization.draw_geometries([pcd],window_name=name)