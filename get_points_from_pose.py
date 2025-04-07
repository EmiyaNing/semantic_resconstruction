# import torch
#
# # 指定 .pth 文件路径
# file_path = 'E:\mmdetection3d\\tools\work_dirs\\resnet50_rnn__mp3d.pth'
#
# # 加载 .pth 文件内容
# content = torch.load(file_path, map_location=torch.device('cpu'))
#
# # 打印内容
# print("内容类型：", type(content))
# print("内容：", content)

import numpy as np
import open3d as op
import json
with open("E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1\conferenceRoom_1.txt", 'r') as file:
    lines = file.readlines()
# 解析点云数据
data = []
for line in lines:
    # 移除行尾的换行符并分割字符串
    parts = line.strip().split()
    # 将字符串转换为浮点数
    point = [float(part) for part in parts]
    data.append(point)
data = np.array(data)
# data = np.load("E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1\conferenceRoom_1\conferenceRoom_1.txt")
points = data[:, :3]  # 点云坐标
colors = data[:, 3:]
with open("E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1\Annotations\chair_6.txt", 'r') as file:
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
indices=[]
for i, item in enumerate(points):
    if np.any(np.all(item == points_item, axis=1)):
        indices.append(i)
colors[indices]=(100.0,200.0,100.0)
# xyz_min = np.amin(points, axis=0)
# xyz_max = np.amax(points, axis=0)
# print(xyz_min,xyz_max)
# pcd = op.geometry.PointCloud()
# pcd.points = op.utility.Vector3dVector(points)
# pcd.colors = op.utility.Vector3dVector(colors / 255.)
# op.visualization.draw_geometries([pcd],window_name="1")
# camera_rt_matrix = np.linalg.inv(np.array([
#     [-0.5060870051383972, -0.8624778985977173, -0.0028067133389413357, 35.84617233276367],
#     [0.0013655807124450803, 0.0024529320653527975, -0.9999960660934448, 1.285560965538025],
#     [0.8624813556671143, -0.5060887932777405, -6.361379928421229e-05, 20.13923454284668],
#     [0.0, 0.0, 0.0, 1.0]
# ]))
# rotate=np.array([
#     [np.cos(2.1609878540039062-np.pi/2), np.sin(-(2.1609878540039062-np.pi/2)), 0.0, 0.0],
#     [np.sin(2.1609878540039062-np.pi/25), np.cos(2.1609878540039062-np.pi/2), 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0],
#     [0.0, 0.0, 0.0, 1.0]])
#可视化一下原始点云
# pcd = op.geometry.PointCloud()
# pcd.points = op.utility.Vector3dVector(points)
# pcd.colors = op.utility.Vector3dVector(colors / 255.)
# op.visualization.draw_geometries([pcd],window_name="1")
#读取pose
pose_info = "E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_2\conferenceRoom_1\camera_29c4d3039cb14983b563b7697a8c333c_conferenceRoom_1_frame_equirectangular_domain_pose.json"
with open(pose_info, 'r', encoding='utf-8') as file:
    pose_dict = json.load(file)
camera_location=np.array(pose_dict["camera_location"])
camera_rt_matrix=np.array(pose_dict["camera_rt_matrix"])
camera_rt_matrix=np.vstack((camera_rt_matrix,np.array([0,0,0,1])))
camera_rt_matrix_inv=np.linalg.inv(camera_rt_matrix)
camera_rotation=np.array(pose_dict["final_camera_rotation"])
flip_x,flip_y,flip_z=camera_location
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
#这部分用于预先处理点云，先按照x方向对称，然后旋转之后，再执行rotate
theta_x_pred=-np.pi/2
theta_y_pred=np.pi/2
pred_x=np.array([
    [1, 0, 0.0, 0.0],
    [0.0, np.cos(theta_x_pred), np.sin(-theta_x_pred), 0.0],
    [0.0,np.sin(theta_x_pred), np.cos(theta_x_pred), 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
pred_y=np.array([
    [np.cos(theta_y_pred), 0,np.sin(-theta_y_pred), 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [np.sin(theta_y_pred),0, np.cos(theta_y_pred), 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
pred_rotate_matrix=np.matmul(pred_y,pred_x)
center_x = np.mean(points[:, 0])
points[:,0] = 2 * center_x - points[:, 0]
flip_x=2*center_x-flip_x
flip_matrix1=np.array([
    [1.0, 0.0, 0.0, flip_x],
    [0.0, 1.0, 0.0, flip_y],
    [0.0, 0.0, 1.0, flip_z],
    [0.0, 0.0, 0.0, 1.0]
])
flip_matrix2=np.array([
    [1.0, 0.0, 0.0, -flip_x],
    [0.0, 1.0, 0.0, -flip_y],
    [0.0, 0.0, 1.0, -flip_z],
    [0.0, 0.0, 0.0, 1.0]
])
theta_x,theta_y,theta_z=camera_rotation
# theta_x=0.0
# theta_y=0.0
# theta_z=theta_z-np.pi/2
rotate_x=np.array([
    [1, 0, 0.0, 0.0],
    [0.0, np.cos(theta_x), np.sin(-theta_x), 0.0],
    [0.0,np.sin(theta_x), np.cos(theta_x), 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
rotate_y=np.array([
    [np.cos(theta_y), 0,np.sin(-theta_y), 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [np.sin(theta_y),0, np.cos(theta_y), 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
rotate_z=np.array([
    [np.cos(theta_z), np.sin(-theta_z), 0.0, 0.0],
    [np.sin(theta_z), np.cos(theta_z), 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
# camera_rotate_matrix=np.matmul(flip_matrix1,np.matmul(pred_rotate_matrix,flip_matrix2))
camera_rotate_matrix=np.matmul(flip_matrix1,np.matmul(rotate_z,np.matmul(rotate_y,np.matmul(rotate_x,(np.matmul(pred_rotate_matrix,flip_matrix2))))))
# #location 0.76981, 41.105618, 1.387447
#theta 15.086201, -22.210253, 1.240263
# 将点云坐标从 N*3 转换为 N*4 以应用变换矩阵
point_cloud_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

# 应用变换矩阵
transformed_point_cloud = np.dot(camera_rotate_matrix, point_cloud_homogeneous.T).T

# 转换回 N*3 格式
transformed_point_cloud = transformed_point_cloud[:, :3]
# transformed_point_cloud[:,1] =-transformed_point_cloud[:,1]
transformed_xyz_min = np.amin(transformed_point_cloud, axis=0)
transformed_xyz_max = np.amax(transformed_point_cloud, axis=0)
pcd_transformed = op.geometry.PointCloud()
pcd_transformed.points = op.utility.Vector3dVector(transformed_point_cloud)
pcd_transformed.colors = op.utility.Vector3dVector(colors / 255.)
op.visualization.draw_geometries([pcd_transformed],window_name="1")
#可视化物体
# point_cloud_homogeneous_item = np.hstack((points_item, np.ones((points_item.shape[0], 1))))
# # 应用变换矩阵
# transformed_point_cloud_item = np.dot(camera_rotate_matrix, point_cloud_homogeneous_item.T).T
# # colors_item[:]=(100.0,200.0,100.0)
# # 转换回 N*3 格式
# transformed_point_cloud_item = transformed_point_cloud_item[:, :3]
# transformed_point_cloud[:,1] =-transformed_point_cloud[:,1]
# transformed_xyz_min_item = np.amin(transformed_point_cloud, axis=0)
# transformed_xyz_max_item = np.amax(transformed_point_cloud, axis=0)
# pcd_transformed_item = op.geometry.PointCloud()
# pcd_transformed_item.points = op.utility.Vector3dVector(transformed_point_cloud_item)
# pcd_transformed_item.colors = op.utility.Vector3dVector(colors_item/255.0)
# op.visualization.draw_geometries([pcd_transformed_item],window_name="1")


