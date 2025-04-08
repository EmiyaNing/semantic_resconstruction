import pdb
import os

import pickle
import json
import cv2


import numpy as np

from visualize_ori_point import draw_scenes

def get_file_names(directory):
    file_names = os.listdir(directory)
    return file_names


def get_3Dpointsv2(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (2*x/float(W)-1.0)*np.pi
    phi = (y/float(H)-0.5)*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),-np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])


def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (1.0-2*x/float(W))*np.pi
    phi = (0.5-y/float(H))*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])


if __name__ == '__main__':
    dir_path  = './Stanford3dDataset_v1.2_Aligned_Version/Area_2/'
    file_list = get_file_names(dir_path)
    file_list = sorted(file_list)
    file_list = file_list[1:]

    with open('s3dis_infos_Area_2.pkl', 'rb') as f:
        pkl_file = pickle.load(f)

    instance_list = pkl_file['data_list']
    
    #pdb.set_trace()
    pano_rgb_list  = []
    pano_depth_list= []
    pano_pose_list = []
    points_list    = []
    for name in file_list:
        rgb_name   = dir_path + name + '/' + name + '_rgb.png'
        depth_name = dir_path + name + '/' + name + '_depth.png'
        pose_name  = dir_path + name + '/' + name + '_pose.json'
        point_name = './points/Area_2_' + name +'.bin'
        pano_rgb_list.append(rgb_name)
        pano_depth_list.append(depth_name)
        pano_pose_list.append(pose_name)
        points_list.append(point_name)

    for rgb_path, depth_path, pose_depth, points_path, instance_name in zip(pano_rgb_list, pano_depth_list, pano_pose_list, points_list, file_list):
        pano_rgb   = cv2.imread(rgb_path) 
        try:
            pano_rgb   = cv2.resize(pano_rgb, (1024, 512)) / 255.
        except cv2.error:
            pdb.set_trace()

        with open(pose_depth, 'r', encoding='utf-8') as file:
            pano_pose = json.load(file)
    

        camera_location      = np.array(pano_pose["camera_location"])
        camera_rt_matrix     = np.array(pano_pose["camera_rt_matrix"])
        camera_rt_matrix     = np.vstack((camera_rt_matrix,np.array([0,0,0,1])))
        camera_rt_matrix_inv = np.linalg.inv(camera_rt_matrix)


        camera_rotation      = np.array(pano_pose["final_camera_rotation"])
        flip_x,flip_y,flip_z = camera_location
        
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

        


        theta_x,theta_y,theta_z=camera_rotation
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

        camera_rotate_matrix = np.matmul(pred_rotate_matrix,np.matmul(rotate_x, np.matmul(rotate_y, rotate_z)))
        #pdb.set_trace()




        pano_depth = cv2.imread(depth_path, -1)
        pano_depth = cv2.resize(pano_depth, (1024, 512)) / 512.
        #pdb.set_trace()
        pano_depth[pano_depth > 10] = 0
        pano_pts   = get_3Dpointsv2(pano_depth, 512, 1024)
        pano_pts   = pano_pts.reshape(-1, 3)
        pts_color  = pano_rgb.reshape(-1, 3)


        homo_pano_pts   = np.hstack((pano_pts, np.ones((pano_pts.shape[0], 1))))

        rotated_pts     = np.dot(camera_rotate_matrix, homo_pano_pts.T).T
        rotated_pts     = rotated_pts[:, :3]

        rotated_pts[:, 0] = - rotated_pts[:, 0]#左右对称了
        rotated_pts[:,:]  += [flip_x,flip_y,flip_z]

        #pdb.set_trace()
        print("Draw pano point sences" + depth_path)
        draw_scenes(rotated_pts, point_colors=pts_color)
        points     = np.fromfile(points_path, dtype=np.float32).reshape(-1, 6)
        point_xyz  = points[:, :3]
        point_color= points[:, 3:] / 255
        print("Draw original point sences" + points_path)
        draw_scenes(point_xyz, point_colors=point_color)

    
    #pdb.set_trace()