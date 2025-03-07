

import os
import shutil

# 示例用法


# def rename_and_move_files(source_dir, target_dir, keyword, new_name_prefix):
#     """
#     从 source_dir 中读取名字包含指定字符串的文件，
#     重命名后保存到 target_dir 中。
#
#     参数:
#         source_dir (str): 源文件夹路径
#         target_dir (str): 目标文件夹路径
#         keyword (str): 文件名中需要包含的字符串
#         new_name_prefix (str): 重命名时的新文件名前缀
#     """
#     # 确保目标文件夹存在
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     # 遍历源文件夹中的所有文件
#     for filename in os.listdir(source_dir):
#         # 检查文件名是否包含指定字符串
#         if keyword in filename:
#             # 构造完整的文件路径
#             source_path = os.path.join(source_dir, filename)
#
#             # 构造新的文件名
#             new_filename = f"{new_name_prefix}_{filename}"
#             target_path = os.path.join(target_dir, new_filename)
#
#             # 复制并重命名文件
#             shutil.copy(source_path, target_path)
#             print(f"文件 {filename} 已重命名为 {new_filename} 并移动到 {target_dir}")
source_dir="E:\mmdetection3d\data\s3dis1\\area_1\pano"
target_dir="E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1"
# target_rgb_name_list=[]
# target_depth_name_list=[]
# target_json_name_list=[]
target_key_list=[]
for i in os.listdir(target_dir):
    # target_rgb_name_list.append(os.path.join(target_dir),f"{i}_rgb.png")
    # target_depth_name_list.append(os.path.join(target_dir),f"{i}_depth.png")
    # target_json_name_list.append(os.path.join(target_dir),f"{i}_pose.json")
    if os.path.isdir(os.path.join(target_dir,i)):
        target_key_list.append(i+"_")
for key in ["depth","pose","rgb"]:
    subfix=".png"
    if key=="pose":
        subfix=".json"
    for i in target_key_list:
        is_first = False
        for item in os.listdir(os.path.join(source_dir,key)):
            if i in item:
                if is_first==False:
                    is_first=True
                    old_file=os.path.join(source_dir,key,item)
                    new_file= os.path.join(target_dir,i[:-1],f"{i}{key}{subfix}")
                    shutil.copy(old_file,new_file)
                    print(f"文件 {old_file} 已重命名为 {new_file}")
                elif is_first==True:
                    shutil.copy(os.path.join(source_dir, key, item), os.path.join(target_dir, i[:-1],item))
                    print(f"文件 {os.path.join(source_dir, key, item)} 已重命名为 {os.path.join(target_dir, i[:-1],item)}")







# 示例用法

# keyword_to_search = "example"  # 文件名中需要包含的字符串
# new_name_prefix = "new_"  # 新文件名前缀
#
# rename_and_move_files(source_directory, target_directory, keyword_to_search, new_name_prefix)