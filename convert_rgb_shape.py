import cv2
import os
def resize_and_save_image(input_path, output_path, new_size):
    """
    读取图像，调整大小，并保存到指定路径。
    :param input_path: 输入图像的路径
    :param output_path: 输出图像的路径
    :param new_size: 新的图像大小 (width, height)
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像：{input_path}")

    # 调整图像大小
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # 保存图像
    cv2.imwrite(output_path, img_resized)
    print(f"图像已保存到：{output_path}")

# 示例使用
folder_path = "E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1"
output_path = "E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1"
new_size = (1024, 512)  # 新的宽度和高度
image_files=[]
for filename in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path,filename)):
            file_path = os.path.join(folder_path, filename,f"{filename}_rgb.png")
            image_files.append(file_path)
for image_file in image_files:
    resize_and_save_image(image_file,image_file,new_size)
