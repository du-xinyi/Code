import os, random, shutil

base_dir = "./datasets/coco/images/train2017/"  # 源图片文件夹路径
final_Dir = "./images/"  # 移动到新的文件夹路径

pathDir = os.listdir(base_dir)
filenumber = len(pathDir)

if filenumber > 30:
    picknumber = 150  # 所取图片数量
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    if not os.path.exists(final_Dir):
        os.makedirs(final_Dir)
    for name in sample:
        shutil.copy(base_dir + name, final_Dir + name)