import os, random, shutil

base_dir = "./datasets/COCO2017/train2017/"  # 源图片文件夹路径
final_Dir = "./images/"  # 移动到新的文件夹路径

pathDir = os.listdir(base_dir)
filenumber = len(pathDir)
rate = 0.75  # 自定义抽取图片的比例
picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
if not os.path.exists(final_Dir):
    os.makedirs(final_Dir)
for name in sample:
    shutil.copy(base_dir + name, final_Dir + name)