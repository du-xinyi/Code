import os
import random
import shutil

def moveFile(train_img_Dir, train_mask_Dir):
        img_pathDir = os.listdir(train_img_Dir)                    # 提取图片的原始路径
        filenumber = len(img_pathDir)
        # 自定义test的数据比例
        test_rate = 0.1                                            # 0.1，就是10%
        test_picknumber = int(filenumber*test_rate)                # 按照test_rate比例从文件夹中取一定数量图片
        # 自定义val的数据比例
        val_rate = 0.1
        val_picknumber = int(filenumber*val_rate)                  # 按照val_rate比例从文件夹中取一定数量图片
        # 选取移动到test中的样本
        sample1 = random.sample(img_pathDir, test_picknumber)      # 随机选取picknumber数量的样本图片
        for i in range(0, len(sample1)):
            sample1[i] = sample1[i][:-4]                           # 去掉图片的拓展名，移动标注时需要这个列表
        for name in sample1:
            src_img_name1 = train_img_Dir + name
            dst_img_name1 = test_img_Dir + name
            shutil.move(src_img_name1 + '.jpg', dst_img_name1 + '.jpg')     # 加上图片的拓展名，移动图片
            src_mask_name1 = train_mask_Dir + name
            dst_mask_name1 = test_mask_Dir + name
            shutil.move(src_mask_name1 + '.txt', dst_mask_name1 + '.txt')   # 加上标注文件的拓展名，移动标注文件
        # 选取移动到val中的样本
        img_pathDir = os.listdir(train_img_Dir)                    # 这时图片目录里的文件数目会变
        sample2 = random.sample(img_pathDir, val_picknumber)       # 但是抽出来的数目，还是用之前算的
        for i in range(0, len(sample2)):
            sample2[i] = sample2[i][:-4]
        for name in sample2:
            src_img_name2 = train_img_Dir + name
            dst_img_name2 = val_img_Dir + name
            shutil.move(src_img_name2 + '.jpg', dst_img_name2 + '.jpg')
            src_mask_name2 = train_mask_Dir + name
            dst_mask_name2 = val_mask_Dir + name
            shutil.move(src_mask_name2 + '.txt', dst_mask_name2 + '.txt')
        return

if __name__ == '__main__':
    # 分割前的图片和标注目录
    img_Dir = './train/images/'
    mask_Dir = './train/labels/'
    # test路径：图片和标注目录
    test_img_Dir = './test/images/'
    test_mask_Dir = './test/labels/'
    # val路径：图片和标注文目录
    val_img_Dir = './val/images/'
    val_mask_Dir = './val/labels/'

    if not os.path.exists(test_img_Dir):
        os.makedirs(test_img_Dir)
    if not os.path.exists(test_mask_Dir):
        os.makedirs(test_mask_Dir)
    if not os.path.exists(val_img_Dir):
        os.makedirs(val_img_Dir)
    if not os.path.exists(val_mask_Dir):
        os.makedirs(val_mask_Dir)
    # 运行划分数据集函数
    moveFile(img_Dir, mask_Dir)
    print("Partition Completion")
