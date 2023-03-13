import os

os.chdir(r'./img') # 需要修改后缀的文件目录

files = os.listdir('./') # 列出当前目录下所有的文件
for fileName in files:
    print(fileName)
    portion = os.path.splitext(fileName)
    newName = portion[0] + ".png" # 修改为目标后缀
    os.rename(fileName, newName)
    print("Change Suffix Right")
