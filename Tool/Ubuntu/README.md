# 更新日志
## 2023.03.10
- 添加OpenCV/Compilation
- 添加OpenCV/DependencyLibrary
## 2023.02.14
- 修复Google-Chinesization存在汉化失败的情况
## 2023.02.13
- 添加Backup
- 添加Restore
## 2023.02.11
- 添加Google-Chinesization

# 文件说明
## OpenCV/Compilation
编译安装OpenCV，放置于如下位置
```
    CMakeLists
    build
        Compilation.sh
    contrilb
    ...
```
## OpenCV/DependencyLibrary
安装OpenCV依赖库大全
## Backup
打包Ubuntu除/proc /tmp /lost+found /media /mnt /run /dev 外所有文件
## Google-Chinesization
将英文系统下的谷歌浏览器改为中文
## Restore
将Backup打包的文件恢复至原位置
