#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>

using namespace cv;

int main()
{
    Mat digits = imread("../digits.png");
    if(digits.empty())
    {
        std::cout << "检查图片路径" << std::endl;
    }

    Mat gray;
    cvtColor(digits, gray, COLOR_RGB2GRAY);

    mkdir("../Number", 0777);

    int imgName = 0;
    int imgIndex = 0;

    int step = 20;
    int rowsCount = gray.rows / step;   //原图为1000*2000
    int colsCount = gray.cols / step;   //裁剪为5000个20*20的小图块

    for (int i = 0; i < rowsCount; i++)
    {
        if (i % 5 == 0 && i != 0)
        {
            imgName++;
            imgIndex = 0;
        }

        int offsetRow = i * step;  //行上的偏移量
        for (int j = 0; j < colsCount; j++)
        {
            int offsetCol = j * step; //列上的偏移量
            Mat temp = gray(Rect (offsetCol, offsetRow, step, step));
            std::string filestring = std::string("../") + "Number" + "/" + std::string(std::to_string(imgName));
            std::string imgname = std::string(filestring) + "/" + std::string(std::to_string(imgIndex)) + ".png";
            std::fstream _file;
            _file.open(filestring, std::ios::in); //判断文件夹是否存在
            if(!_file)
            {
                mkdir(filestring.c_str(), 0777);
            }

            imwrite(imgname, temp);
            imgIndex++;
        }
    }

    return 0;
}