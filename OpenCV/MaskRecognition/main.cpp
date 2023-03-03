#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

//#define CAMERA //相机

#ifndef CAMERA
    #define VIDEO //视频
#endif

int main()
{
    cv::CascadeClassifier mFaceDetector;
    cv::CascadeClassifier mEyeDetector;
    cv::CascadeClassifier mMouthDetector;
    cv::CascadeClassifier mNoseDetector;

    /*载入人脸特征分类器文件*/
    if (mFaceDetector.empty())
        mFaceDetector.load("../XML/haarcascade_frontalface_default.xml");
    if (mEyeDetector.empty())
        mEyeDetector.load("../XML/haarcascade_mcs_eyepair_big.xml");
    if (mNoseDetector.empty())
        mNoseDetector.load("../XML/haarcascade_mcs_nose.xml");
    if (mMouthDetector.empty())
        mMouthDetector.load("../XML/haarcascade_mcs_mouth.xml");

#ifdef CAMERA
    cv::VideoCapture capture(0);
#endif

#ifdef VIDEO
    cv::VideoCapture capture("../test.mp4");
#endif

    //检查视频是否打开
    if (!capture.isOpened())
    {
        std::cout << "Cannot Open the Camera or Video" << std::endl;

        return -1;
    }

    /*视频相关参数*/
    double rate = capture.get(CAP_PROP_FPS); //帧率
    double height = capture.get(CAP_PROP_FRAME_HEIGHT); //高度
    double width = capture.get(CAP_PROP_FRAME_WIDTH); //宽度
    double fps = capture.get(CAP_PROP_FPS); //刷新率
    double fourcc = capture.get(CAP_PROP_FOURCC); //编码方式

    cv::Mat frame; //现在的视频帧
    cv::Mat mElabImage; //备份frame图像

    cv::namedWindow("Extracted Frame");

    int delay = 1000 / rate; //两帧之间的间隔时间


    VideoWriter SaveVideo("../save.mp4", fourcc, fps, Size(width,height ),true); //保存视频

    /*循环播放所有的帧*/
    while (1)
    {
        /*下一帧*/
        if (!capture.read(frame))
            break;

        frame.copyTo(mElabImage);

        /*检测脸*/
        float scaleFactor = 3.0f; //缩放因子
        vector< cv::Rect > faceVec;
        mFaceDetector.detectMultiScale(frame, faceVec, scaleFactor);
        int i, j;
        for (i = 0; i < faceVec.size(); i++)
        {
            cv::rectangle(mElabImage, faceVec[i], CV_RGB(255, 0, 0), 2);
            cv::Mat face = frame(faceVec[i]);

            /*检测眼睛*/
            vector< cv::Rect > eyeVec;
            mEyeDetector.detectMultiScale(face, eyeVec);

            for (j = 0; j < eyeVec.size(); j++)
            {
                cv::Rect rect = eyeVec[j];
                rect.x += faceVec[i].x;
                rect.y += faceVec[i].y;

                cv::rectangle(mElabImage, rect, CV_RGB(0, 255, 0), 2);
            }

            /*检测鼻子*/
            vector< cv::Rect > noseVec;
            mNoseDetector.detectMultiScale(face, noseVec, 3);

            for (j = 0; j < noseVec.size(); j++)
            {
                cv::Rect rect = noseVec[j];
                rect.x += faceVec[i].x;
                rect.y += faceVec[i].y;

                cv::rectangle(mElabImage, rect, CV_RGB(0, 0, 255), 2);
            }

            /*检测嘴巴*/
            vector< cv::Rect > mouthVec;

            cv::Rect halfRect = faceVec[i];
            halfRect.height /= 2;
            halfRect.y += halfRect.height;

            cv::Mat halfFace = frame(halfRect);
            mMouthDetector.detectMultiScale(halfFace, mouthVec, 3);

            for (j = 0; j < mouthVec.size(); j++)
            {
                cv::Rect rect = mouthVec[j];
                rect.x += halfRect.x;
                rect.y += halfRect.y;

                cv::rectangle(mElabImage, rect, CV_RGB(255, 255, 255), 2);
            }

            if (noseVec.empty()&&!eyeVec.empty())
            {
                putText(mElabImage, "MASK!", Point(100, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
            }
        }

        cv::imshow("Extracted Frame", mElabImage);

        SaveVideo.write(mElabImage); //视频保存

        /*ESC退出*/
        if(waitKey(20) == 27)
        {
            cv::destroyAllWindows();

            break;
        }
    }

    capture.release();

    return 0;
}