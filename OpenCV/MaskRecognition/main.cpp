#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

//#define CAMERA //���

#ifndef CAMERA
    #define VIDEO //��Ƶ
#endif

int main()
{
    cv::CascadeClassifier mFaceDetector;
    cv::CascadeClassifier mEyeDetector;
    cv::CascadeClassifier mMouthDetector;
    cv::CascadeClassifier mNoseDetector;

    /*�������������������ļ�*/
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

    //�����Ƶ�Ƿ��
    if (!capture.isOpened())
    {
        std::cout << "Cannot Open the Camera or Video" << std::endl;

        return -1;
    }

    /*��Ƶ��ز���*/
    double rate = capture.get(CAP_PROP_FPS); //֡��
    double height = capture.get(CAP_PROP_FRAME_HEIGHT); //�߶�
    double width = capture.get(CAP_PROP_FRAME_WIDTH); //���
    double fps = capture.get(CAP_PROP_FPS); //ˢ����
    double fourcc = capture.get(CAP_PROP_FOURCC); //���뷽ʽ

    cv::Mat frame; //���ڵ���Ƶ֡
    cv::Mat mElabImage; //����frameͼ��

    cv::namedWindow("Extracted Frame");

    int delay = 1000 / rate; //��֮֡��ļ��ʱ��


    VideoWriter SaveVideo("../save.mp4", fourcc, fps, Size(width,height ),true); //������Ƶ

    /*ѭ���������е�֡*/
    while (1)
    {
        /*��һ֡*/
        if (!capture.read(frame))
            break;

        frame.copyTo(mElabImage);

        /*�����*/
        float scaleFactor = 3.0f; //��������
        vector< cv::Rect > faceVec;
        mFaceDetector.detectMultiScale(frame, faceVec, scaleFactor);
        int i, j;
        for (i = 0; i < faceVec.size(); i++)
        {
            cv::rectangle(mElabImage, faceVec[i], CV_RGB(255, 0, 0), 2);
            cv::Mat face = frame(faceVec[i]);

            /*����۾�*/
            vector< cv::Rect > eyeVec;
            mEyeDetector.detectMultiScale(face, eyeVec);

            for (j = 0; j < eyeVec.size(); j++)
            {
                cv::Rect rect = eyeVec[j];
                rect.x += faceVec[i].x;
                rect.y += faceVec[i].y;

                cv::rectangle(mElabImage, rect, CV_RGB(0, 255, 0), 2);
            }

            /*������*/
            vector< cv::Rect > noseVec;
            mNoseDetector.detectMultiScale(face, noseVec, 3);

            for (j = 0; j < noseVec.size(); j++)
            {
                cv::Rect rect = noseVec[j];
                rect.x += faceVec[i].x;
                rect.y += faceVec[i].y;

                cv::rectangle(mElabImage, rect, CV_RGB(0, 0, 255), 2);
            }

            /*������*/
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

        SaveVideo.write(mElabImage); //��Ƶ����

        /*ESC�˳�*/
        if(waitKey(20) == 27)
        {
            cv::destroyAllWindows();

            break;
        }
    }

    capture.release();

    return 0;
}