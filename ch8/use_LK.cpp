#include<iostream>
#include<opencv2/opencv.hpp>
#include<chrono>
using namespace std;
using namespace cv;

string file_1 = "/home/xiaoyu/slam14/ch8/1.png";
string file_2 = "/home/xiaoyu/slam14/ch8/2.png";

void pose_estimation_2d2d(vector<Point2f>points1, vector<Point2f>points2, Mat &R, Mat &t)
{
        //给出相机内参矩阵
        Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

        //计算基础矩阵
        Mat fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
        cout<<"fundamental_matrix is "<<endl<<fundamental_matrix<<endl;

        //计算本质矩阵
        Point2d principal_point(325.1, 249.7);
        double focal_length = 521;
        Mat essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
        cout<<"essential_matrix is "<<endl<<essential_matrix<<endl;

        //计算单应矩阵
        Mat homography_matrix;
        homography_matrix = findHomography(points1, points2, RANSAC, 3);
        cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

        //从本质矩阵svd求解r和t
        recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
        cout<<"R is "<<endl<<R<<endl;
        cout<<"t is "<<endl<<t<<endl;
}

int main()
{
        //获取前后帧图片
        Mat img1 = imread(file_1, 0);
        Mat img2 = imread(file_2, 0);

        //获取前一帧中关键点
        vector<KeyPoint> kp1;
        Ptr<GFTTDetector> detector = GFTTDetector::create(1000, 0.05, 20);
        detector->detect(img1, kp1);

        //初始化参数
        vector<Point2f>pt1, pt2;
        for(auto &kp: kp1)
                pt1.push_back(kp.pt);
        vector<uchar>status;
        vector<float>error;

        //LK计算，得到第二帧中对应第一帧关键点的像素位置
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"opencv compute LK cost: "<<time_used.count()<<" seconds."<<endl;
        cout<<"find point num: "<<pt1.size()<<endl;

        //绘图显示
        Mat img1_cv_LK, img2_cv_LK;
        cvtColor(img1, img1_cv_LK, CV_GRAY2BGR);
        cvtColor(img2, img2_cv_LK, CV_GRAY2BGR);
        for(int i=0;i<pt1.size();i++){
                if(status[i]){
                        circle(img1_cv_LK, pt1[i], 2, Scalar(0, 255, 0));
                        circle(img2_cv_LK, pt2[i], 2, Scalar(0, 255, 0));
                        line(img2_cv_LK, pt1[i], pt2[i], Scalar(0, 255, 0));
                }
        }

        namedWindow("img1_lk", WINDOW_NORMAL);
        namedWindow("img2_lk", WINDOW_NORMAL);
        imshow("img1_lk", img1_cv_LK);
        imshow("img2_lk", img2_cv_LK);
        waitKey(0);
        destroyAllWindows();

        //利用匹配点与对极几何计算R和t
        t1 = chrono::steady_clock::now();
        Mat R, t;
        pose_estimation_2d2d(pt1, pt2, R, t);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"compute R and t cost = "<<time_used.count()<<" seconds."<<endl;

        return 0;
}