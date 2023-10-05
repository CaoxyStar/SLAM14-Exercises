#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<chrono>
#include<Eigen/Core>
#include<Eigen/Geometry>

using namespace std;
using namespace cv;


//由匹配的特征点计算R和t
void pose_estimation_2d2d(vector<KeyPoint>keypoint_1, vector<KeyPoint>keypoint_2, vector<DMatch>matches, Mat &R, Mat &t)
{
        //给出相机内参矩阵
        Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

        //找出两幅图中匹配到的特征点
        vector<Point2f>points1;
        vector<Point2f>points2;
        for(int i =0;i<(int)matches.size();i++)
        {
                points1.push_back(keypoint_1[matches[i].queryIdx].pt);
                points2.push_back(keypoint_2[matches[i].trainIdx].pt);
        }

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

//由像素坐标计算归一化坐标(相机坐标系下)
Point2d pixel2cam(const Point2d &p, const Mat &K)
{
        return Point2d(
                (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}

//三角测量求特征点深度
void triangulation(const vector<KeyPoint> &keypoint_1, const vector<KeyPoint> &keypoint_2,
        const vector<DMatch> &matches, const Mat &R, const Mat &t, vector<Point3d> &pts3d_1, vector<Point3d> &pts3d_2){
                Mat T1 = (Mat_<float>(3, 4)<<
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0);
                //初始化变换矩阵
                Mat T2 = (Mat_<float>(3, 4)<<
                        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
                Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

                //找到配对特征点的归一化坐标
                vector<Point2f>points1;
                vector<Point2f>points2;
                for(int i =0;i<(int)matches.size();i++)
                {
                        points1.push_back(pixel2cam(keypoint_1[matches[i].queryIdx].pt, K));
                        points2.push_back(pixel2cam(keypoint_2[matches[i].trainIdx].pt, K));
                }

                Mat pts_4d;
                triangulatePoints(T1, T2, points1, points2, pts_4d);

                //该处存储的深度信息为相对于points1即图1， 计算图2深度可采用R*p+t[2]
                for(int i =0;i<pts_4d.cols;i++){
                        Mat x = pts_4d.col(i);
                        x /= x.at<float>(3, 0);
                        Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
                        pts3d_1.push_back(p);

                        Mat q_ = T2*(Mat_<float>(4, 1)<< p.x, p.y, p.z, 1.0);
                        Point3d q(q_.at<float>(0, 0), q_.at<float>(1, 0), q_.at<float>(2, 0));
                        pts3d_2.push_back(q);
                }
        }

//ICP求解R和t，适用于3D-3D
void SVD_sloveICP(const vector<Point3d> &pts1, const vector<Point3d> &pts2, Mat &R, Mat &t){
        //计算两幅图中特征点质心
        Point3d p1, p2;
        int N = pts1.size();
        for(int i=0;i<N;i++){
                p1+=pts1[i];
                p2+=pts2[i];
        }
        p1 = Point3d(Vec3d(p1)/N);
        p2 = Point3d(Vec3d(p2)/N);

        //求出去质心3D坐标
        vector<Point3d>q1(N), q2(N);
        for(int i =0;i<N;i++)
        {
                q1[i] = pts1[i] - p1;
                q2[i] = pts2[i] - p2;
        }

        //计算W矩阵
        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for(int i = 0;i<N;i++){
                W+=Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z)*Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
        }
        cout<<"W = "<<W<<endl;

        //对W矩阵进行SVD分解
        Eigen::JacobiSVD<Eigen::Matrix3d>svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        cout<<"U = "<<U<<endl;
        cout<<"V = "<<V<<endl;

        //计算R和t
        Eigen::Matrix3d R_ = U*(V.transpose());
        if(R_.determinant()<0){
                R_ = -R_;
        }
        Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_*Eigen::Vector3d(p2.x, p2.y, p2.z);

        R = (Mat_<double>(3, 3)<<
                R_(0, 0), R_(0, 1), R_(0, 2),
                R_(1, 0), R_(1, 1), R_(1, 2),
                R_(2, 0), R_(2, 1), R_(2, 2));
        t = (Mat_<double>(3, 1)<<t_(0, 0), t_(1, 0), t_(2, 0));
}

int main(int argc, char **argv)
{
        //判断传入参数
        if(argc!=3)
        {
                cout<<"can't find img1 and img2!"<<endl;
                return 1;
        }
        Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
        assert(img_1.data!=nullptr && img_2.data!=nullptr);

        //相机内参
        Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

        //初始化对象
        vector<KeyPoint> keypoint_1, keypoint_2;        //存储特征点
        Mat descriptors_1, descriptors_2;               //存储特征点描述子
        Ptr<FeatureDetector> detector = ORB::create();  //创建寻找特征点容器
        Ptr<DescriptorExtractor> descriptor = ORB::create();    //创建描述容器
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");       //创建特征点匹配器

        //检测特征点
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        detector->detect(img_1, keypoint_1);
        detector->detect(img_2, keypoint_2);

        //描述子
        descriptor->compute(img_1, keypoint_1, descriptors_1);
        descriptor->compute(img_2, keypoint_2, descriptors_2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"extract ORB cost = "<<time_used.count()<<" seconds."<<endl;

        //显示寻找到的特征点
        Mat outimg1;
        drawKeypoints(img_1, keypoint_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("img1 ORB features", outimg1);
        Mat outimg2;
        drawKeypoints(img_2, keypoint_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("img2 ORB features", outimg2);
        
        //对两幅图的特征点进行匹配
        vector<DMatch> matches;
        t1 = chrono::steady_clock::now();
        matcher->match(descriptors_1, descriptors_2, matches);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"matches ORB cost = "<<time_used.count()<<" seconds."<<endl;

        //寻找最佳匹配
        auto min_max = minmax_element(matches.begin(), matches.end());
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;

        cout<<"--Max dist: "<<max_dist<<endl;
        cout<<"--Min dist: "<<min_dist<<endl;

        vector<DMatch> good_matches;
        for(int i=0;i<descriptors_1.rows;i++){
                if(matches[i].distance<=max(2*min_dist, 30.0)){
                        good_matches.push_back(matches[i]);
                }
        }

        cout<<"matched point num: "<<matches.size()<<endl;
        cout<<"good matches point num: "<<good_matches.size()<<endl;

        //显示最佳匹配
        Mat img_goodmatch;
        drawMatches(img_1, keypoint_1, img_2, keypoint_2, good_matches, img_goodmatch);
        namedWindow("good_match", WINDOW_NORMAL);
        imshow("good_match", img_goodmatch);
        waitKey(0);
        destroyAllWindows();

        //计算位姿变换R和t
        t1 = chrono::steady_clock::now();
        Mat R, t;
        pose_estimation_2d2d(keypoint_1, keypoint_2, good_matches, R, t);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"compute R and t cost = "<<time_used.count()<<" seconds."<<endl;

        Mat t_x = (Mat_<double>(3, 3)<<0, -t.at<double>(2,0), t.at<double>(1,0),
                t.at<double>(2,0), 0, -t.at<double>(0,0), -t.at<double>(1,0), t.at<double>(0,0), 0);
        cout<<"本质矩阵 t^R = "<<endl<<t_x*R<<endl;

        // //验证对极约束成立
        // Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
        // for(DMatch m:good_matches){
        //         Point2d pt1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        //         Mat y1 = (Mat_<double>(3, 1)<<pt1.x, pt1.y, 1);
        //         Point2d pt2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        //         Mat y2 = (Mat_<double>(3, 1)<<pt2.x, pt2.y, 1);
        //         Mat d = y2.t()*t_x*R*y1;
        //         cout<<"epipolar constraint = "<<d<<endl;
        // }

        //三角测量求特征点深度
        t1 = chrono::steady_clock::now();
        vector<Point3d>pts3d_1, pts3d_2;
        triangulation(keypoint_1, keypoint_2, good_matches, R, t, pts3d_1, pts3d_2);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"compute triangulation cost = "<<time_used.count()<<" seconds."<<endl;

        //通过pnp计算R和t，传入第一张图中特征点在相机坐标系下的3d坐标，传入配对特征点在第二张图中的像素坐标
        Mat pnp_r, pnp_t;
        //初始化第二张图中配对特征点像素坐标
        vector<Point2f>points2d_2;
        for(int i =0;i<(int)good_matches.size();i++)
                points2d_2.push_back(keypoint_2[good_matches[i].trainIdx].pt);
        t1 = chrono::steady_clock::now();
        solvePnP(pts3d_1, points2d_2, K, Mat(), pnp_r, pnp_t, false);
        //将旋转向量转化为旋转矩阵
        Mat pnp_R;
        Rodrigues(pnp_r, pnp_R);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"compute pnp cost = "<<time_used.count()<<" seconds."<<endl;
        cout<<"pnp R = "<<endl<<pnp_R<<endl;
        cout<<"pnp t = "<<endl<<pnp_t<<endl;

        //验证R精度
        Mat pnp_21_r, pnp_21_t;
        //初始化第二张图中配对特征点像素坐标
        vector<Point2f>points2d_1;
        for(int i =0;i<(int)good_matches.size();i++)
                points2d_1.push_back(keypoint_1[good_matches[i].queryIdx].pt);
        solvePnP(pts3d_2, points2d_1, K, Mat(), pnp_21_r, pnp_21_t, false);
        //将旋转向量转化为旋转矩阵
        Mat pnp_21_R;
        Rodrigues(pnp_21_r, pnp_21_R);
        cout<<"pnp 21 R = "<<endl<<pnp_21_R<<endl;
        cout<<"pnp 21 t = "<<endl<<pnp_21_t<<endl;
        //计算R12和R21，理论结果应该接近单位矩阵
        Mat R_R = pnp_21_R*pnp_R;
        cout<<"R * R21 = "<<R_R<<endl;


        //ICP计算R和t(注意此处由p=R*p‘+t推导，故为2帧到1帧的转化，所以将pts3d_2置前,pts3d_1置后可得到1帧到2帧的转化)
        Mat icp_R, icp_t;
        SVD_sloveICP(pts3d_2, pts3d_1, icp_R, icp_t);
        cout<<"ICP R = "<<endl<<icp_R<<endl;
        cout<<"ICP t = "<<endl<<icp_t<<endl;
        
        return 0;
}