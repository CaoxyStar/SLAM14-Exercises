#include<iostream>
#include<sophus/se3.hpp>
#include<opencv2/opencv.hpp>
#include<boost/format.hpp>
#include<pangolin/pangolin.h>
#include<mutex>
#include<chrono>
using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

//相机内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
//baseline
double baseline = 0.573;
//paths
string left_file = "/home/xiaoyu/slam14/ch8/left.png";
string disparity_file = "/home/xiaoyu/slam14/ch8/disparity.png";
boost::format fmt_others("/home/xiaoyu/slam14/ch8/%06d.png");

typedef Eigen::Matrix<double, 6, 6>Matrix6d;
typedef Eigen::Matrix<double, 2, 6>Matrix26d;
typedef Eigen::Matrix<double, 6, 1>Vector6d;


class JacobianAccumulator{
public:
        JacobianAccumulator(
                const cv::Mat &img1_,
                const cv::Mat &img2_,
                const VecVector2d &px_ref_,
                const vector<double> depth_ref_,
                Sophus::SE3d &T21_) : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_){
                        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
                }


        void accumulate_jacobian(const cv::Range &range);


        Matrix6d hessian() const { return H; }


        Vector6d bias() const { return b; }


        double cost_func() const { return cost; }


        VecVector2d projected_points() const { return projection; }


        void reset() {
                H = Matrix6d::Zero();
                b = Vector6d::Zero();
                cost = 0;
        }

private:
        const cv::Mat &img1;
        const cv::Mat &img2;
        const VecVector2d &px_ref;
        const vector<double> depth_ref;
        Sophus::SE3d &T21;
        VecVector2d projection;

        std::mutex hessian_mutex;
        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        double cost = 0;
};


int main()
{
        cv::Mat left_img = cv::imread(left_file, 0);
        cv::Mat disparity_img = cv::imread(disparity_file, 0);

        cv::RNG rng;
        int nPoints = 2000;
        int boarder = 20;
        VecVector2d pixels_ref;
        vector<double> depth_ref;

        for(int i=0;i<nPoints;i++){
                int x = rng.uniform(boarder, left_img.cols-boarder);
                int y = rng.uniform(boarder, left_img.rows-boarder);
                int disparity = disparity_img.at<uchar>(y, x);
                double depth = fx*baseline/disparity;
                depth_ref.push_back(depth);
                pixels_ref.push_back(Eigen::Vector2d(x, y));
        }



        return 0;
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21){
        
        const int iterations = 10;
        double cost = 0, lastCost = 0;
        auto t1 = chrono::steady_clock::now();
        JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

        for(int iter =0;iter<iterations;iter++){
                jaco_accu.reset();
        }
        
}

//二次插值法获取像素点灰度值
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
{
        const int half_patch_size = 1;
        int cnt_good = 0;
        Matrix6d hessian = Matrix6d::Zero();
        Vector6d bias = Vector6d::Zero();
        double cost_tmp =  0;

        for(size_t i = range.start;i<range.end;i++){
                Eigen::Vector3d point_ref = depth_ref[i]*Eigen::Vector3d((px_ref[i][0]-cx)/fx, (px_ref[i][1]-cy)/fy, 1);
                Eigen::Vector3d point_cur = T21*point_ref;
                if(point_cur[2]<0)
                        continue;
                
                float u = fx*point_cur[0]/point_cur[2]+cx, v=fy*point_cur[1]/point_cur[2]+cy;
                if(u<half_patch_size||u>img2.cols-half_patch_size||v<half_patch_size||v>img2.rows-half_patch_size)
                        continue;
                
                projection[i] = Eigen::Vector2d(u, v);
                double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
                double Z2Z = Z*Z, Z_inv = 1.0/Z, Z2_inv = Z_inv*Z_inv;
                cnt_good++;

                Matrix26d J_pixel_xi;
                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                for(int x = -half_patch_size; x<=half_patch_size; x++){
                        for(int y = -half_patch_size; y<=half_patch_size; y++){
                                double error = GetPixelValue(img1, px_ref[i][0]+x, px_ref[i][1]+y) - 
                                        GetPixelValue(img2, u+x, v+y);

                                
                                Eigen::Vector2d J_img_pixel;
                                J_img_pixel = Eigen::Vector2d(
                                        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                                        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                                Vector6d J = -1.0*(J_img_pixel.transpose()*J_pixel_xi).transpose();
                                hessian += J*J.transpose();
                                bias += -error*J;
                                cost_tmp += error*error;
                        }
                }
                if(cnt_good){
                        unique_lock<mutex> lck(hessian_mutex);
                        H+=hessian;
                        b+=bias;
                        cost+=cost_tmp/cnt_good;
                }
        }
}