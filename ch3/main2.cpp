#include<iostream>
using namespace std;
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
using namespace Eigen;

int main()
{
        Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
        //使用直接初始化四元数需要首先归一化
        q1.normalize();
        q2.normalize();
        Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
        Vector3d p1(0.5, 0, 0.2);

        //可用四元数直接初始化变换矩阵中的旋转矩阵项
        Isometry3d T1w(q1), T2w(q2);
        T1w.pretranslate(t1);
        T2w.pretranslate(t2);

        //1w为world坐标系的向量向萝卜头1变换， 逆为反变换
        Vector3d p2 = T2w*T1w.inverse()*p1;
        cout<<"p1 in 2axix = "<<p2.transpose()<<endl;

        return 0;
}