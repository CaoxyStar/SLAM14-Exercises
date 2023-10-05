#include<iostream>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include<cmath>
#include<sophus/se3.hpp>
using namespace std;
using namespace Eigen;

int main()
{
        //从旋转矩阵/四元数到so3的转化
        Matrix3d R = AngleAxisd(M_PI/2, Vector3d(0, 0, 1)).toRotationMatrix();
        Quaterniond q(R);
        Sophus::SO3d SO3_R(R);
        Sophus::SO3d SO3_q(q);
        cout<<"SO(3) from matrix:\n"<<SO3_R.matrix()<<endl;
        cout<<"SO(3) from quaternion:\n"<<SO3_q.matrix()<<endl;

        //对数映射得到旋转矩阵对应的李代数（旋转向量），以及李代数和反对称矩阵之间的转化
        Vector3d so3 = SO3_R.log();
        cout<<"so3 = "<<so3.transpose()<<endl;
        cout<<"so3 hat =\n"<<Sophus::SO3d::hat(so3)<<endl;
        cout<<"so3 hat vee = "<<Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose()<<endl;

        //增加扰动，exp将扰动旋转向量转化为旋转矩阵与原始矩阵相乘（此处不需要hat）
        Vector3d update_so3(1e-4, 0, 0);
        Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3)*SO3_R;
        cout<<"SO3 updated = \n"<<SO3_updated.matrix()<<endl;
        cout<<"*****************************"<<endl;

        //对SE(3)操作
        Vector3d t(1, 0, 0);
        Sophus::SE3d SE3_Rt(R, t);
        Sophus::SE3d SE3_qt(q, t);
        cout<<"SE3 from R,t =\n"<<SE3_Rt.matrix()<<endl;
        cout<<"SE3 from q,t =\n"<<SE3_qt.matrix()<<endl;

        //SE3到se3的转化通过exp和log实现，以及hat和vee实现se3与反对称矩阵的转化
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        Vector6d se3 = SE3_Rt.log();
        cout<<"se3 = "<<se3.transpose()<<endl;
        cout<<"se3 hat = \n"<<Sophus::SE3d::hat(se3)<<endl;
        cout<<"se3 hat vee = "<<Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose()<<endl;

        //SE3的扰动
        Vector6d update_se3;
        update_se3.setZero();
        update_se3(0, 0)=1e-4;
        cout<<update_se3.transpose()<<endl;
        Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3)*SE3_Rt;
        cout<<"updated SE3 = \n"<<SE3_updated.matrix()<<endl;
        return 0;
}