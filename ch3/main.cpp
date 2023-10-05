#include<iostream>
#include<math.h>
using namespace std;

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
using namespace Eigen;

int main()
{
        //旋转矩阵与旋转向量的定义与转化
        Matrix3d rotation_matrix = Matrix3d::Identity();
        AngleAxisd rotation_vector(M_PI/4, Vector3d(0, 0, 1));
        cout.precision(3);
        rotation_matrix = rotation_vector.toRotationMatrix();
        cout<<"rotation_matrix = \n"<<rotation_matrix<<endl;

        //向量可通过乘以旋转矩阵或旋转向量进行旋转变化
        Vector3d v(1, 0, 0);
        Vector3d v_rotated = rotation_matrix*v;
        cout<<"(1, 0, 0) after rotation = "<<v_rotated.transpose()<<endl;

        //旋转矩阵与欧拉角的转化
        Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
        cout<<"yaw pitch roll = "<<euler_angles.transpose()<<endl;

        //可通过旋转向量或旋转矩阵构造变换矩阵
        Isometry3d T = Isometry3d::Identity();
        T.rotate(rotation_matrix);
        T.pretranslate(Vector3d(1, 3, 4));
        cout<<"Transform matrix = \n"<<T.matrix()<<endl;

        //向量通过变换矩阵变换
        Vector3d v_transformed = T*v;
        cout<<"v transformed = "<<v_transformed.transpose()<<endl;

        //四元数与旋转矩阵和旋转向量间的变换，程序计算进行了简化
        Quaterniond q = Quaterniond(rotation_vector);
	cout<<"******************"<<endl;
        cout<<"quaternion form rotation vector = "<<q.coeffs().transpose()<<endl;
        Vector4d p = q.coeffs().transpose();
	for(int i =0;i<4;i++)
		cout<<p[i]<<endl;
	for(int i =0;i<4;i++)
		cout<<p[i]<<endl;
	cout<<"**********************"<<endl;
	Vector3d v_rotated_by_quater = q*v;
        cout<<"v rotated by quater = "<<v_rotated_by_quater.transpose()<<endl;
        cout<<"math compute = "<<(q*Quaterniond(0, 1, 0, 0)*q.inverse()).coeffs().transpose()<<endl;
        return 0;
}
