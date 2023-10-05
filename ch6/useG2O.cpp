#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
using namespace std;


//创建顶点（继承父类，父类传入变量依次为：待优化变量纬度和数据类型）
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW //解决Eigen字节对齐问题

        //virtual虚函数可使类多态，override保证子类虚函数在父类存在
        //重置
        virtual void setToOriginImpl() override {
                _estimate << 0, 0, 0;
        }

        //更新
        virtual void oplusImpl(const double *update) override {
                _estimate += Eigen::Vector3d(update);
        }

        virtual bool read(istream &in) {}
        //函数名后加const防止成员被修改
        virtual bool write(ostream &out) const {}
};


//创建边（改边为一元边，初始化时父类参数：观测值纬度，类型，链接顶点类型）
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        CurveFittingEdge(double x):BaseUnaryEdge(), _x(x) {}

        //计算误差
        virtual void computeError() override {
                const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
                const Eigen::Vector3d abc = v->estimate();
                _error(0, 0) = _measurement - std::exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);
        }
        //计算雅可比矩阵
        virtual void linearizeOplus() override {
                const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
                const Eigen::Vector3d abc = v->estimate();
                double y = exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);
                _jacobianOplusXi[0] = -_x * _x * y;
                _jacobianOplusXi[1] = -_x * y;
                _jacobianOplusXi[2] = -y;
        }

        virtual bool read(istream &in) {}

        virtual bool write(ostream &out) const {}

public:
        double _x;
};


//数据初始化、构建图与优化
int main()
{
        double ar = 5.0, br = 2.0, cr = 1.0;         // 真实参数值
        double ae = 4.0, be = 0.0, ce = 4.0;        // 估计参数值
        int N = 100;                                 // 数据点
        double w_sigma = 1.0;                        // 噪声Sigma值
        double inv_sigma = 1.0 / w_sigma;
        cv::RNG rng;                                 // OpenCV随机数产生器

        vector<double> x_data, y_data;      // 数据
        for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        }

        //<3, 1>代表优化变量纬度与误差值纬度
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
        //求解器类型
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

        //求解器类型初始化block进一步初始化求解器，然后初始化优化器
        auto solver = new g2o::OptimizationAlgorithmGaussNewton(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);

        //加入顶点
        CurveFittingVertex *v = new CurveFittingVertex();
        v->setEstimate(Eigen::Vector3d(ae, be, ce));
        v->setId(0);
        optimizer.addVertex(v);

        //加入边
        for(int i =0;i<N;i++){
                CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
                edge->setId(i);
                edge->setVertex(0, v);
                edge->setMeasurement(y_data[i]);
                edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1/(w_sigma*w_sigma));
                optimizer.addEdge(edge);
        }

        cout<<"start optimization !"<<endl;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        
        cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

        // 输出优化值
        Eigen::Vector3d abc_estimate = v->estimate();
        cout << "estimated model: " << abc_estimate.transpose() << endl;


        return 0;
}