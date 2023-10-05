#include<iostream>
#include<fstream>
#include<vector>
#include<unistd.h>
#include<pangolin/pangolin.h>
#include<sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groudtruth_file = "../groundtruth.txt";
string estimated_file = "../estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryTpye;

TrajectoryTpye ReadTrajectory(const string &path){
        ifstream fin(path);
        TrajectoryTpye trajectory;
        if(!fin){
                cout<<"file "<<path<<" not found."<<endl;
                return trajectory;
        }
        while(!fin.eof()){
                double time, tx, ty, tz, qx, qy, qz, qw;
                fin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
                Sophus::SE3d p1(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz));
                trajectory.push_back(p1);
        }
        return trajectory;
}

int main()
{
        TrajectoryTpye groudtruth = ReadTrajectory(groudtruth_file);
        TrajectoryTpye estimated =  ReadTrajectory(estimated_file);
        assert(!groudtruth.empty()&&!estimated.empty());
        assert(groudtruth.size() == estimated.size());
        
        double rmse = 0;
        for(size_t i = 0;i<estimated.size();i++){
                Sophus::SE3d p1 = estimated[i], p2 = groudtruth[i];
                double error = (p2.inverse()*p1).log().norm();
                rmse+=error*error;
        }
        rmse = rmse/double(estimated.size());
        rmse = sqrt(rmse);
        cout<<"RMSE="<<rmse<<endl;
        return 0;
}