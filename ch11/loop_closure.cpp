#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
        cout<<"reading database..."<<endl;
        DBoW3::Vocabulary vocab("/home/xiaoyu/slam14/ch11/vocabulary.yml.gz");
        if(vocab.empty()){
                cout<<"vocabulary does not exist."<<endl;
                return 1;
        }
        cout<<"reading images..."<<endl;
        vector<Mat>images;
        for(int i =0;i<10;i++){
                string path = "/home/xiaoyu/slam14/ch11/data/"+to_string(i+1)+".png";
                images.push_back(imread(path));
        }

        cout<<"detecting ORB feature..."<<endl;
        Ptr<Feature2D> detector = ORB::create();
        vector<Mat> descriptors;
        for(Mat &image:images){
                vector<KeyPoint> keypoint;
                Mat descriptor;
                detector->detectAndCompute(image, Mat(), keypoint, descriptor);
                descriptors.push_back(descriptor);
        }

        cout<<"comparimg images with images "<<endl;
        for(int i =0;i<images.size();i++){
                DBoW3::BowVector v1;
                vocab.transform(descriptors[i], v1);
                for(int j=0;j<images.size();j++){
                        DBoW3::BowVector v2;
                        vocab.transform(descriptors[j], v2);
                        double score = vocab.score(v1, v2);
                        cout<<"image "<<i<<"vs image "<<j<<" : "<<score<<endl;
                }
                cout<<endl;
        }

        cout<<"comparing images with database "<<endl;
        DBoW3::Database db(vocab, false, 0);
        for(int i =0;i<descriptors.size();i++){
                db.add(descriptors[i]);
        }
        cout<<"database info: "<<db<<endl;
        for(int i=0;i<descriptors.size();i++){
                DBoW3::QueryResults ret;
                db.query(descriptors[i], ret, 4);
                cout<<"searching for image "<<i<<" returns "<<ret<<endl<<endl;
        }

        return 0;
}