#include <pangolin/pangolin.h>
#include <thread>
#include <iostream>

void Setup() {
    //创建显示窗口
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);

    //创建传感器
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 100),
        pangolin::ModelViewLookAt(2, 0, 2, 0, 0, 0, pangolin::AxisY)
    );

    //创建传感器句柄，并据此创建交互视图
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 0.5, 0.0, 0.5)
        .SetHandler(&handler);   

    pangolin::View& d_cam2 = pangolin::CreateDisplay()
        .SetBounds(0.5, 1.0, 0.5, 1.0)
        .SetHandler(&handler); 

    while(!pangolin::ShouldQuit()){
        //清空颜色和深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //启动传感器
        d_cam.Activate(s_cam);
        pangolin::glDrawColouredCube();
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f(0.8f, 0.f, 0.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(0, -1, -1);
        glColor3f(0.f, 0.8f, 0.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(-1, 0, -1);
        glColor3f(0.f, 0.f, 8.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(-1, -1, 0);
        glEnd();


        d_cam2.Activate(s_cam);

        pangolin::glDrawColouredCube();
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f(0.8f, 0.f, 0.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(0, -1, -1);
        glColor3f(0.f, 0.8f, 0.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(-1, 0, -1);
        glColor3f(0.f, 0.f, 8.f);
        glVertex3f(-1, -1, -1);
        glVertex3f(-1, -1, 0);
        glEnd();
        
        //刷新
        pangolin::FinishFrame();
    }   
}


int main(){
    std::thread rander_loop = std::thread(Setup);
    
    while(1){
        std::cout<<"hhh"<<std::endl;
    }
    rander_loop.join();
    return 0;

}