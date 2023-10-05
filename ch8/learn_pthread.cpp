#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

mutex test;

void doit(int i) { 
	std::unique_lock<std::mutex> lck(test);
	cout << i << " ll finished!" << endl; 
}



int main() {

	thread th[10];
	for(int i = 0;i<10;i++){
		th[i] = thread(doit, i);
	}
	for(int i =0;i<10;i++){
		th[i].join();
	}
	return 0;
}

