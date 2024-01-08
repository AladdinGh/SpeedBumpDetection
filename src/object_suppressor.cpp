#include "../inc/object_suppressor.h"
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

object_suppressor::object_suppressor()
{
    cout << "constructor of object suppressor"<< std::endl;
}
object_suppressor::~object_suppressor(){};

cv::Mat object_suppressor::suppress_object(cv::Mat img)
{
    cout << "removing objects from image"<< std::endl;

    CascadeClassifier car_cascade;
    car_cascade.load("/home/Aladdin/Desktop/Speed_Bump/speedbump/haarcascade_car.xml");
  
    Mat gray; 
    cvtColor(img,gray, cv::COLOR_BGR2GRAY);

    vector<cv::Rect> cars;
    //scaleFactor=1.1, 
    //minNeighbors=1, 
    //minSize=(10, 10)
    car_cascade.detectMultiScale(gray,cars,1.1,1,0,cv::Size(10,10));


    for (const auto& car : cars)
    {
        rectangle(img,car,cv::Scalar(255,255,255),-1); 
    }
   
    return(img);
}