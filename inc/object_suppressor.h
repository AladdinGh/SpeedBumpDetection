#ifndef OBJECT_SUPPRESSOR_H
#define OBJECT_SUPPRESSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>


class object_suppressor{


public:
    object_suppressor(); 
    ~object_suppressor();
    cv::Mat suppress_object(cv::Mat img);

};

#endif