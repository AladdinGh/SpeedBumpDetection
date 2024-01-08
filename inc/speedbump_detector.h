#ifndef SPEEDBUMP_DETECTOR_H
#define SPEEDBUMP_DETECTOR_H


#include <iostream>
#include <opencv2/opencv.hpp>


class speedbump_detector
{

public:
speedbump_detector(); 
cv::Mat detect_speedbump(cv::Mat img);

};

#endif