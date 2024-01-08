#ifndef image_preprocessor_H
#define image_preprocessor_H
#include <opencv2/opencv.hpp>


#include <iostream>
#include <vector>

class image_preprocessor
{
public : 
    image_preprocessor(); 
    ~image_preprocessor();
    cv::Mat detect_road(cv::Mat);

private: 
    cv::Mat  m_image;
    int m_height, m_width; 
};

#endif