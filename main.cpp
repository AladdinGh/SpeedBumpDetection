#include <iostream>
#include <vector>
#include <string>
#include "object_suppressor.h"
#include "image_preprocessor.h"
#include "speedbump_detector.h"

#include <opencv2/opencv.hpp>

int main()
{

    cv::Mat img = cv::imread("Sample_Data/IMG_1935.jpg", cv::IMREAD_COLOR);

    object_suppressor _os; 
    img = _os.suppress_object(img); 

    image_preprocessor _ip; 
    img = _ip.detect_road(img); 


    //display_image(img); 

    

    
    
    speedbump_detector _sd; 
    img = _sd.detect_speedbump(img); 
    /*
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
    */
}