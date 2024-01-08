#include "../inc/image_preprocessor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

image_preprocessor::image_preprocessor()
{
    std::cout << "constructor of image preprocessor" << std::endl;
}

image_preprocessor::~image_preprocessor()
{
}


cv::Mat crop_image(cv::Mat input_image, float scale_rows, float scale_cols)
{


    int mid_img_height = (int)input_image.rows*scale_rows; //0.45
    int mid_img_width = (int)input_image.cols*scale_cols; //0.15

    // Create a mask (black image)
    cv::Mat mask = cv::Mat::zeros(input_image.size(), CV_8UC1);

    // Define the vertices of the triangle
    std::vector<cv::Point> vertices;
    vertices.push_back(cv::Point(0, input_image.rows));  // Vertex 1
    vertices.push_back(cv::Point(0,mid_img_height));  // Vertex 3
    vertices.push_back(cv::Point(input_image.cols, mid_img_height));  // Vertex 1
    vertices.push_back(cv::Point(input_image.cols, input_image.rows));  // Vertex 2

    // Fill the triangle in the mask with white color
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));

    /// Create an output image with the same size as the input image
    cv::Mat src = cv::Mat::zeros(input_image.size(), input_image.type());
   

    // Copy pixels from the input image to the output image only where the mask is white
    input_image.copyTo(src, mask);
    return(src); 

}


cv::Mat segment_road(cv::Mat input_image)
{
    // Create a kernel that we will use to sharpen our image
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    cv::Mat imgLaplacian;
    cv::filter2D(input_image, imgLaplacian, CV_32F, kernel);
    cv::Mat sharp;
    input_image.convertTo(sharp, CV_32F);
    cv::Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    //cv::imshow( "Laplace Filtered Image", imgLaplacian );
    //cv::imshow( "New Sharped Image", imgResult );
    // Create binary image from source image
    cv::Mat bw;
    cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
    cv::threshold(bw, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    //cv::imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    cv::Mat dist;
    distanceTransform(bw, dist, cv::DIST_L1, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
    //cv::imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
    // Dilate a bit the dist image
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(dist, dist, kernel1);
    //cv::imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<cv::Point> > contours;
    cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    cv::Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    //cv::imshow("Markers", markers8u);
    // Perform the watershed algorithm
    cv::watershed(imgResult, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
   cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }

    //cv::imshow("final result", dst);
    return(dst);
} 
cv::Mat image_preprocessor::detect_road(cv::Mat input_image)
{
    
    cv::resize(input_image, input_image, cv::Size(500,500), cv::INTER_LINEAR);
    cv::Mat Original_image = input_image; // to be used to get only the road pixels

    float scale_rows = 0.45; 
    float scale_cols = 0.15 ; 

    cv::Mat cropped_image  = crop_image(input_image,scale_rows,scale_cols);
    cv::imshow("cropped image", cropped_image);

    cv::Mat segmented_image = segment_road(cropped_image); 
    cv::imshow("segmented image", segmented_image);

    

    // take a reference pixel for the road . Assumption is pixel close to bottom of image
    auto Ref = segmented_image.at<cv::Vec3b>(490,250);
    
    for (int i = 0; i < Original_image.rows; i++)
    {
        for (int j = 0; j < Original_image.cols; j++)
        {
            if (Ref !=  segmented_image.at<cv::Vec3b>(i,j))
            {
                Original_image.at<cv::Vec3b>(i,j) = (0,0,0); 
            }
        }
    }
    


    // Visualize the final image
    cv::imshow("Segmented Road", Original_image);
    cv::waitKey();



    return(input_image);
    
    
    
    
    
    
    
    
    
    /*    // Load the pre-trained DeepLabV3 model
    cv::dnn::Net net = cv::dnn::readNetFromTorch("Model/1.tflite");

    if (net.empty()) {
        std::cerr << "Error: Could not load the DeepLabV3 model." << std::endl;
        //return -1;
    }

    // Resize the image to match the input size of the DeepLabV3 model
    cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0 / 127.5, cv::Size(513, 513), cv::Scalar(127.5, 127.5, 127.5), true, true);

    // Set the input for the neural network
    net.setInput(inputBlob);

    // Forward pass to get the segmentation mask
    cv::Mat segmentationMask = net.forward();

    // Post-process the segmentation mask
    cv::Mat binaryMask;
    cv::threshold(segmentationMask, binaryMask, 15, 255, cv::THRESH_BINARY);

    // Convert the binary mask to 8-bit format for visualization
    binaryMask.convertTo(binaryMask, CV_8U);

    // Resize the binary mask to match the dimensions of the original image
    cv::resize(binaryMask, binaryMask, img.size());

    // Apply the mask to the original image
    cv::Mat segmentedRoad;
    
    // Check if the dimensions of the image and binaryMask match
    if (img.size() != segmentationMask.size()) {
        std::cout << "Error: Image and binaryMask dimensions do not match." << std::endl;
    //    return -1;
    }
 
    img.copyTo(segmentedRoad, binaryMask);

    return(segmentedRoad); 
    */

    /*
    std::cout << "Preprocessing image" << std::endl;

    if(img.empty())
    {
        std::cout << "Could not read the image: " << std::endl;
        //return NULL;
    }
    cv::Mat output;
    cv::resize(img, output, cv::Size(), scale, scale);

    int mid_img_height = (int)output.rows/2; 
    int mid_img_width = (int)output.cols/2; 

     // Create a mask (black image)
    cv::Mat mask = cv::Mat::zeros(output.size(), CV_8UC1);

    // Define the vertices of the triangle
    std::vector<cv::Point> vertices;
    vertices.push_back(cv::Point(0, output.rows));  // Vertex 1
    vertices.push_back(cv::Point(output.cols, output.rows));  // Vertex 2
    vertices.push_back(cv::Point(mid_img_width,mid_img_height));  // Vertex 3

    // Fill the triangle in the mask with white color
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));

    /// Create an output image with the same size as the input image
    cv::Mat result = cv::Mat::zeros(output.size(), output.type());

    // Copy pixels from the input image to the output image only where the mask is white
    output.copyTo(result, mask);
    */


}
