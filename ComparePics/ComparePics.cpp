
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/xfeatures2d.hpp> //!
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>

#include "compute.h"

#define scale 35
#define selectedDescr eFeatureDescriptors::akaze
#define selectedAlg eFeatureAlg::knn

int main(int argc, char* argv[])
{
    //std::string pathFrame = "../pics/Big1.png";
    //std::string pathPattern = "../pics/Big1_90grad.png";

    std::string pathFrame = "../pics/test_pic_1.png";
    std::string pathPattern = "../pics/test_pic_2.png";
    //std::string pathPattern = "../pics/test_pic_3.png";
    
    Mat frame =imread(pathFrame, IMREAD_COLOR);
    Mat pattern = imread(pathPattern, IMREAD_COLOR);

    cv::resize(frame, frame, cv::Size(), 1.0 * scale / 100.0, 1.0 * scale / 100.0, INTER_CUBIC);
    cv::resize(pattern, pattern, cv::Size(), 1.0 * scale / 100.0, 1.0 * scale / 100.0, INTER_CUBIC);

    //imshow("image", frame);
    //imshow("pattern", pattern);

    float angle(0.0);
    float percent(0.0);
    compute::FindAngle(frame, pattern, selectedDescr, selectedAlg, angle, percent);

    waitKey(10000);

    return 0;
}
