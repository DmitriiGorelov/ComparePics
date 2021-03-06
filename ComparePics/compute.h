#pragma once

#include <vector>
#include <map>
#include <string>

//#include <opencv2/type>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc.hpp>

#include <opencv2/xfeatures2d.hpp> //! opencv_contrib
#include <opencv2/xfeatures2d/nonfree.hpp> //! opencv_contrib

#include <opencv2/core/utils/logger.hpp>
#define LOG_NOTHING() cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
#define LOG_ERRORS() cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
#define LOG_VERBOSE() cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_VERBOSE);

/// you will need this library
///https://github.com/opencv/opencv_contrib

using namespace cv;
using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SURF;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;

using cv::Mat;
using cv::Point2f;
using cv::KeyPoint;
using cv::Scalar;
using cv::Ptr;

using cv::FastFeatureDetector;
using cv::SimpleBlobDetector;

using cv::DMatch;
using cv::BFMatcher;
using cv::DrawMatchesFlags;
using cv::Feature2D;
using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::KAZE;

namespace eFeatureDescriptors {
    typedef unsigned short T;
    enum E :T {
        akaze = 0,
        surf = 1,
        sift = 2,
        sink = 3,
        orb = 4,
        brisk = 5,
        kaze = 6,
        blobfreak = 7,
        fastfreak = 8,
        fastdaisy = 9,
        blobbrief = 10,
        fastbrief = 11,
    };
}

namespace eFeatureAlg {
    typedef unsigned short T;
    enum E :T {
        bf = 0,
        knn = 1,
    };
}

class compute {
    static Mat res;
private:
	static void match(eFeatureAlg::E type, Mat& desc1, Mat& desc2, std::vector<DMatch>& matches);
	static void detect_and_compute(std::string type, const Mat& img, std::vector<KeyPoint>& kpts, Mat& desc);
public:

    static void resize(Mat& frame, float scale);

	static bool FindAngle(const Mat& frame, const cv::Mat& pattern, eFeatureDescriptors::E desc_type, eFeatureAlg::E match_type, float& angle, float& identity, bool visualize);
    static void showResult(float scale);
    static void wait(int timeout);
    static void release();
};
