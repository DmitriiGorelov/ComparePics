#include "compute.h"

//#include <math.h>
#include <minmax.h>

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning (disable : 4244)
#endif

using namespace std;

map<int, string> featureDescriptors = {
            {eFeatureDescriptors::akaze, "akaze"},
            {eFeatureDescriptors::surf, "surf"},
            {eFeatureDescriptors::sift, "sift"},
            {eFeatureDescriptors::sink, "sink"},
            {eFeatureDescriptors::orb,"orb"},
            {eFeatureDescriptors::brisk,"brisk"},
            {eFeatureDescriptors::kaze,"kaze"},
            {eFeatureDescriptors::blobfreak, "blobfreak"},
            {eFeatureDescriptors::fastfreak, "fastfreak"},
            {eFeatureDescriptors::fastdaisy, "fastdaisy"},
            {eFeatureDescriptors::blobbrief, "blobbrief"},
            {eFeatureDescriptors::fastbrief, "fastbrief"}
};

map<int, string> featureAlg = {
            {eFeatureAlg::bf, "fb"}, {eFeatureAlg::knn, "knn"}
};

#define M_PI       3.14159265358979323846   // pi
#define kDistanceCoef 4.0
#define kMaxMatchingSize 500

#define WFindAngleResult "Find Angle Result"

int m_aHomoMethods[3] = { 4,8,16 };
int m_iHomoMethod = 0;

void compute::match(eFeatureAlg::E type, Mat& desc1, Mat& desc2, std::vector<DMatch>& matches)
{
    matches.clear();
    switch (type)
    {
    case eFeatureAlg::bf:
    {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
        break;
    }
    case eFeatureAlg::knn:
    {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (size_t i = 0; i < vmatches.size(); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
        break;
    }
    default:
        break;
    }

    if (matches.size() < 2)
    {
        return;
    }
    
    std::sort(matches.begin(), matches.end()); // needful in case of any below uncommented
    
    // this does not work if distance of 1st element is zero!
    /*while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }*/

    // minimize size of matches for time saving, ignore on desktop PCs
    if (matches.size() > kMaxMatchingSize)
        matches.assign(matches.begin(), matches.begin()+ kMaxMatchingSize);    
}

void compute::detect_and_compute(std::string type, const Mat& img, std::vector<KeyPoint>& kpts, Mat& desc)
{
    if (type.find("fast") == 0) {
        type = type.substr(4);
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }
    if (type.find("blob") == 0) {
        type = type.substr(4);
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "surf") {
        Ptr<Feature2D> surf = SURF::create(800.0);
        surf->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "sift") {
        Ptr<Feature2D> sift = SIFT::create();
        sift->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "orb") {
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "brisk") {
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "kaze") {
        Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "akaze") {
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "freak") {
        Ptr<FREAK> freak = FREAK::create();
        freak->compute(img, kpts, desc);
    }
    if (type == "daisy") {
        Ptr<DAISY> daisy = DAISY::create();
        daisy->compute(img, kpts, desc);
    }
    if (type == "brief") {
        Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
        brief->compute(img, kpts, desc);
    }
}

cv::String ToString(float value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void getComponents(cv::Mat const normalised_homography, float& theta)
{
    double a = normalised_homography.at<double>(0, 0);
    double b = normalised_homography.at<double>(0, 1);

    theta = atan2(b, a) * (180.0 / M_PI);
}

void drawInclination(float theta, float identity)
{
    String Stheta = ToString(theta); // save grad to string before transforming
    theta = -theta * M_PI / 180.0;

    int tam = 200;
    Mat canvas = Mat::zeros(tam, tam, CV_8UC3);
    

    line(canvas, Point(0, tam / 2), Point(tam, tam / 2), Scalar(255, 255, 255));
    line(canvas, Point(tam / 2, tam / 2), Point(tam / 2 + tam * cos(theta), tam / 2 + tam * sin(theta)), Scalar(0, 255, 0), 2);
    
    putText(canvas, Stheta, Point(tam - 90, tam), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));

    String Sidentity = ToString(identity) + "%";
    putText(canvas, Sidentity, Point(tam - 190, tam), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));

    imshow("Inclunacion", canvas);

}

void compute::resize(Mat& frame, float scale)
{
    if (scale!=100)
        cv::resize(frame, frame, cv::Size(), 1.0 * scale / 100.0, 1.0 * scale / 100.0, INTER_CUBIC);
}

bool compute::FindAngle(const Mat& frame, const cv::Mat& pattern, eFeatureDescriptors::E desc_type, eFeatureAlg::E match_type, float& angle, float& identity, bool visualize)
{    
    /*if (pattern.channels() != 1)
    {
        cvtColor(pattern, pattern, cv::COLOR_RGB2GRAY);
    }

    if (frame.channels() != 1)
    {
        cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
    }*/

    std::vector<KeyPoint> kpts1;
    std::vector<KeyPoint> kpts2;

    Mat desc1;
    Mat desc2;

    std::vector<DMatch> good_matches;

    detect_and_compute(featureDescriptors[desc_type], pattern, kpts1, desc1);
    detect_and_compute(featureDescriptors[desc_type], frame, kpts2, desc2);

    match(match_type, desc1, desc2, good_matches);

    std::vector<char> match_mask(good_matches.size(), 1);
    //findKeyPointsHomography(kpts1, kpts2, good_matches, match_mask);
    
    if (visualize)
    {        
        cv::drawMatches(pattern, kpts1, frame, kpts2, good_matches, res, Scalar::all(-1),
            Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    if (good_matches.size() >= 4)
    {
        std::vector<Point2f> obj;
        std::vector<Point2f> obj_corners(4);
        std::vector<Point2f> scene;
        std::vector<Point2f> scene_corners(4);
        Mat H;

        //Get the corners from the pattern
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f(pattern.cols, 0);
        obj_corners[2] = Point2f(pattern.cols, pattern.rows);
        obj_corners[3] = Point2f(0, pattern.rows);

        for (int i = 0; i < (int)good_matches.size(); i++)
        {
            //Get the keypoints from the good matches
            obj.push_back(kpts1[good_matches[i].queryIdx].pt);
            scene.push_back(kpts2[good_matches[i].trainIdx].pt);
        }

        H = findHomography(obj, scene, m_aHomoMethods[m_iHomoMethod], 4, match_mask);

        if (H.empty())
        {
            desc1.release();
            desc2.release();
            H.release();
            return false;
        }
        perspectiveTransform(obj_corners, scene_corners, H);

        scene_corners[0].x = min(max(scene_corners[0].x, 0), frame.cols);
        scene_corners[1].x = min(max(scene_corners[1].x, 0), frame.cols);
        scene_corners[2].x = min(max(scene_corners[2].x, 0), frame.cols);
        scene_corners[3].x = min(max(scene_corners[3].x, 0), frame.cols);

        scene_corners[0].y = min(max(scene_corners[0].y, 0), frame.rows);
        scene_corners[1].y = min(max(scene_corners[1].y, 0), frame.rows);
        scene_corners[2].y = min(max(scene_corners[2].y, 0), frame.rows);
        scene_corners[3].y = min(max(scene_corners[3].y, 0), frame.rows);

        auto P1 = scene_corners[0] + Point2f(pattern.cols, 0);
        auto P2 = scene_corners[1] + Point2f(pattern.cols, 0);
        auto P3 = scene_corners[2] + Point2f(pattern.cols, 0);
        auto P4 = scene_corners[3] + Point2f(pattern.cols, 0);
        if (visualize)
        {
            //Draw lines between the corners (the mapped pattern in the scene image )
            line(res, P1, P2, Scalar(0, 255, 0), 4);
            line(res, P2, P3, Scalar(0, 255, 0), 4);
            line(res, P3, P4, Scalar(0, 255, 0), 4);
            line(res, P4, P1, Scalar(0, 255, 0), 4);
        }

        float S1 = 0.5f * abs((P1.x - P2.x) * (P1.y + P2.y) + (P2.x - P3.x) * (P2.y + P3.y) + (P3.x - P4.x) * (P3.y + P4.y) + (P4.x - P1.x) * (P4.y + P1.y));
        float S0 = frame.cols * frame.rows;
        if (S0 > 0.0)
            identity = min(100.0, 100.0 * S1 / S0);
        else
            identity = 0.0;
        
        getComponents(H, angle);

        //if (visualize)
        //    drawInclination(angle, identity);

        desc1.release();
        desc2.release();
        H.release();
    }

    return true;
    //if (show)
    //    imshow(WFindAngleResult, res);
    //cv::waitKey(0);
}

Mat compute::res;
void compute::showResult(float scale)
{
    if (!res.empty() && scale > 0.0)
    {
        cv::resize(res, res, cv::Size(), scale / 100.0, scale / 100.0, INTER_CUBIC);
        imshow(WFindAngleResult, res);
    }
}

void compute::wait(int timeout)
{
    waitKey(timeout);
}

void compute::release()
{
    cv::destroyAllWindows();
    res.release();
}
