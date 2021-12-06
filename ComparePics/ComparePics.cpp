
#include <iostream>

#include "utils_filesystem.h"
#include "compute.h"

/// pictures are scaled before calculations. Can be reduced for speed up!!! set it to 35, for example
//#define scaleOrig 100
#define scaleOrig 35

/// different compute methods. For more possible options plese have a look in compute.h
#define selectedDescr eFeatureDescriptors::akaze
#define selectedAlg eFeatureAlg::knn

/// change to true to visualize calculation results
#define visualizeResults true
/// if showResults is true, the value below inidcates how long the good resuls are on a screen
#define visualizeDelayForGood 1000
/// result is scaled after original pictures scaled also
#define visualizeScaledResult 100

/// result of identity percentage calculation between pictures
/// map1: key is filePath, map2: key is the other filePath; value is the calculated identity percentage between key1 and key2
std::map<std::string, std::map<std::string, float> > result_Identities; 

int main(int argc, char* argv[])
{
    LOG_NOTHING();

    /// get input from user - threshold value
    std::cout << "Enter threshold value (0..100):" << std::endl;
    int threshold(0);
    std::cin >> threshold;    
    threshold = max(0, min(threshold, 100));

    /// get vector of file names in the working dir
    auto files = get_filenames(std::filesystem::current_path());

    float angle(0.0);
    float identity(0.0);
    /// iterate over files and compare every with the others
    for (auto it = files.begin(); it!=files.end(); it++)
    {
        Mat frame = imread(*it, IMREAD_COLOR);        
        if (frame.empty())
            continue;

        if (scaleOrig != 100)
        {
            compute::resize(frame, scaleOrig);
        }
        
        auto it2 = it;
        it2++;
        Mat pattern;
        for (; it2 != files.end(); it2++)
        {
            pattern = imread(*it2, IMREAD_COLOR);
            if (pattern.empty())
                continue;            

            if (scaleOrig != 100)
            {              
                compute::resize(pattern, scaleOrig);
            }
            
            if (!compute::FindAngle(frame, pattern, selectedDescr, selectedAlg, angle, identity, visualizeResults))
                continue;            

            int idi = static_cast<int>(ceil(identity));
            if (idi >= threshold)
            {
                if (visualizeResults)
                {
                    compute::showResult(visualizeScaledResult);
                    compute::wait(visualizeDelayForGood);
                }
                result_Identities[*it][*it2] = identity;
            }
        }
    }

    for (const auto& it : result_Identities)
    {        
        if (it.second.size() > 0)
        {
            for (const auto& it2 : it.second)
            {
                std::cout << std::filesystem::path(it.first).filename() << ", " << 
                    std::filesystem::path(it2.first).filename() << ", identity: " << static_cast<int>(ceil(it2.second))  << "%" 
                    //<< ", rotated:" << angle << " degrees" 
                    << std::endl;
            }
        }
        else 
        {
            std::cout << it.first;
        }

        std::cout << std::endl;
    }

    compute::release();

    system("pause");
    return 0;
}
