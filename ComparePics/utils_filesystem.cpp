
#include "utils_filesystem.h"

std::vector<std::string> get_filenames(std::filesystem::path path)
{
    namespace stdfs = std::filesystem;

    std::vector<std::string> filenames;

    const stdfs::directory_iterator end{};

    for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
    {
        if (stdfs::is_regular_file(*iter)) // comment out if all names (names of directories tc.) are required
            filenames.push_back(iter->path().string());
    }

    return filenames;
}