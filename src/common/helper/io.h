#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace Rain {
namespace IO {
std::vector<char> ReadFile(const std::string& filename);
};
};  // namespace Rain