#include "io.h"

namespace Rain::IO {
std::vector<char> ReadFile(const std::string& filename){
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if(!file.is_open()) {
    return std::vector<char>{};
  }
  size_t file_size = (size_t) file.tellg();
  std::vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();
  return buffer;
};
};