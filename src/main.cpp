#include <spdlog/spdlog.h>

#include <iostream>

#include "cudasim/explicit.h"
#include "engine.h"
#include "helper/argsparser.h"

using namespace Rain;

inline std::unique_ptr<ArgsParser> BuildArgsParser() {
  auto parser = std::make_unique<ArgsParser>();
  parser->addArgument<std::string>("config", 'c', "configuration file",
                                   "../configs/wave.yaml");
  return parser;
}

int main(int argc, char *argv[]) {
  spdlog::set_pattern("[%^%l%$] %v");
#ifdef NDEBUG
  spdlog::set_level(spdlog::level::info);
#else
  spdlog::set_level(spdlog::level::debug);
#endif

  auto parser = BuildArgsParser();
  parser->parse(argc, argv);
  const auto config_file =
      std::any_cast<std::string>(parser->getValueByName("config"));
  Engine engine;
  engine.Init(config_file);
  engine.MainLoop();
  engine.CleanUp();
}