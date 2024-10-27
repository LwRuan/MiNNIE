// ported from Xingyu Ni's code
#pragma once

#include <spdlog/fmt/fmt.h>

#include <any>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace Rain {

class ArgDataBase {
 protected:
  const std::type_info &_type;
  std::string _name;
  char _flag;
  std::string _desc;
  bool _mandatory;
  bool _set = false;

 public:
  ArgDataBase(const std::type_info &type, const std::string &name,
              const char flag, const std::string &desc, const bool mandatory)
      : _type(type),
        _name(name),
        _flag(flag),
        _desc(desc.empty() ? "<no description available>" : desc),
        _mandatory(mandatory) {}

  virtual ~ArgDataBase() = default;

  const std::type_info &type() const { return _type; }
  const std::string &name() const { return _name; }
  char flag() const { return _flag; }
  const std::string &desc() const { return _desc; }
  bool isMandatory() const { return _mandatory; }
  bool isSet() const { return _set; }
  virtual std::string getShortDesc() const = 0;
  virtual bool parseValue(const std::string &str) = 0;
  virtual std::any getValue() const = 0;
};

template <typename ArgType>
class ArgData final : public ArgDataBase {
 protected:
  ArgType _setValue;
  ArgType _defaultValue;

 public:
  ArgData(const std::string &name, const char flag, const std::string desc)
      : ArgDataBase(typeid(ArgType), name, flag, desc, true),
        _setValue(ArgType()),
        _defaultValue(ArgType()) {}

  ArgData(const std::string &name, const char flag, const std::string desc,
          const ArgType &defaultValue)
      : ArgDataBase(typeid(ArgType), name, flag, desc, false),
        _setValue(ArgType()),
        _defaultValue(defaultValue) {}

  virtual ~ArgData() = default;

  virtual std::string getShortDesc() const override {
    return _mandatory ? fmt::format("--{}", _name)
                      : fmt::format("--{}={}", _name, _defaultValue);
  }
  virtual bool parseValue(const std::string &str) override {
    return _set = true, bool(std::istringstream(str) >> _setValue);
  }
  virtual std::any getValue() const override {
    return _set ? _setValue : (_mandatory ? std::any() : _defaultValue);
  }
};

class ArgsParser final {
 protected:
  std::string _progName;
  std::vector<std::unique_ptr<ArgDataBase>> _args;
  std::vector<std::string> _extraArgs;

 public:
  ArgsParser() = default;
  ArgsParser(const ArgsParser &rhs) = delete;
  ArgsParser operator=(const ArgsParser &rhs) = delete;
  virtual ~ArgsParser() = default;

  template <typename ArgType>
  void addArgument(const std::string &name, const char flag = 0,
                   const std::string &desc = "") {
    _args.push_back(std::make_unique<ArgData<ArgType>>(name, flag, desc));
  }

  template <typename ArgType>
  void addArgument(const std::string &name, const char flag,
                   const std::string &desc, const ArgType &defaultValue) {
    _args.push_back(
        std::make_unique<ArgData<ArgType>>(name, flag, desc, defaultValue));
  }

  std::string generateUsage() const;

  void parse(const int argc, const char *const argv[]);
  void parse(const char *cmdLine);

  std::any getValueByName(const std::string &name) const {
    return findArgByName(name)->getValue();
  }
  std::any getValueByFlag(char flag) const {
    return findArgByFlag(flag)->getValue();
  }

 protected:
  ArgDataBase *findArgByName(const std::string &name) const;
  ArgDataBase *findArgByFlag(const char flag) const;
  void reportError(const std::string &msg) const;
};

}  // namespace Rain
