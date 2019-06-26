#ifndef JSON_H_
#define JSON_H_

#ifdef OPENMIND
#include <experimental/string_view>

namespace std {

using string_view = experimental::string_view;

} // namespace std;

#endif

#include <nlohmann/json.hpp>

#endif // JSON_H_
