#pragma once

#include <cstdlib>
#include <string_view>

namespace splash::metal {

// Default-on engine switch (the exact, measured-faster paths): on when the
// variable is unset or equals `on`; any other value turns it off, so
// SPLASH_<NAME>=0 restores the path it replaced.
[[nodiscard]] inline bool envSwitch(const char *name, std::string_view on = "1") noexcept {
  const char *value = std::getenv(name);
  return !value || std::string_view(value) == on;
}

} // namespace splash::metal
