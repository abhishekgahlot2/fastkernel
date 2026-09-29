#pragma once

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <time.h>

namespace splash::metal {

// SPLASH_HOST_PHASE_LOG=1 (diagnostic, default off): host phase marks on the
// clock of MTLCommandBuffer GPUStartTime/GPUEndTime (mach absolute time in
// seconds, as CACurrentMediaTime), so they line up with SPLASH_GPU_GAP_LOG.
// Marks are buffered per thread and printed by hostPhaseFlush(), which the
// engine calls once the next command is committed: the printing stays out of
// the host gap being measured.
inline double hostPhaseClock() noexcept {
  return static_cast<double>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) * 1e-9;
}

inline bool hostPhaseLog() noexcept {
  static const bool on = std::getenv("SPLASH_HOST_PHASE_LOG") != nullptr;
  return on;
}

struct HostPhaseMarks {
  struct Mark {
    const char *name;
    double seconds;
    long long a, b;
  };
  std::array<Mark, 256> marks;
  size_t count = 0;
  size_t dropped = 0;
};
inline thread_local HostPhaseMarks hostPhaseMarks;

inline void hostPhase(const char *name, long long a = -1, long long b = -1) noexcept {
  if (!hostPhaseLog())
    return;
  HostPhaseMarks &buffer = hostPhaseMarks;
  if (buffer.count == buffer.marks.size()) {
    ++buffer.dropped;
    return;
  }
  buffer.marks[buffer.count++] = {name, hostPhaseClock(), a, b};
}

inline void hostPhaseFlush() noexcept {
  if (!hostPhaseLog())
    return;
  HostPhaseMarks &buffer = hostPhaseMarks;
  for (size_t index = 0; index < buffer.count; ++index) {
    const HostPhaseMarks::Mark &mark = buffer.marks[index];
    std::fprintf(stderr, "host_phase %s %.9f %lld %lld\n", mark.name,
                 mark.seconds, mark.a, mark.b);
  }
  if (buffer.dropped)
    std::fprintf(stderr, "host_phase dropped %zu\n", buffer.dropped);
  buffer.count = 0;
  buffer.dropped = 0;
}

} // namespace splash::metal
