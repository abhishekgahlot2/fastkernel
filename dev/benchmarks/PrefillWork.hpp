#pragma once

#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace splash::benchmark {

// Work expected by cold and prefix-hit benchmark requests, including the
// engine's rolling recovery points and final reusable replay state.
inline uint64_t expectedDraftContextRows(uint32_t promptTokens,
                                         uint32_t checkpointTokens,
                                         uint32_t restoredTokens = 0) {
  if (!promptTokens)
    return 0;
  const uint32_t replayBoundary =
      (promptTokens - 1) / kv::kPageTokens * kv::kPageTokens;
  std::vector<uint32_t> boundaries;
  if (checkpointTokens) {
    for (uint64_t boundary =
             (uint64_t{restoredTokens} / checkpointTokens + 1) * checkpointTokens;
         boundary < replayBoundary; boundary += checkpointTokens)
      boundaries.push_back(static_cast<uint32_t>(boundary));
  }
  if (replayBoundary > restoredTokens)
    boundaries.push_back(replayBoundary);
  return planDraftContext(
             restoredTokens, promptTokens,
             restoredTokens ? std::optional<uint32_t>(restoredTokens)
                            : std::nullopt,
             boundaries)
      .draftContextRows();
}

} // namespace splash::benchmark
