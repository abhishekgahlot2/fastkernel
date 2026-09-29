#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace splash::model {

inline constexpr size_t kPromptLookupHistory = 65536;
// Match-length buckets, longest first. minMatch (default 16) cuts the list,
// so the default keeps the original {32, 16} behavior.
inline constexpr std::array<size_t, 6> kPromptLookupLengths{32, 16, 12, 8, 6, 4};

// History includes the pending anchor. Every proposed token must already
// exist in that history; prefer the longest match bucket, then its latest
// occurrence. One backward scan measures each candidate's match length.
template <size_t ProposalTokens = 7>
inline std::optional<std::array<uint32_t, ProposalTokens>>
promptLookup(std::span<const uint32_t> history, size_t minMatch = 16) {
  static_assert(ProposalTokens == 7 || ProposalTokens == 15 || ProposalTokens == 31);
  history = history.last(std::min(history.size(), kPromptLookupHistory));
  minMatch = std::max<size_t>(minMatch, 1);
  const size_t n = history.size();
  if (n < minMatch + ProposalTokens)
    return std::nullopt;
  // latest[b]: latest proposal start whose match falls in bucket b (0 = none).
  std::array<size_t, kPromptLookupLengths.size()> latest{};
  for (size_t start = n - ProposalTokens; start >= minMatch && !latest[0]; --start) {
    size_t match = 0;
    while (match < kPromptLookupLengths[0] && match < start &&
           history[start - 1 - match] == history[n - 1 - match])
      ++match;
    for (size_t bucket = 0; bucket < kPromptLookupLengths.size() &&
                            kPromptLookupLengths[bucket] >= minMatch; ++bucket) {
      if (match >= kPromptLookupLengths[bucket]) {
        if (!latest[bucket]) latest[bucket] = start;
        break;
      }
    }
  }
  for (size_t start : latest) {
    if (!start) continue;
    std::array<uint32_t, ProposalTokens> proposal;
    std::copy_n(history.begin() + start, proposal.size(), proposal.begin());
    return proposal;
  }
  return std::nullopt;
}

// SPLASH_LOOKUP_ADAPTIVE: a request's match threshold steps one bucket shorter
// after a strong lookup cycle (>= 7 tokens retained, about 2x the drafter) and
// one bucket longer after a weak one (<= 3 retained), never below floor.
inline size_t adaptLookupMatch(size_t current, uint32_t retained, size_t floor) {
  const auto &lengths = kPromptLookupLengths;
  size_t index = static_cast<size_t>(
      std::find(lengths.begin(), lengths.end(), current) - lengths.begin());
  if (index == lengths.size()) return 16;
  if (retained >= 7 && index + 1 < lengths.size() && lengths[index + 1] >= floor)
    ++index;
  else if (retained <= 3 && index > 0)
    --index;
  return lengths[index];
}

} // namespace splash::model
