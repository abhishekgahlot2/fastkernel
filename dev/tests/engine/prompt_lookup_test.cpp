#include "model/PromptLookup.hpp"

#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using splash::model::promptLookup;

// The original two-length scan ({32, 16}), kept as the reference for the default.
template <size_t K>
std::optional<std::array<uint32_t, K>> referenceLookup(std::span<const uint32_t> history) {
  history = history.last(std::min(history.size(), splash::model::kPromptLookupHistory));
  for (size_t length : {32U, 16U}) {
    if (history.size() < length + K) continue;
    const auto suffix = history.last(length);
    size_t position = history.size() - length - K;
    do {
      if (std::equal(suffix.begin(), suffix.end(), history.begin() + position)) {
        std::array<uint32_t, K> proposal;
        std::copy_n(history.begin() + position + length, K, proposal.begin());
        return proposal;
      }
    } while (position-- != 0);
  }
  return std::nullopt;
}

int main() {
  std::vector<uint32_t> prefix(32);
  std::iota(prefix.begin(), prefix.end(), 100);
  const std::array<uint32_t, 7> first{1, 2, 3, 4, 5, 6, 7};
  const std::array<uint32_t, 7> latest{8, 9, 10, 11, 12, 13, 14};
  assert(!promptLookup({}));
  assert(!promptLookup(prefix));
  auto history = prefix;
  history.insert(history.end(), first.begin(), first.end());
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(promptLookup(history) == first);
  history.insert(history.end(), latest.begin(), latest.end());
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(promptLookup(history) == latest);

  // A newer short match must not override the older 32-token match.
  history.insert(history.end(), first.begin(), first.end());
  history.insert(history.end(), prefix.begin() + 16, prefix.end());
  history.insert(history.end(), latest.begin(), latest.end());
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(promptLookup(history) == first);

  std::vector<uint32_t> shortHistory(prefix.begin() + 16, prefix.end());
  shortHistory.insert(shortHistory.end(), latest.begin(), latest.end());
  shortHistory.insert(shortHistory.end(), prefix.begin() + 16, prefix.end());
  assert(promptLookup(shortHistory) == latest);
  shortHistory.back() = 999;
  assert(!promptLookup(shortHistory));

  // A match with fewer than seven known following tokens must fall back.
  std::vector<uint32_t> incomplete(22, 42);
  assert(!promptLookup(incomplete));
  incomplete.push_back(42);
  assert(promptLookup(incomplete) == (std::array<uint32_t, 7>{42,42,42,42,42,42,42}));
  // Dropped history is not a source of proposals.
  history = prefix;
  history.insert(history.end(), first.begin(), first.end());
  history.insert(history.end(), splash::model::kPromptLookupHistory, 999);
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(!promptLookup(history));
  std::array<uint32_t, 15> wide;
  std::iota(wide.begin(), wide.end(), 1000);
  history = prefix;
  history.insert(history.end(), wide.begin(), wide.end());
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(promptLookup<15>(history) == wide);
  assert(!promptLookup<15>(std::vector<uint32_t>(30, 42)));
  assert(promptLookup<15>(std::vector<uint32_t>(31, 42)));
  // SPLASH_WIDE_LOOKUP32: 31 proposals need 31 continuation tokens after the match.
  std::array<uint32_t, 31> wide32;
  std::iota(wide32.begin(), wide32.end(), 2000);
  history = prefix;
  history.insert(history.end(), wide32.begin(), wide32.end());
  history.insert(history.end(), prefix.begin(), prefix.end());
  assert(promptLookup<31>(history) == wide32);
  assert(!promptLookup<31>(std::vector<uint32_t>(46, 42)));
  assert(promptLookup<31>(std::vector<uint32_t>(47, 42)));
  // Default minMatch reproduces the original scan on random low-entropy histories.
  std::mt19937 rng(20260927);
  for (int trial = 0; trial < 20000; ++trial) {
    std::vector<uint32_t> random(rng() % 400);
    const uint32_t alphabet = 1 + rng() % 3;
    for (auto &token : random) token = rng() % alphabet;
    assert(promptLookup(random) == referenceLookup<7>(random));
    assert(promptLookup<15>(random) == referenceLookup<15>(random));
  }
  // Lower thresholds: an 8-token repeat proposes only when minMatch <= 8.
  std::vector<uint32_t> eight(prefix.begin() + 24, prefix.end());
  eight.insert(eight.end(), latest.begin(), latest.end());
  eight.insert(eight.end(), prefix.begin() + 24, prefix.end());
  assert(!promptLookup(eight));
  assert(!promptLookup(eight, 12));
  assert(promptLookup(eight, 8) == latest);
  // A longer bucket still wins over a newer shorter match at a low threshold.
  auto mixed = prefix;
  mixed.insert(mixed.end(), first.begin(), first.end());
  mixed.insert(mixed.end(), prefix.begin() + 24, prefix.end());
  mixed.insert(mixed.end(), latest.begin(), latest.end());
  mixed.insert(mixed.end(), prefix.begin(), prefix.end());
  assert(promptLookup(mixed, 4) == first);
  // Adaptive threshold: earn shorter matches after strong cycles, back off after weak ones.
  using splash::model::adaptLookupMatch;
  assert(adaptLookupMatch(16, 8, 8) == 12);
  assert(adaptLookupMatch(12, 7, 8) == 8);
  assert(adaptLookupMatch(8, 16, 8) == 8);   // floor
  assert(adaptLookupMatch(8, 2, 8) == 12);
  assert(adaptLookupMatch(16, 5, 8) == 16);  // neutral
  assert(adaptLookupMatch(32, 1, 8) == 32);  // ceiling
  assert(adaptLookupMatch(16, 3, 8) == 32);
  assert(adaptLookupMatch(7, 8, 8) == 16);   // unknown value resets to the default
  std::cout << "prompt lookup: PASS\n";
}
