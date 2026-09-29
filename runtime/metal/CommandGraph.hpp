// Modified by meowkernels.
#pragma once

#include "MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// An ordered dispatch list for one command buffer. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&other) noexcept
      : payloads_(std::move(other.payloads_)),
        dispatches_(std::move(other.dispatches_)),
        dispatchCount_(std::exchange(other.dispatchCount_, 0)),
        payloadCount_(std::exchange(other.payloadCount_, 0)),
        streamCount_(std::exchange(other.streamCount_, 0)),
        streamed_(std::exchange(other.streamed_, 0)),
        streamSink_(std::move(other.streamSink_)) {}
  CommandGraph &operator=(CommandGraph &&other) noexcept {
    if (this != &other) {
      payloads_ = std::move(other.payloads_);
      dispatches_ = std::move(other.dispatches_);
      dispatchCount_ = std::exchange(other.dispatchCount_, 0);
      payloadCount_ = std::exchange(other.payloadCount_, 0);
      streamCount_ = std::exchange(other.streamCount_, 0);
      streamed_ = std::exchange(other.streamed_, 0);
      streamSink_ = std::move(other.streamSink_);
    }
    return *this;
  }

  // Release every buffer owner, including a partially built unused slot after
  // an allocation exception. Keep only CPU container capacities for reuse.
  void clear() noexcept {
    for (auto &dispatch : dispatches_) {
      dispatch.buffers.clear();
      dispatch.bytes.clear();
      dispatch.pipelineName.clear();
    }
    dispatchCount_ = 0;
    payloadCount_ = 0;
    streamed_ = 0;
    streamAt(0, {});
  }

  // SPLASH_STREAMED_SUBMIT: calls `sink` once with the first `count` dispatches
  // as soon as they are complete (when the next dispatch starts), so their
  // command can commit while the rest of the graph is built. 0 clears it.
  void streamAt(size_t count,
                std::function<void(std::span<const ComputeDispatch>)> sink) noexcept {
    streamCount_ = count;
    streamSink_ = std::move(sink);
  }

  void add(std::string_view pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(pipeline, std::move(buffers), groups, threads);
  }

  void add(std::string_view pipeline, std::initializer_list<MetalBuffer> buffers,
           DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(pipeline, buffers, groups, threads);
  }

  template <class Params>
  void add(std::string_view pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    addWithParams(pipeline, std::move(buffers), params, groups, threads);
  }

  template <class Params>
  void add(std::string_view pipeline, std::initializer_list<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    addWithParams(pipeline, buffers, params, groups, threads);
  }

  // SPLASH_SEAM_SIBLING: moves dispatches [first, end) so that each sits
  // directly behind its partner (partners[i] < first; one partner's siblings
  // keep their order) and marks them siblings. Dispatch objects move whole, so
  // their parameter payload pointers stay valid.
  void placeSiblings(size_t first, std::span<const size_t> partners) {
    if (first > dispatchCount_ || partners.size() != dispatchCount_ - first)
      throw std::invalid_argument("sibling placement does not cover the tail");
    std::vector<ComputeDispatch> ordered;
    ordered.reserve(dispatchCount_);
    for (size_t target = 0; target < first; ++target) {
      ordered.push_back(std::move(dispatches_[target]));
      for (size_t seam = 0; seam < partners.size(); ++seam) {
        if (partners[seam] >= first || partners[seam] < streamed_)
          throw std::invalid_argument("sibling partner is not an earlier unstreamed dispatch");
        if (partners[seam] != target) continue;
        ordered.push_back(std::move(dispatches_[first + seam]));
        ordered.back().sibling = true;
      }
    }
    for (size_t index = 0; index < ordered.size(); ++index)
      dispatches_[index] = std::move(ordered[index]);
  }

  [[nodiscard]] bool empty() const noexcept { return dispatchCount_ == 0; }
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return {dispatches_.data(), dispatchCount_};
  }

private:
  template <class Buffers, class Params>
  void addWithParams(std::string_view pipeline, Buffers buffers,
                     const Params &params, DispatchSize groups,
                     DispatchSize threads) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    if (payloadCount_ == payloads_.size())
      payloads_.emplace_back();
    auto &payload = payloads_[payloadCount_++];
    payload.resize(sizeof(Params));
    std::memcpy(payload.data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(pipeline, std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payload.data(), sizeof(Params)});
  }

  template <class Buffers>
  ComputeDispatch &push(std::string_view pipeline, Buffers buffers,
                        DispatchSize groups, DispatchSize threads) {
    if (streamCount_ && dispatchCount_ == streamCount_) {
      streamed_ = streamCount_;
      streamCount_ = 0;
      std::exchange(streamSink_, {})(dispatches());
    }
    if (dispatchCount_ == dispatches_.size()) {
      ComputeDispatch dispatch;
      dispatch.pipelineName = pipeline;
      dispatches_.push_back(std::move(dispatch));
    } else {
      dispatches_[dispatchCount_].pipelineName = pipeline;
    }
    ComputeDispatch &dispatch = dispatches_[dispatchCount_];
    dispatch.buffers.clear();
    dispatch.bytes.clear();
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.sibling = false;
    dispatch.buffers.reserve(buffers.size());
    uint32_t index = 0;
    for (auto &buffer : buffers) {
      dispatch.buffers.push_back({index++, std::move(buffer)});
    }
    ++dispatchCount_;
    return dispatch;
  }

  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
  size_t dispatchCount_ = 0;
  size_t payloadCount_ = 0;
  size_t streamCount_ = 0;
  size_t streamed_ = 0;  // dispatches already handed to the stream sink
  std::function<void(std::span<const ComputeDispatch>)> streamSink_;
};

} // namespace splash::metal
