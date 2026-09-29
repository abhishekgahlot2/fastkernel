#pragma once

#include "engine/Types.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace splash::test {

class ImmediateTicket final : public ModelBatchTicket {
public:
  explicit ImmediateTicket(std::vector<ModelStepResult> results)
      : results_(std::move(results)) {}

  [[nodiscard]] bool ready() const noexcept override { return true; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return 0.0;
  }

private:
  std::vector<ModelStepResult> results_;
};

inline std::unique_ptr<ModelBatchTicket>
immediateTicket(std::vector<ModelStepResult> results,
                const std::function<void()> &completion) {
  if (completion)
    completion();
  return std::make_unique<ImmediateTicket>(std::move(results));
}

} // namespace splash::test
