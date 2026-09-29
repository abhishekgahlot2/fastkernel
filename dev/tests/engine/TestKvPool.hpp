#pragma once

#include "engine/KvPool.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace splash::test {

class TestKvBacking final : public engine::KvBacking {
public:
    TestKvBacking(uint32_t pages, uint64_t bytesPerPage)
        : pageCount_(pages), bytesPerPage_(bytesPerPage), resident_(pages, true) {
        if (!pages || !bytesPerPage) {
            throw std::invalid_argument("invalid test KV backing");
        }
    }

    uint32_t pageCount() const noexcept override { return pageCount_; }
    uint64_t bytesPerPage() const noexcept override { return bytesPerPage_; }
    bool isResident(uint32_t page) const override {
        return resident_.at(page);
    }
    splash::metal::AllocationResult ensureResident(uint32_t page) override {
        resident_.at(page) = true;
        return true;
    }
    bool releaseBackingForPage(uint32_t page) override {
        bool wasResident = resident_.at(page);
        resident_[page] = false;
        return wasResident;
    }
    uint32_t extentFirstPage(uint32_t page) const override {
        if (page >= pageCount_) throw std::out_of_range("invalid test page");
        return page;
    }
    uint32_t extentPageCount(uint32_t page) const override {
        if (page >= pageCount_) throw std::out_of_range("invalid test page");
        return 1;
    }
private:
    uint32_t pageCount_ = 0;
    uint64_t bytesPerPage_ = 0;
    std::vector<bool> resident_;
};

}  // namespace splash::test
