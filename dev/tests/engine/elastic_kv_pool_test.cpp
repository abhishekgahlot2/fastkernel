#include "engine/KvPool.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using splash::engine::KvPool;
using splash::engine::KvBacking;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

class TestBacking final : public KvBacking {
public:
    TestBacking(uint32_t pages, uint32_t extentPages,
                uint32_t residentExtents = 0)
        : TestBacking(std::vector<uint32_t>((pages + extentPages - 1) /
                                                extentPages,
                                            extentPages),
                      residentExtents) {
        if (!pages || !extentPages || pages % extentPages) {
            throw std::invalid_argument("invalid test backing geometry");
        }
    }
    // Explicit extent sizes, e.g. a shorter trailing extent as the real
    // storage produces when the pool is not a whole number of extents.
    TestBacking(std::vector<uint32_t> extentSizes, uint32_t residentExtents)
        : resident_(extentSizes.size(), false) {
        for (uint32_t size : extentSizes) {
            if (!size) throw std::invalid_argument("empty test extent");
            for (uint32_t page = 0; page < size; ++page) {
                extentOf_.push_back(static_cast<uint32_t>(firstPage_.size()));
            }
            firstPage_.push_back(pages_);
            extentPages_.push_back(size);
            pages_ += size;
        }
        for (uint32_t index = 0;
             index < residentExtents && index < resident_.size(); ++index) {
            resident_[index] = true;
        }
    }

    uint32_t pageCount() const noexcept override { return pages_; }
    uint64_t bytesPerPage() const noexcept override { return 100; }
    bool isResident(uint32_t page) const override {
        return resident_.at(extentOf_.at(page));
    }
    splash::metal::AllocationResult ensureResident(uint32_t page) override {
        uint32_t extent = extentOf_.at(page);
        ++mappingAttempts;
        if (!admission) return false;
        if (failExtent && extent == *failExtent) return false;
        resident_.at(extent) = true;
        return true;
    }
    bool releaseBackingForPage(uint32_t page) override {
        uint32_t extent = extentOf_.at(page);
        if (!resident_.at(extent)) return false;
        resident_[extent] = false;
        ++releasedExtents;
        if (pacedReleases) ready = false;
        return true;
    }
    bool releaseReady() const noexcept override { return ready; }
    void awaitRelease() override {
        ++awaitedReleases;
        ready = true;
    }
    bool ready = true;
    bool pacedReleases = false;
    uint32_t awaitedReleases = 0;
    uint32_t extentFirstPage(uint32_t page) const override {
        return firstPage_.at(extentOf_.at(page));
    }
    uint32_t extentPageCount(uint32_t page) const override {
        return extentPages_.at(extentOf_.at(page));
    }
    bool admission = true;
    std::optional<uint32_t> failExtent;
    uint32_t mappingAttempts = 0;
    uint32_t releasedExtents = 0;

private:
    uint32_t pages_ = 0;
    std::vector<uint32_t> extentPages_;
    std::vector<uint32_t> firstPage_;
    std::vector<uint32_t> extentOf_;
    std::vector<bool> resident_;
};

void release(KvPool &pool, const std::vector<uint32_t> &pages,
             bool prefix = false) {
    for (uint32_t page : pages) pool.releasePage(page, prefix);
}

void testGrowthPacksResidentExtents() {
    TestBacking backing(12, 4, 1);
    KvPool pool(backing);
    auto pages = pool.acquirePages(5, false);
    require(pages.granted() && pages.pages.size() == 5,
            "elastic pool did not acquire requested pages");
    require(pages.pages[0] == 0 && pages.pages[3] == 3 &&
                pages.pages[4] == 4,
            "elastic pool did not fill its resident runway first");
    auto live = pool.snapshot();
    require(live.pagesResident == 8 && live.pagesActive == 5 &&
                live.pagesFreeResident == 3 &&
                live.residentBackingBytes == 800,
            "elastic growth accounting is incorrect");

    release(pool, pages.pages);
    require(pool.reclaimEmptyExtents(true) == 1,
            "reclaim did not retain exactly one warm runway");
    auto reclaimed = pool.snapshot();
    require(reclaimed.pagesResident == 4 &&
                reclaimed.reclaimableExtents == 1 &&
                backing.releasedExtents == 1,
            "empty physical extent was not returned exactly");
}

void testFailedGrowthRollsBackAtomically() {
    TestBacking backing(12, 4);
    backing.failExtent = 1;
    KvPool pool(backing);
    auto pages = pool.acquirePages(5, false);
    require(!pages.granted() &&
                pages.failure ==
                    splash::engine::KvPageAcquireFailure::PhysicalCapacity,
            "failed physical growth was not reported as physical capacity");
    auto status = pool.snapshot();
    require(status.pagesFree == 12 && status.pagesActive == 0 &&
                status.pagesPrefix == 0 && status.pagesResident == 0 &&
                backing.mappingAttempts == 2 &&
                backing.releasedExtents == 1,
            "failed physical growth leaked references or backing");
}

void testPressureReusesResidentPagesAndDeniesGrowth() {
    TestBacking backing(8, 4, 1);
    KvPool pool(backing);
    auto active = pool.acquirePages(2, false);
    require(active.granted() && active.pages.size() == 2,
            "pressure setup did not acquire active pages");
    backing.admission = false;
    auto resident = pool.acquirePages(1, false);
    require(resident.granted() && resident.pages.size() == 1 &&
                resident.pages.front() == 2,
            "critical pressure rejected an already-resident free page");
    auto denied = pool.acquirePages(2, false);
    require(!denied.granted() &&
                denied.failure ==
                    splash::engine::KvPageAcquireFailure::PhysicalCapacity,
            "critical pressure admitted a new physical extent");
    require(pool.activeReferences(active.pages[0]) == 1 &&
                pool.activeReferences(active.pages[1]) == 1 &&
                pool.activeReferences(resident.pages.front()) == 1 &&
                pool.snapshot().pagesActive == 3,
            "critical pressure corrupted existing active references");
    release(pool, resident.pages);
    release(pool, active.pages);
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesResident == 0,
            "pressure cleanup did not reclaim the empty extent");
}

void testReleasesArePacedBehindTheBacking() {
    TestBacking backing(16, 4, 4);
    KvPool pool(backing);
    auto pages = pool.acquirePages(16, false);
    require(pages.granted() && pool.snapshot().pagesResident == 16,
            "paced-release setup did not acquire every page");
    release(pool, pages.pages);
    require(pool.reclaimableExtentCount() == 4 && pool.releaseReady(),
            "four empty resident extents were not reclaimable");

    // A release still being torn down blocks further unmaps without
    // touching the remaining empty extents.
    backing.ready = false;
    require(pool.reclaimEmptyExtents(false) == 0 && !pool.releaseReady() &&
                pool.reclaimableExtentCount() == 4 &&
                backing.releasedExtents == 0,
            "pool released backing while the previous release was in flight");
    // Deferral never escalates to a wait on the serving path, however many
    // passes find the release still in flight.
    for (int pass = 0; pass < 3; ++pass)
      require(pool.reclaimEmptyExtents(false) == 0,
              "pool released backing behind an in-flight release");
    require(backing.awaitedReleases == 0,
            "serving-path reclaim waited on an in-flight release");
    backing.ready = true;
    require(pool.reclaimEmptyExtents(false, 2) == 2 &&
                pool.reclaimableExtentCount() == 2 &&
                pool.snapshot().pagesResident == 8,
            "extent release limit was not honored");

    // A paced backing becomes busy after each release: one extent per pass,
    // and awaitRelease() makes the next pass possible.
    backing.pacedReleases = true;
    require(pool.reclaimEmptyExtents(false) == 1 && !pool.releaseReady() &&
                pool.reclaimableExtentCount() == 1,
            "paced backing did not stop after one release");
    require(pool.reclaimEmptyExtents(false) == 0,
            "pool ignored an in-flight paced release");
    pool.awaitRelease();
    require(backing.awaitedReleases == 1 && pool.releaseReady() &&
                pool.reclaimEmptyExtents(true) == 0 &&
                pool.reclaimableExtentCount() == 1,
            "runway was not retained by the paced final pass");
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesResident == 0 &&
                backing.releasedExtents == 4,
            "final paced release did not return the last extent");
}

void testFullestExtentFillsFirstSoColdExtentsDrain() {
    TestBacking backing(12, 4, 3);
    KvPool pool(backing);
    auto all = pool.acquirePages(12, false);
    require(all.granted() && all.pages.size() == 12,
            "fill setup did not acquire every page");
    // Leave extent 0 with three holes, extent 1 with one and extent 2 with two.
    release(pool, {0, 1, 2, 5, 8, 9});
    require(pool.snapshot().pagesFreeResident == 6 &&
                pool.reclaimableExtentCount() == 0,
            "partial release accounting is incorrect");

    // New pages come from the fullest extents; the coldest keeps its holes.
    auto refill = pool.acquirePages(2, false);
    require(refill.granted() && refill.pages.size() == 2 &&
                refill.pages[0] == 5 &&
                (refill.pages[1] == 8 || refill.pages[1] == 9),
            "refill did not take pages from the fullest extents first");
    auto again = pool.acquirePages(1, false);
    require(again.granted() && again.pages.front() / 4 == 2,
            "allocation did not continue with the fullest extent");

    // Its last page going cold empties the extent so it can be unmapped.
    release(pool, {3});
    require(pool.reclaimableExtentCount() == 1 &&
                pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesResident == 8 &&
                backing.releasedExtents == 1,
            "drained extent was not released");
}

void testShorterTrailingExtentIsNotPreferredForBeingSmall() {
    // The real storage ends with a shorter extent. Having fewer free pages
    // only because it is small must not rank it as the fullest.
    TestBacking backing({4, 4, 2}, 3);
    KvPool pool(backing);
    auto all = pool.acquirePages(10, false);
    require(all.granted() && all.pages.size() == 10,
            "trailing extent setup did not acquire every page");

    // Partially used: extent 0 keeps two live pages, the tail keeps one.
    release(pool, {0, 1, 8});
    auto page = pool.acquirePages(1, false);
    require(page.granted() && page.pages.front() < 4,
            "partial trailing extent was refilled ahead of a fuller extent");
    release(pool, {9});
    require(pool.reclaimableExtentCount() == 1,
            "trailing extent did not drain after its last page was freed");

    // Empty: the tail's two free pages still lose to extent 0's one live page.
    release(pool, {page.pages.front(), 2});
    auto next = pool.acquirePages(1, false);
    require(next.granted() && next.pages.front() < 4,
            "empty trailing extent was refilled ahead of a partial extent");
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesResident == 8,
            "empty trailing extent was not released");
}

void testPrefixAndActiveReferencesShareResidency() {
    TestBacking backing(8, 4, 1);
    KvPool pool(backing);
    auto active = pool.acquirePages(1, false);
    require(active.granted(), "shared reference setup failed");
    pool.retainPage(active.pages.front(), true);
    pool.releasePage(active.pages.front(), false);
    require(pool.reclaimEmptyExtents(false) == 0 &&
                pool.snapshot().pagesPrefix == 1,
            "prefix-owned extent was reclaimed while live");
    pool.releasePage(active.pages.front(), true);
    require(pool.reclaimEmptyExtents(false) == 1,
            "last prefix release did not make extent reclaimable");
}

void testReleaseProgressIncludesAllocationRollback() {
    TestBacking backing(8, 4);
    KvPool pool(backing);
    backing.failExtent = 1;
    backing.pacedReleases = true;
    const auto before = pool.releaseGeneration();
    auto allocation = pool.acquirePages(8, false);
    require(!allocation.granted() && pool.releaseGeneration() != before &&
                !pool.releaseReady(),
            "allocation rollback did not record its pending release");
    const auto pending = pool.releaseGeneration();
    backing.ready = true;
    require(pool.releaseReady() && pool.releaseGeneration() != pending,
            "completion did not invalidate the prior allocation decision");
    const auto complete = pool.releaseGeneration();
    require(pool.releaseReady() && pool.releaseGeneration() == complete,
            "unchanged completion would keep retrying a hard capacity denial");
    require(pool.snapshot().pagesActive == 0 && pool.snapshot().pagesResident == 0,
            "rollback retained page ownership or residency");
}

}  // namespace

int main() {
    try {
        testReleaseProgressIncludesAllocationRollback();
        testGrowthPacksResidentExtents();
        testFailedGrowthRollsBackAtomically();
        testPressureReusesResidentPagesAndDeniesGrowth();
        testReleasesArePacedBehindTheBacking();
        testFullestExtentFillsFirstSoColdExtentsDrain();
        testShorterTrailingExtentIsNotPreferredForBeingSmall();
        testPrefixAndActiveReferencesShareResidency();
        std::cout << "elastic KV pool tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "elastic KV pool test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
