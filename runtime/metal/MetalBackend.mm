// Modified by meowkernels.
#import "MetalBackend.hpp"
#include "CommandWatchdog.hpp"
#include "DeviceQueries.hpp"
#include "EnvSwitch.hpp"
#include "HostPhase.hpp"
#include "MetalEvent.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>
#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <unistd.h>

namespace splash::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value),
                                 kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
            KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}

NSUInteger checkedNSUInteger(uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<NSUInteger>::max()) {
        throw MetalBackendError(std::string(field) + " exceeds NSUInteger");
    }
    return static_cast<NSUInteger>(value);
}

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(checkedNSUInteger(size.x, field),
                       checkedNSUInteger(size.y, field),
                       checkedNSUInteger(size.z, field));
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

constexpr uint64_t kPlacementSparsePageBytes = MetalBackend::kPlacementSparsePageBytes;
constexpr MTLSparsePageSize kPlacementSparsePageSize = MTLSparsePageSize64;
constexpr NSUInteger kSparseUnmapTimeoutMilliseconds = 30000;
constexpr NSUInteger kSparseMapTimeoutMilliseconds = 30000;

MTLSparsePageSize metalSparsePageSize(uint64_t bytes) {
    if (bytes != kPlacementSparsePageBytes) {
        throw MetalBackendError(
            "placement-sparse page size must be exactly 64 KiB");
    }
    return kPlacementSparsePageSize;
}

double steadySeconds() noexcept {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not_enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

NSString *checkedNSString(std::string_view value, std::string_view field) {
    NSString *result = [[NSString alloc]
        initWithBytes:value.data()
        length:value.size()
        encoding:NSUTF8StringEncoding];
    if (!result) {
        throw MetalBackendError(std::string(field) + " is not UTF-8");
    }
    return result;
}

struct StringViewHash final {
    using is_transparent = void;

    size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    size_t operator()(const std::string &value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

}  // namespace

struct AllocationAccounting {
    std::atomic<uint64_t> allocatedBytes{0};
    std::atomic<uint64_t> peakAllocatedBytes{0};
    std::atomic<uint64_t> sparseVirtualBytes{0};
    std::atomic<uint64_t> sparseResidentBytes{0};
    std::atomic<uint64_t> peakSparseResidentBytes{0};
    std::atomic<uint64_t> residentBytes{0};
    std::atomic<uint64_t> peakResidentBytes{0};

    void addResident(uint64_t bytes) noexcept {
        raisePeak(peakResidentBytes,
                  residentBytes.fetch_add(bytes, std::memory_order_relaxed) +
                      bytes);
    }
};

struct MetalAllocation {
    // Own the host mapping for our views as well as the Metal deallocator.
    // Validation wrappers may not retain the supplied deallocator block.
    std::shared_ptr<void> externalOwner;
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;
    uint64_t sparseVirtualBytes = 0;
    bool placementSparse = false;
    BufferStorage storage = BufferStorage::Shared;

    ~MetalAllocation() {
        if (accounting && bytes) {
            accounting->allocatedBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
            accounting->residentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
        if (accounting && sparseVirtualBytes) {
            accounting->sparseVirtualBytes.fetch_sub(
                sparseVirtualBytes, std::memory_order_relaxed);
        }
    }
};

struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct SparseHeap::Impl {
    __strong id<MTLHeap> heap = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;

    ~Impl() {
        if (accounting && bytes) {
            accounting->sparseResidentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
            accounting->residentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
    }
};

struct BackendAsyncState {
    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    uint64_t activeSequence = 0;
    size_t activeDispatchCount = 0;
    __weak id<MTLCommandBuffer> activeCommand = nil;
    std::function<void(id<MTLCommandBuffer>)> activeCompletion;
    CommandWatchdog commandWatchdog;
    std::stop_source stopping;
    std::atomic<uint64_t> mapWaitEvent{0};
    std::atomic<double> mapWaitStarted{0.0};
    std::atomic<double> lastMapWaitSeconds{0.0};
    std::atomic<double> maxMapWaitSeconds{0.0};

    uint64_t sampleDeviceMemory() const noexcept {
        if (!device) return 0;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
        return current;
    }

    void ensureHealthy() const {
        if (healthy.load(std::memory_order_acquire)) return;
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void markUnhealthy(std::string reason) {
        {
            std::lock_guard lock(healthMutex);
            if (healthReason.empty()) healthReason = std::move(reason);
        }
        healthy.store(false, std::memory_order_release);
    }

    uint64_t beginSubmission(size_t dispatchCount) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping.stop_requested())
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        if (nextSequence == std::numeric_limits<uint64_t>::max()) {
            throw MetalBackendError("Metal command sequence exhausted");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    bool commitSubmission(uint64_t sequence, id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        if (stopping.stop_requested()) return false;
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, steadySeconds());
        [command commit];
        return true;
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
        if (activeSequence == sequence) {
            activeSequence = 0;
            activeCommand = nil;
            activeCompletion = {};
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    void checkCommandHealth() {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (commandWatchdog.expired(steadySeconds())) {
                command = activeCommand;
                const auto status = command ? command.status
                                            : MTLCommandBufferStatusNotEnqueued;
                // Recover terminal results even if the driver has not delivered
                // its callback. Finish outside the gate: it takes the ticket lock.
                if (command && (status == MTLCommandBufferStatusCompleted ||
                                status == MTLCommandBufferStatusError)) {
                    complete = activeCompletion;
                } else {
                    std::ostringstream message;
                    message << "Metal command completion timed out after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status=" << (command ? commandStatusName(status)
                                                       : "unavailable")
                            << ", dispatches=" << activeDispatchCount << ')';
                    markUnhealthy(message.str());
                }
            }
        }
        if (complete) complete(command);
        ensureHealthy();
    }

    [[nodiscard]] bool hasActiveSubmission() const noexcept {
        std::lock_guard lock(gateMutex);
        return activeSequence != 0;
    }
};

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    std::chrono::steady_clock::time_point wallStart;
    uint64_t sparseEventValue = 0;
    // Chunked submission: the early-committed head command; the ticket
    // completes on the tail, which waits on the head through a fence.
    __strong id<MTLCommandBuffer> head = nil;
    bool gated = false;  // the head raises the chain event (SPLASH_GRAMMAR_CHAIN)
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        const double callback = hostPhaseLog() ? hostPhaseClock() : 0.0;
        auto wallEnd = std::chrono::steady_clock::now();
        CommandTiming timing;
        const double gpuStart = head && head.GPUStartTime > 0.0
                                    ? head.GPUStartTime
                                    : command.GPUStartTime;
        timing.gpuSeconds = command.GPUEndTime - gpuStart;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();
        // Diagnostic: SPLASH_GPU_GAP_LOG=1 prints each command's GPU span.
        static const bool gapLog = std::getenv("SPLASH_GPU_GAP_LOG") != nullptr;
        if (gapLog)
          std::fprintf(stderr, "gpu_cmd %llu %.9f %.9f %.9f\n",
                       static_cast<unsigned long long>(sequence), gpuStart,
                       command.GPUEndTime, timing.wallSeconds);
        // A chunked (gpu_head) or gated (gpu_gate) command's head, and where
        // its tail started.
        if (gapLog && head)
          std::fprintf(stderr, gated ? "gpu_gate %llu %.9f %.9f %.9f\n"
                                     : "gpu_head %llu %.9f %.9f %.9f\n",
                       static_cast<unsigned long long>(sequence), head.GPUStartTime,
                       head.GPUEndTime, command.GPUStartTime);

        std::string error;
        if (head && head.status == MTLCommandBufferStatusError) {
            std::ostringstream message;
            message << "Metal command " << sequence << " head chunk failed";
            if (head.error) message << ": " << errorDescription(head.error);
            error = message.str();
        } else if (command.status != MTLCommandBufferStatusCompleted) {
            std::ostringstream message;
            message << "Metal command " << sequence
                    << " failed (sparse event " << sparseEventValue << ')';
            if (command.error) {
                message << ": " << errorDescription(command.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
        if (callback > 0.0)
            std::fprintf(stderr, "host_phase cb %.9f %llu -1\n", callback,
                         static_cast<unsigned long long>(sequence));
    }

    void finish(CommandTiming result, std::string failure = {}) {
        CommandCompletion notify;
        {
            std::lock_guard lock(mutex);
            // Host recovery, late callbacks, and discarded commands all share
            // this completion path; only the first result may publish or notify.
            if (completed) return;
            backend->completeSubmission(sequence);
            if (!failure.empty()) backend->markUnhealthy(failure);
            timing = result;
            error = std::move(failure);
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify(sequence);
            } catch (...) {
                backend->markUnhealthy(
                    "Metal completion callback threw an exception");
            }
        }
        condition.notify_all();
    }

    void release() noexcept {
        bool shouldRelease = false;
        {
            std::lock_guard lock(mutex);
            if (!released) {
                released = true;
                retainedAllocations.clear();
                head = nil;
                shouldRelease = true;
            }
        }
        if (shouldRelease && backend) {
            // Refresh admission telemetry on the consuming thread after GPU
            // completion, before allowing the next submission.
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    void abandon() noexcept {
        {
            std::unique_lock lock(mutex);
            condition.wait(lock, [this] { return completed; });
        }
        release();
    }
};

namespace {

struct PreparedDispatch {
    const ComputeDispatch *source = nullptr;
    MTLSize groups{};
    MTLSize threads{};
    uint64_t threadCount = 0;
    __strong id<MTLComputePipelineState> pipeline = nil;
};

}  // namespace

struct MetalBackend::Impl {
    std::function<void()> operationGuard;

    bool dispatchProfiling = false;
    // SPLASH_CHUNKED_SUBMIT=N (default 48; 0 = off): commit the first N
    // dispatches as their own command so the GPU starts while the rest is encoded.
    // Scheduling only: tokens identical 18/18, ms/cycle -0.19.
    uint32_t chunkHeadDispatches = [] {
        const char *value = std::getenv("SPLASH_CHUNKED_SUBMIT");
        return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 48u;
    }();
    __strong id<MTLFence> chunkFence = nil;
    // SPLASH_SEAM_SIBLING: orders a submission's serial part before its concurrent tail.
    __strong id<MTLFence> siblingFence = nil;
    // SPLASH_GRAMMAR_CHAIN: orders a gated command's post-wait dispatches after
    // its middle ones; the chain event carries the host/GPU steps.
    __strong id<MTLFence> gateFence = nil;
    __strong id<MTLSharedEvent> chainEvent = nil;
    std::atomic<uint64_t> chainHighest{0};
    // SPLASH_DRAFT_AHEAD trailing command buffer (guarded by commandMutex):
    // trailFenceIn orders it after its command, trailFenceOut orders the next
    // submission after it.
    __strong id<MTLFence> trailFenceIn = nil;
    __strong id<MTLFence> trailFenceOut = nil;
    __strong id<MTLCommandBuffer> trailing = nil;
    // Signalled by the trailing completion handler after it drops the
    // buffer retention, so a completed wait also means the bytes are free.
    __strong dispatch_semaphore_t trailingDone = nil;
    bool trailingUnordered = false;
    std::vector<DispatchTiming> dispatchProfile;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTL4CommandQueue> sparseQueue = nil;
    __strong id<MTLSharedEvent> sparseEvent = nil;
    __strong id<MTLLibrary> library = nil;
    __strong NSMutableDictionary<NSString *, id<MTLComputePipelineState>>
        *pipelines = nil;
    // The dictionary remains the canonical strong owner. This byte-keyed index
    // owns only validated UTF-8 names and borrows values for the same Impl
    // lifetime. Every pipeline() caller holds commandMutex.
    std::unordered_map<std::string, void *, StringViewHash, std::equal_to<>>
        pipelineIndex;

    DeviceCapabilities capabilities;
    std::array<uint8_t, 32> metallibSha256{};
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState =
        std::make_shared<BackendAsyncState>();
    mutable std::mutex commandMutex;
    uint64_t nextSparseEventValue = 0;
    uint64_t pendingSparseEventValue = 0;

    // The one outstanding asynchronous unmap; guarded by commandMutex. Its
    // heap stays alive, and counted resident, until the queue signals.
    struct PendingSparseUnmap {
        uint64_t eventValue = 0;
        SparseHeap heap;
        std::chrono::steady_clock::time_point issued;
    };
    std::optional<PendingSparseUnmap> pendingUnmap;
    std::atomic<uint64_t> pendingUnmapCount{0};
    std::atomic<uint64_t> completedUnmaps{0};
    std::atomic<double> lastUnmapSeconds{0.0};
    std::atomic<double> maxUnmapSeconds{0.0};
    std::atomic<double> pendingUnmapIssuedSeconds{0.0};

    ~Impl() {
        // Teardown must not wait for a stalled mapping queue. Keep its backing
        // alive until the driver acknowledges the pending unmap instead.
        if (pendingUnmap && sparseEvent.signaledValue < pendingUnmap->eventValue) {
            auto retainedHeap = std::make_shared<SparseHeap>(std::move(pendingUnmap->heap));
            id<MTL4CommandQueue> retainedQueue = sparseQueue;
            id<MTLSharedEvent> retainedEvent = sparseEvent;
            [sparseEvent notifyListener:[MTLSharedEventListener sharedListener]
                atValue:pendingUnmap->eventValue block:^(id<MTLSharedEvent>, uint64_t) {
                    (void)retainedHeap;
                    (void)retainedQueue;
                    (void)retainedEvent;
                }];
        }
    }

    // Requires commandMutex. Releases the heap of a completed unmap.
    bool reapSparseUnmapsLocked() noexcept {
        if (!pendingUnmap) return false;
        if (sparseEvent.signaledValue < pendingUnmap->eventValue) return false;
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pendingUnmap->issued).count();
        lastUnmapSeconds.store(seconds, std::memory_order_relaxed);
        raisePeak(maxUnmapSeconds, seconds);
        completedUnmaps.fetch_add(1, std::memory_order_relaxed);
        pendingUnmap.reset();
        pendingUnmapCount.store(0, std::memory_order_release);
        sampleDeviceMemory();
        return true;
    }

    // Requires commandMutex. Blocks until the outstanding unmap completes.
    void awaitSparseUnmapLocked() {
        if (!pendingUnmap) return;
        if (![sparseEvent waitUntilSignaledValue:pendingUnmap->eventValue
                                       timeoutMS:kSparseUnmapTimeoutMilliseconds]) {
            std::ostringstream details;
            details << "sparse unmapping timed out: event="
                    << pendingUnmap->eventValue
                    << " signaled=" << sparseEvent.signaledValue
                    << " pending_map=" << pendingSparseEventValue
                    << " waited_ms=" << kSparseUnmapTimeoutMilliseconds;
            std::string message = details.str();
            markUnhealthy(message);
            throw MetalBackendError(message);
        }
        static_cast<void>(reapSparseUnmapsLocked());
    }

    uint64_t sampleDeviceMemory() const noexcept {
        return asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    MetalBuffer wrap(std::shared_ptr<MetalAllocation> allocation) {
        auto result = std::make_shared<MetalBuffer::Impl>();
        result->lengthBytes = allocation->buffer.length;
        result->allocation = std::move(allocation);
        return MetalBuffer(std::move(result));
    }

    MetalBuffer registerBuffer(id<MTLBuffer> buffer, BufferStorage storage,
                               std::shared_ptr<void> externalOwner = {}) {
        auto allocation = std::make_shared<MetalAllocation>();
        allocation->externalOwner = std::move(externalOwner);
        allocation->buffer = buffer;
        allocation->accounting = accounting;
        allocation->bytes = buffer.allocatedSize;
        allocation->storage = storage;
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      allocation->bytes, std::memory_order_relaxed) +
                      allocation->bytes);
        accounting->addResident(allocation->bytes);
        sampleDeviceMemory();
        return wrap(std::move(allocation));
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        const auto indexed = pipelineIndex.find(name);
        if (indexed != pipelineIndex.end()) {
            return (__bridge id<MTLComputePipelineState>)indexed->second;
        }
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLComputePipelineState> cached = [pipelines objectForKey:key];
        if (cached) {
            pipelineIndex.emplace(std::string(name), (__bridge void *)cached);
            return cached;
        }

        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        [pipelines setObject:result forKey:key];
        pipelineIndex.emplace(std::string(name), (__bridge void *)result);
        sampleDeviceMemory();
        return result;
    }

    // Validates a dispatch list before anything is encoded.
    static std::vector<PreparedDispatch> prepareDispatches(
        std::span<const ComputeDispatch> list, const AllocationAccounting *accounting) {
        std::vector<PreparedDispatch> prepared;
        prepared.reserve(list.size());
        for (const ComputeDispatch &dispatch : list) {
            PreparedDispatch item;
            item.source = &dispatch;
            item.groups = metalSize(dispatch.threadgroups, "threadgroups");
            item.threads = metalSize(
                dispatch.threadsPerThreadgroup, "threadsPerThreadgroup");
            if (multiplyOverflows(dispatch.threadsPerThreadgroup.x,
                                  dispatch.threadsPerThreadgroup.y) ||
                multiplyOverflows(dispatch.threadsPerThreadgroup.x *
                                      dispatch.threadsPerThreadgroup.y,
                                  dispatch.threadsPerThreadgroup.z)) {
                throw MetalBackendError("threadsPerThreadgroup size overflows");
            }
            item.threadCount = dispatch.threadsPerThreadgroup.x *
                dispatch.threadsPerThreadgroup.y *
                dispatch.threadsPerThreadgroup.z;

            uint64_t indexMask = 0;
            std::unordered_set<uint32_t> highIndices;
            const auto insertIndex = [&](uint32_t index) {
                if (index >= 64) return highIndices.insert(index).second;
                const uint64_t bit = uint64_t{1} << index;
                const bool inserted = !(indexMask & bit);
                indexMask |= bit;
                return inserted;
            };
            for (const BufferBinding &binding : dispatch.buffers) {
                if (!binding.buffer.impl_ || !binding.buffer.impl_->allocation) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' contains an empty buffer at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                if (binding.buffer.impl_->allocation->accounting.get() !=
                    accounting) {
                    throw MetalBackendError(
                        "compute dispatch buffer belongs to another backend");
                }
                if (!insertIndex(binding.index)) {
                    throw MetalBackendError("duplicate compute binding index");
                }
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                if (!binding.data || !binding.sizeBytes) {
                    throw MetalBackendError("compute byte binding is empty");
                }
                checkedNSUInteger(binding.sizeBytes, "byte binding size");
                if (!insertIndex(binding.index)) {
                    throw MetalBackendError("duplicate compute binding index");
                }
            }
            prepared.push_back(item);
        }
        return prepared;
    }

    static void encodeDispatches(id<MTLComputeCommandEncoder> encoder,
                                 const std::vector<PreparedDispatch> &list,
                                 size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                const PreparedDispatch &item = list[index];
                const ComputeDispatch &dispatch = *item.source;
                [encoder setComputePipelineState:item.pipeline];
                // CommandGraph emits a dense prefix; custom layouts retain
                // scalar binding. The capacity bounds stack storage, not the API.
                constexpr size_t kBatchCapacity = 32;
                const size_t count = dispatch.buffers.size();
                bool dense = count > 1 && count <= kBatchCapacity;
                if (dense) {
                    for (size_t index = 0; index < count; ++index)
                        dense &= dispatch.buffers[index].index == index;
                }
                if (dense) {
                    id<MTLBuffer> __unsafe_unretained buffers[kBatchCapacity];
                    NSUInteger offsets[kBatchCapacity];
                    for (size_t index = 0; index < count; ++index) {
                        const MetalBuffer::Impl &buffer =
                            *dispatch.buffers[index].buffer.impl_;
                        buffers[index] = buffer.allocation->buffer;
                        offsets[index] = checkedNSUInteger(buffer.offsetBytes,
                                                           "buffer offset");
                    }
                    [encoder setBuffers:buffers offsets:offsets
                              withRange:NSMakeRange(0, count)];
                } else {
                    for (const BufferBinding &binding : dispatch.buffers) {
                        const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                        [encoder setBuffer:buffer.allocation->buffer
                                    offset:checkedNSUInteger(buffer.offsetBytes,
                                                             "buffer offset")
                                   atIndex:binding.index];
                    }
                }
                for (const BytesBinding &binding : dispatch.bytes) {
                    [encoder setBytes:binding.data
                               length:checkedNSUInteger(binding.sizeBytes,
                                                        "byte binding size")
                              atIndex:binding.index];
                }
                [encoder dispatchThreadgroups:item.groups
                         threadsPerThreadgroup:item.threads];
            }
    }

    // Requires commandMutex. Resolves the pipelines of list[begin, end).
    void resolvePipelines(std::vector<PreparedDispatch> &list, size_t begin = 0) {
        // SPLASH_SPLIT4_FOOTER (exact scheduling, default on): the two M8 N32 split4 entries
        // run as their _ftr twins, which skip each threadgroup's last-tile scratch-reuse barrier. Read at every
        // submission, so one binary serves both arms of an in-process A/B. Lockstep -0.154 ms/step, oracle identical.
        const bool footer = envSwitch("SPLASH_SPLIT4_FOOTER");
        // SPLASH_SPLIT4_HOIST selects the metadata-hoisted explicit-FMA M8 bodies (stock's FMA order).
        // Default on; read per submission and compose independently with footer. Lockstep -0.312 ms/step,
        // oracle identical. Byte identity is proven with this Mac's GPU compiler; re-check per chip.
        const bool hoist = envSwitch("SPLASH_SPLIT4_HOIST");
        const char *split4Suffix = hoist ? (footer ? "_hoist_ftr" : "_hoist") : "_ftr";
        for (size_t index = begin; index < list.size(); ++index) {
            PreparedDispatch &item = list[index];
            const std::string &name = item.source->pipelineName;
            item.pipeline = pipeline(
                (footer || hoist) &&
                        (name == "decode_linear_q4_n32_split4_precomputed_sums" ||
                         name == "decode_linear_q4_n32_split4_precomputed_sums_residual")
                    ? name + split4Suffix : name);
            if (item.threadCount >
                item.pipeline.maxTotalThreadsPerThreadgroup) {
                throw MetalBackendError(
                    "threadsPerThreadgroup exceeds pipeline capability");
            }
        }
    }

    // SPLASH_STREAMED_SUBMIT (guarded by commandMutex): the committed head of
    // the next submission, streamed while its graph was still being built.
    struct StreamedHead {
        std::shared_ptr<CommandTicket::State> ticket;
        __strong id<MTLCommandBuffer> head = nil;
        size_t count = 0;
        bool gated = false;
        std::unordered_set<const MetalAllocation *> retained;
        std::chrono::steady_clock::time_point wallStart;
    };
    std::optional<StreamedHead> streamed;
};

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ && impl_->allocation && impl_->allocation->buffer;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

bool MetalBuffer::overlaps(const MetalBuffer &other) const noexcept {
    return impl_ && other.impl_ && impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes < other.impl_->offsetBytes + other.impl_->lengthBytes &&
           other.impl_->offsetBytes < impl_->offsetBytes + impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ && impl_->allocation ? impl_->allocation->storage
                                      : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || !impl_->allocation ||
        impl_->allocation->storage != BufferStorage::Shared) {
        return nullptr;
    }
    void *base = impl_->allocation->buffer.contents;
    if (!base) return nullptr;
    return static_cast<uint8_t *>(base) + impl_->offsetBytes;
}

SparseHeap::SparseHeap() = default;
SparseHeap::~SparseHeap() = default;
SparseHeap::SparseHeap(SparseHeap &&) noexcept = default;
SparseHeap &SparseHeap::operator=(SparseHeap &&) noexcept = default;

SparseHeap::SparseHeap(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SparseHeap::operator bool() const noexcept {
    return impl_ && impl_->heap;
}

uint64_t SparseHeap::sizeBytes() const noexcept {
    return impl_ ? impl_->bytes : 0;
}

CommandTicket::CommandTicket() = default;

CommandTicket::CommandTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CommandTicket::~CommandTicket() {
    if (state_) state_->abandon();
}

CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;

CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
    if (this == &other) return *this;
    if (state_) state_->abandon();
    state_ = std::move(other.state_);
    return *this;
}

CommandTicket::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

uint64_t CommandTicket::sequence() const noexcept {
    return state_ ? state_->sequence : 0;
}

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    CommandTiming timing;
    std::string error;
    {
        std::unique_lock lock(state_->mutex);
        state_->condition.wait(lock, [this] { return state_->completed; });
        timing = state_->timing;
        error = state_->error;
    }
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath, double commandTimeoutSeconds)
    : impl_(std::make_unique<Impl>()) {
    impl_->asyncState->commandWatchdog = CommandWatchdog(commandTimeoutSeconds);
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        const NSOperatingSystemVersion os =
            NSProcessInfo.processInfo.operatingSystemVersion;
        const auto component = [](NSInteger value) {
            return value > 0 ? static_cast<uint32_t>(value) : 0U;
        };
        impl_->capabilities.macosMajor = component(os.majorVersion);
        impl_->capabilities.macosMinor = component(os.minorVersion);
        impl_->capabilities.macosPatch = component(os.patchVersion);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(
                "Splash requires macOS " +
                std::to_string(DeviceCapabilities::kMinimumMacosMajor) + '.' +
                std::to_string(DeviceCapabilities::kMinimumMacosMinor) +
                " or newer; this Mac runs macOS " +
                impl_->capabilities.macosVersion());
        }
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw MetalBackendError("Metal device unavailable");
        }
        impl_->asyncState->device = impl_->device;
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }

        NSString *path = checkedNSString(metallibPath, "metallib path");
        NSError *error = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                 options:0
                                                   error:&error];
        if (!fileData) {
            throw MetalBackendError(
                "unable to read metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        if (!fileData.length ||
            fileData.length > std::numeric_limits<CC_LONG>::max()) {
            throw MetalBackendError("metallib is empty or too large to hash: " +
                                    metallibPath);
        }
        // DEFAULT copies into immutable dispatch-owned storage. Hash the same
        // contiguous data passed to Metal, never a second read of the path.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (!data)
            throw MetalBackendError("unable to copy metallib data: " + metallibPath);
        const void *bytes = nullptr;
        size_t byteCount = 0;
        dispatch_data_t mapped = dispatch_data_create_map(data, &bytes, &byteCount);
        if (!mapped || !bytes || byteCount != fileData.length ||
            !CC_SHA256(bytes, static_cast<CC_LONG>(byteCount),
                       impl_->metallibSha256.data())) {
            throw MetalBackendError("unable to hash metallib data: " + metallibPath);
        }
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:mapped error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        impl_->pipelines = [NSMutableDictionary dictionary];
        if (!impl_->pipelines) {
            throw MetalBackendError("unable to create Metal pipeline cache");
        }
        impl_->sampleDeviceMemory();

        DeviceCapabilities &capabilities = impl_->capabilities;
        capabilities.deviceName = stringFromNSString(impl_->device.name);
        capabilities.gpuCoreCount = gpuCoreCountForDevice(impl_->device.registryID);
        for (uint32_t family = 10; family >= 7; --family) {
            if ([impl_->device supportsFamily:
                    static_cast<MTLGPUFamily>(1000 + family)]) {
                capabilities.appleGpuFamily = family;
                break;
            }
        }
        capabilities.physicalMemoryBytes =
            NSProcessInfo.processInfo.physicalMemory;
        capabilities.recommendedMaxWorkingSetBytes =
            impl_->device.recommendedMaxWorkingSetSize;
        capabilities.maxBufferLengthBytes = impl_->device.maxBufferLength;
        capabilities.maxThreadgroupMemoryBytes =
            impl_->device.maxThreadgroupMemoryLength;
        MTLSize maximumThreads = impl_->device.maxThreadsPerThreadgroup;
        capabilities.maxThreadgroupWidth = maximumThreads.width;
        capabilities.hasUnifiedMemory = impl_->device.hasUnifiedMemory;

        // Query sparse support and exercise the private-buffer/placement-heap ABI.
        if (@available(macOS 26.4, *)) {
            if (queryPlacementSparseSupport(impl_->device)) {
                impl_->sparseQueue = [impl_->device newMTL4CommandQueue];
                impl_->sparseEvent = [impl_->device newSharedEvent];
                if (!impl_->sparseQueue || !impl_->sparseEvent) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its queue or event");
                }

                id<MTLBuffer> canaryBuffer = [impl_->device
                    newBufferWithLength:kPlacementSparsePageBytes
                    options:MTLResourceStorageModePrivate
                    placementSparsePageSize:kPlacementSparsePageSize];
                MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
                if (!descriptor) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its heap descriptor");
                }
                descriptor.type = MTLHeapTypePlacement;
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.size = kPlacementSparsePageBytes;
                descriptor.maxCompatiblePlacementSparsePageSize =
                    kPlacementSparsePageSize;
                id<MTLHeap> canaryHeap =
                    [impl_->device newHeapWithDescriptor:descriptor];
                if (!canaryBuffer || !canaryHeap) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its buffer or heap");
                }
                MTLSharedEventListener *listener =
                    [MTLSharedEventListener sharedListener];
                if (!listener) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its completion listener");
                }
                MTL4UpdateSparseBufferMappingOperation operation{};
                operation.mode = MTLSparseTextureMappingModeMap;
                operation.bufferRange = NSMakeRange(0, 1);
                operation.heapOffset = 0;
                [impl_->sparseQueue updateBufferMappings:canaryBuffer
                                                   heap:canaryHeap
                                             operations:&operation
                                                  count:1];
                [impl_->sparseQueue signalEvent:impl_->sparseEvent value:1];
                BOOL mapped = [impl_->sparseEvent
                    waitUntilSignaledValue:1 timeoutMS:5000];

                operation.mode = MTLSparseTextureMappingModeUnmap;
                [impl_->sparseQueue updateBufferMappings:canaryBuffer
                                                   heap:nil
                                             operations:&operation
                                                  count:1];
                [impl_->sparseQueue signalEvent:impl_->sparseEvent value:2];
                BOOL unmapped = [impl_->sparseEvent
                    waitUntilSignaledValue:2 timeoutMS:5000];
                if (!unmapped) {
                    // Only an unfinished probe needs asynchronous ownership.
                    id<MTL4CommandQueue> probeQueue = impl_->sparseQueue;
                    id<MTLSharedEvent> probeEvent = impl_->sparseEvent;
                    [impl_->sparseEvent notifyListener:listener atValue:2
                        block:^(id<MTLSharedEvent>, uint64_t) {
                            (void)canaryBuffer;
                            (void)canaryHeap;
                            (void)probeQueue;
                            (void)probeEvent;
                        }];
                }
                if (!mapped || !unmapped) {
                    throw MetalBackendError(
                        std::string("placement-sparse probe timed out after 5000 ms waiting for ") +
                        (!mapped ? "mapping" : "unmapping") +
                        " (last signaled event=" +
                        std::to_string(impl_->sparseEvent.signaledValue) + ')');
                }
                capabilities.supportsPlacementSparse = true;
                impl_->nextSparseEventValue = 2;
            }
        }
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() { stop(); }

void MetalBackend::stop() noexcept {
    if (!impl_) return;
    {
        std::lock_guard lock(impl_->asyncState->gateMutex);
        impl_->asyncState->stopping.request_stop();
    }
    signalChain(impl_->chainHighest.load());
}
MetalBackend::MetalBackend(MetalBackend &&) noexcept = default;
MetalBackend &MetalBackend::operator=(MetalBackend &&other) noexcept {
    if (this != &other) {
        stop();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

const std::array<uint8_t, 32> &MetalBackend::metallibSha256() const noexcept {
    return impl_->metallibSha256;
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes,
                                         BufferStorage storage,
                                         std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }

    MTLResourceOptions options = storage == BufferStorage::Shared
        ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(bytes, "buffer size")
        options:options];
    if (!buffer) throw MetalAllocationError("Metal buffer allocation failed");
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, storage);
}

MetalBuffer MetalBackend::allocatePlacementSparseBuffer(
    uint64_t virtualBytes, uint64_t sparsePageBytes, std::string_view label) {
    checkOperation();
    const MTLSparsePageSize pageSize = metalSparsePageSize(sparsePageBytes);
    if (!impl_->capabilities.supportsPlacementSparse) {
        throw MetalBackendError("placement-sparse Metal is unavailable");
    }
    if (!virtualBytes || virtualBytes % sparsePageBytes) {
        throw MetalBackendError(
            "placement-sparse buffer size must be tile-aligned");
    }
    if (virtualBytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError(
            "placement-sparse buffer exceeds maxBufferLength");
    }

    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(virtualBytes, "sparse buffer size")
        options:MTLResourceStorageModePrivate
        placementSparsePageSize:pageSize];
    if (!buffer) {
        throw MetalAllocationError(
            "placement-sparse buffer creation failed");
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");

    auto allocation = std::make_shared<MetalAllocation>();
    allocation->buffer = buffer;
    allocation->accounting = impl_->accounting;
    allocation->sparseVirtualBytes = virtualBytes;
    allocation->placementSparse = true;
    allocation->storage = BufferStorage::Private;
    impl_->accounting->sparseVirtualBytes.fetch_add(
        virtualBytes, std::memory_order_relaxed);
    impl_->sampleDeviceMemory();
    return impl_->wrap(std::move(allocation));
}

SparseHeap MetalBackend::allocatePlacementHeap(
    uint64_t physicalBytes, uint64_t sparsePageBytes, std::string_view label) {
    impl_->ensureHealthy();
    const MTLSparsePageSize pageSize = metalSparsePageSize(sparsePageBytes);
    if (!impl_->capabilities.supportsPlacementSparse) {
        throw MetalBackendError("placement-sparse Metal is unavailable");
    }
    if (!physicalBytes || physicalBytes % sparsePageBytes) {
        throw MetalBackendError(
            "placement heap size must be tile-aligned");
    }

    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = checkedNSUInteger(physicalBytes, "placement heap size");
    descriptor.maxCompatiblePlacementSparsePageSize = pageSize;
    id<MTLHeap> heap = [impl_->device newHeapWithDescriptor:descriptor];
    if (!heap) {
        throw MetalAllocationError("placement heap allocation failed");
    }
    if (!label.empty()) heap.label = checkedNSString(label, "heap label");

    auto result = std::make_shared<SparseHeap::Impl>();
    result->heap = heap;
    const uint64_t heapBytes = static_cast<uint64_t>(heap.size);
    if (heapBytes < physicalBytes || heapBytes % sparsePageBytes) {
        throw MetalBackendError("placement heap has unexpected size");
    }
    result->accounting = impl_->accounting;
    result->bytes = heapBytes;
    raisePeak(impl_->accounting->peakSparseResidentBytes,
              impl_->accounting->sparseResidentBytes.fetch_add(
                  result->bytes, std::memory_order_relaxed) + result->bytes);
    impl_->accounting->addResident(result->bytes);
    impl_->sampleDeviceMemory();
    return SparseHeap(std::move(result));
}

void MetalBackend::mapSparse(
    const SparseHeap &heap, std::span<const SparseMapping> mappings) {
    if (!heap.impl_ || !heap.impl_->heap ||
        heap.impl_->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("placement heap belongs to another backend");
    }
    if (mappings.empty()) {
        throw MetalBackendError("sparse mapping list must not be empty");
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    if (impl_->asyncState->hasActiveSubmission()) {
        throw MetalBackendError(
            "cannot map sparse memory while a command is in flight");
    }
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    const uint64_t tileBytes = kPlacementSparsePageBytes;
    for (const SparseMapping &mapping : mappings) {
        if (!mapping.buffer.impl_ ||
            !mapping.buffer.impl_->allocation ||
            mapping.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get() ||
            !mapping.buffer.impl_->allocation->placementSparse) {
            throw MetalBackendError("invalid placement-sparse buffer");
        }
        if (!mapping.sizeBytes ||
            mapping.bufferOffsetBytes % tileBytes ||
            mapping.sizeBytes % tileBytes ||
            mapping.heapOffsetBytes % tileBytes ||
            mapping.bufferOffsetBytes > mapping.buffer.sizeBytes() ||
            mapping.sizeBytes >
                mapping.buffer.sizeBytes() - mapping.bufferOffsetBytes ||
            mapping.heapOffsetBytes > heap.impl_->bytes ||
            mapping.sizeBytes > heap.impl_->bytes - mapping.heapOffsetBytes) {
            throw MetalBackendError("sparse mapping range is invalid");
        }
    }
    if (impl_->nextSparseEventValue ==
        std::numeric_limits<uint64_t>::max()) {
        throw MetalBackendError("sparse event sequence exhausted");
    }

    // A range released moments ago may be mapped again to a new heap. Make
    // the map depend on the in-flight unmap explicitly rather than relying
    // on queue order alone; compute submission follows both completions.
    if (impl_->pendingUnmap) {
        [impl_->sparseQueue waitForEvent:impl_->sparseEvent
                                 value:impl_->pendingUnmap->eventValue];
    }
    for (const SparseMapping &mapping : mappings) {
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeMap;
        operation.bufferRange = NSMakeRange(
            checkedNSUInteger(mapping.bufferOffsetBytes / tileBytes,
                              "sparse buffer tile offset"),
            checkedNSUInteger(mapping.sizeBytes / tileBytes,
                              "sparse mapping tile count"));
        operation.heapOffset = checkedNSUInteger(
            mapping.heapOffsetBytes / tileBytes, "sparse heap tile offset");
        [impl_->sparseQueue
            updateBufferMappings:mapping.buffer.impl_->allocation->buffer
                             heap:heap.impl_->heap
                       operations:&operation
                            count:1];
    }
    const uint64_t eventValue = ++impl_->nextSparseEventValue;
    // A failed dependency wait must not release backing still being mapped.
    auto retainedHeap = heap.impl_;
    std::vector<SparseMapping> retainedMappings(mappings.begin(), mappings.end());
    id<MTL4CommandQueue> retainedQueue = impl_->sparseQueue;
    id<MTLSharedEvent> retainedEvent = impl_->sparseEvent;
    [impl_->sparseEvent notifyListener:[MTLSharedEventListener sharedListener]
        atValue:eventValue block:^(id<MTLSharedEvent>, uint64_t) {
            (void)retainedHeap;
            (void)retainedMappings;
            (void)retainedQueue;
            (void)retainedEvent;
        }];
    [impl_->sparseQueue signalEvent:impl_->sparseEvent value:eventValue];
    impl_->pendingSparseEventValue = eventValue;
}

void MetalBackend::unmapSparse(
    std::span<const SparseMapping> mappings, SparseHeap &&heap) {
    if (mappings.empty()) {
        throw MetalBackendError("sparse unmapping list must not be empty");
    }
    if (!heap.impl_ || !heap.impl_->heap ||
        heap.impl_->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError(
            "sparse unmapping requires the mapped placement heap");
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    if (impl_->asyncState->hasActiveSubmission()) {
        throw MetalBackendError(
            "cannot unmap sparse memory while a command is in flight");
    }
    const uint64_t tileBytes = kPlacementSparsePageBytes;
    for (const SparseMapping &mapping : mappings) {
        if (!mapping.buffer.impl_ ||
            !mapping.buffer.impl_->allocation ||
            mapping.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get() ||
            !mapping.buffer.impl_->allocation->placementSparse ||
            !mapping.sizeBytes ||
            mapping.bufferOffsetBytes % tileBytes ||
            mapping.sizeBytes % tileBytes ||
            mapping.bufferOffsetBytes > mapping.buffer.sizeBytes() ||
            mapping.sizeBytes >
                mapping.buffer.sizeBytes() - mapping.bufferOffsetBytes) {
            throw MetalBackendError("sparse unmapping range is invalid");
        }
    }
    if (impl_->nextSparseEventValue ==
        std::numeric_limits<uint64_t>::max()) {
        throw MetalBackendError("sparse event sequence exhausted");
    }

    // One outstanding unmap at a time keeps the kernel's per-tile teardown
    // paced; the caller normally checks sparseUnmapPending() first.
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    impl_->awaitSparseUnmapLocked();

    // Allocation rollback may unmap before compute has consumed the map
    // event. Order that dependent update explicitly on the Metal 4 queue.
    if (impl_->pendingSparseEventValue) {
        [impl_->sparseQueue waitForEvent:impl_->sparseEvent
                                 value:impl_->pendingSparseEventValue];
    }
    for (const SparseMapping &mapping : mappings) {
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeUnmap;
        operation.bufferRange = NSMakeRange(
            checkedNSUInteger(mapping.bufferOffsetBytes / tileBytes,
                              "sparse buffer tile offset"),
            checkedNSUInteger(mapping.sizeBytes / tileBytes,
                              "sparse unmapping tile count"));
        [impl_->sparseQueue
            updateBufferMappings:mapping.buffer.impl_->allocation->buffer
                             heap:nil
                       operations:&operation
                            count:1];
    }
    const uint64_t eventValue = ++impl_->nextSparseEventValue;
    [impl_->sparseQueue signalEvent:impl_->sparseEvent value:eventValue];
    const auto issued = std::chrono::steady_clock::now();
    impl_->pendingUnmap.emplace();
    impl_->pendingUnmap->eventValue = eventValue;
    impl_->pendingUnmap->heap = std::move(heap);
    impl_->pendingUnmap->issued = issued;
    impl_->pendingUnmapIssuedSeconds.store(
        std::chrono::duration<double>(issued.time_since_epoch()).count(),
        std::memory_order_relaxed);
    impl_->pendingUnmapCount.store(1, std::memory_order_release);
    impl_->sampleDeviceMemory();
}

bool MetalBackend::sparseUnmapPending() noexcept {
    // Reap opportunistically; a command being encoded on another thread
    // must not stall the caller, which is often the reclaim pacing loop.
    if (std::unique_lock commandLock(impl_->commandMutex, std::try_to_lock);
        commandLock.owns_lock()) {
        static_cast<void>(impl_->reapSparseUnmapsLocked());
    }
    if (impl_->pendingUnmapCount.load(std::memory_order_acquire) == 0)
        return false;
    // An unmap outstanding for longer than the drain's bounded wait is the
    // same fault the drain would report, observed here without blocking the
    // serving loop: the backend marks itself unhealthy and the supervisor
    // replaces the engine.
    const double issued =
        impl_->pendingUnmapIssuedSeconds.load(std::memory_order_relaxed);
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (issued > 0.0 &&
        (now - issued) * 1000.0 > double(kSparseUnmapTimeoutMilliseconds)) {
        try {
            impl_->markUnhealthy(
                "sparse unmapping exceeded " +
                std::to_string(kSparseUnmapTimeoutMilliseconds) +
                " ms without completing");
        } catch (...) {
        }
    }
    return true;
}

void MetalBackend::drainSparseUnmaps() {
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    impl_->awaitSparseUnmapLocked();
}

MetalBuffer MetalBackend::wrapSharedMemory(
    void *address, uint64_t bytes, std::shared_ptr<void> lifetime,
    std::string_view label) {
    checkOperation();
    if (!address || !bytes) {
        throw MetalBackendError("shared memory address and size are required");
    }
    if (!lifetime) {
        throw MetalBackendError("shared memory lifetime token is required");
    }
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("shared memory exceeds maxBufferLength");
    }
    long systemPageSize = sysconf(_SC_PAGESIZE);
    if (systemPageSize <= 0) {
        throw MetalBackendError("unable to determine system page size");
    }
    uint64_t pageSize = static_cast<uint64_t>(systemPageSize);
    if (reinterpret_cast<uintptr_t>(address) % pageSize || bytes % pageSize) {
        throw MetalBackendError(
            "shared memory address and size must be page-aligned");
    }

    id<MTLBuffer> buffer = [impl_->device
        newBufferWithBytesNoCopy:address
        length:checkedNSUInteger(bytes, "shared memory size")
        options:MTLResourceStorageModeShared
        deallocator:^(void *, NSUInteger) {
            // Metal may retain the buffer beyond our last C++ view/ticket,
            // including while a completed command's handler is returning.
            // Keep its backing owner until Metal actually releases it.
            (void)lifetime;
        }];
    if (!buffer) {
        throw MetalBackendError("zero-copy Metal buffer creation failed");
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, BufferStorage::Shared,
                                 std::move(lifetime));
}

MetalBuffer MetalBackend::view(const MetalBuffer &base,
                               uint64_t offsetBytes,
                               uint64_t lengthBytes) const {
    impl_->ensureHealthy();
    if (!base.impl_ || !base.impl_->allocation) {
        throw MetalBackendError("cannot view an empty Metal buffer");
    }
    if (base.impl_->allocation->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("Metal buffer belongs to another backend");
    }
    if (!lengthBytes || offsetBytes > base.impl_->lengthBytes ||
        lengthBytes > base.impl_->lengthBytes - offsetBytes) {
        std::ostringstream message;
        message << "Metal buffer view is out of range: offset=" << offsetBytes
                << " length=" << lengthBytes
                << " base_length=" << base.impl_->lengthBytes;
        throw MetalBackendError(message.str());
    }
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->allocation = base.impl_->allocation;
    result->offsetBytes = base.impl_->offsetBytes + offsetBytes;
    result->lengthBytes = lengthBytes;
    return MetalBuffer(std::move(result));
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitAsync(dispatch).wait();
}

CommandTiming MetalBackend::submitCommand(
    std::span<const ComputeDispatch> dispatches) {
    return submitCommandAsync(dispatches).wait();
}

CommandTicket MetalBackend::submitAsync(
    const ComputeDispatch &dispatch, CommandCompletion completion) {
    return submitCommandAsync(
        std::span<const ComputeDispatch>(&dispatch, 1),
        std::move(completion));
}

void MetalBackend::setDispatchProfiling(bool enabled) noexcept {
    impl_->dispatchProfiling = enabled;
}

std::vector<DispatchTiming> MetalBackend::takeDispatchProfile() {
    return std::exchange(impl_->dispatchProfile, {});
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    bool trailingCommitted = false;
    return submitCommandAsync(dispatches, std::move(completion),
                              TrailingBuilder{}, trailingCommitted);
}

void MetalBackend::awaitTrailing() {
    id<MTLCommandBuffer> trail = nil;
    dispatch_semaphore_t done = nil;
    {
        std::lock_guard lock(impl_->commandMutex);
        trail = impl_->trailing;
        done = impl_->trailingDone;
        impl_->trailing = nil;
        impl_->trailingDone = nil;
    }
    if (!trail) return;
    // Bounded like the command watchdog: a stuck block must not hang the
    // engine thread until the OS GPU timeout.
    const double timeout = impl_->asyncState->commandWatchdog.timeoutSeconds();
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW,
            static_cast<int64_t>(timeout * NSEC_PER_SEC))) != 0) {
        std::string message = "Metal trailing command timed out";
        impl_->markUnhealthy(message);
        throw MetalBackendError(message);
    }
    if (trail.status != MTLCommandBufferStatusCompleted) {
        std::string message = "Metal trailing command failed";
        if (trail.error) message += ": " + errorDescription(trail.error);
        impl_->markUnhealthy(message);
        throw MetalBackendError(message);
    }
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion, const TrailingBuilder &trailing,
    bool &trailingCommitted) {
    return submitCommandAsync(dispatches, std::move(completion), trailing,
                              trailingCommitted, nullptr);
}

uint64_t MetalBackend::chainValue() const {
    return impl_ && impl_->chainEvent ? impl_->chainEvent.signaledValue : 0;
}

void MetalBackend::signalChain(uint64_t value) {
    if (!impl_ || !impl_->chainEvent) return;
    if (impl_->chainEvent.signaledValue < value)
        impl_->chainEvent.signaledValue = value;
}

void MetalBackend::notifyChain(uint64_t value, std::function<void()> callback) {
    if (!impl_->chainEvent) throw MetalBackendError("no gated command was submitted");
    [impl_->chainEvent notifyListener:[MTLSharedEventListener sharedListener]
                              atValue:value
                                block:^(id<MTLSharedEvent>, uint64_t) {
                                  if (callback) callback();
                                }];
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion, const TrailingBuilder &trailing,
    bool &trailingCommitted, const ChainGates *gates) {
    trailingCommitted = false;
    checkOperation();
    if (dispatches.empty()) {
        throw MetalBackendError("Metal command must contain a dispatch");
    }
    if (impl_->dispatchProfiling && dispatches.size() > 1) {
        // Replay serially, one command per dispatch, then hand back an
        // already-completed ticket carrying the summed timing so callers
        // observe the usual asynchronous contract.
        CommandTiming total;
        for (const ComputeDispatch &dispatch : dispatches) {
            CommandTiming timing = submitAsync(dispatch).wait();
            impl_->dispatchProfile.push_back(
                {dispatch.pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = impl_->asyncState;
        ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion(ticketState->sequence);
        return CommandTicket(std::move(ticketState));
    }
    std::vector<PreparedDispatch> prepared =
        Impl::prepareDispatches(dispatches, impl_->accounting.get());
    hostPhase("valid");

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    // SPLASH_STREAMED_SUBMIT: dispatches [0, headEnd) already run as this
    // submission's committed head (streamHead); only the rest is encoded here.
    std::optional<Impl::StreamedHead> streamed = std::exchange(impl_->streamed, std::nullopt);
    const size_t headEnd = streamed ? streamed->count : 0;
    auto ticketState = streamed ? streamed->ticket : std::make_shared<CommandTicket::State>();
    auto failBeforeCommit = [&](std::string message) {
        impl_->markUnhealthy(message);
        impl_->asyncState->releaseSubmission(ticketState->sequence);
        throw MetalBackendError(std::move(message));
    };
    if (streamed && (headEnd >= prepared.size() || streamed->gated != (gates != nullptr) ||
                     (gates && (headEnd < gates->signalBefore || headEnd > gates->waitBefore)))) {
        failBeforeCommit("streamed head does not fit its submission");
    }
    try {
        impl_->resolvePipelines(prepared, headEnd);
    } catch (const std::exception &error) {
        if (streamed) failBeforeCommit(error.what());
        throw;
    }

    ticketState->backend = impl_->asyncState;
    ticketState->completion = std::move(completion);
    std::unordered_set<const MetalAllocation *> retained;
    if (streamed) retained = std::move(streamed->retained);
    for (const ComputeDispatch &dispatch : dispatches.subspan(headEnd)) {
        for (const BufferBinding &binding : dispatch.buffers) {
            const auto &allocation = binding.buffer.impl_->allocation;
            if (retained.insert(allocation.get()).second) {
                ticketState->retainedAllocations.push_back(allocation);
            }
        }
    }
    if (!streamed)
        ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());

    auto wallStart = streamed ? streamed->wallStart : std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    if (!command) {
        failBeforeCommit("unable to create Metal command buffer");
    }
    uint64_t sparseEventValue = impl_->pendingSparseEventValue;
    if (gates) {
        if (!gates->signalBefore || gates->signalBefore > gates->waitBefore ||
            gates->waitBefore >= prepared.size() ||
            gates->waitValue <= gates->signalValue) {
            failBeforeCommit("invalid chain gates");
        }
        if (!impl_->chainEvent) impl_->chainEvent = [impl_->device newSharedEvent];
        if (!impl_->gateFence) impl_->gateFence = [impl_->device newFence];
        if (!impl_->chunkFence) impl_->chunkFence = [impl_->device newFence];
        if (!impl_->chainEvent || !impl_->gateFence || !impl_->chunkFence)
            failBeforeCommit("unable to create Metal chain gate objects");
        if (!streamed && gates->signalValue <= impl_->chainEvent.signaledValue)
            failBeforeCommit("chain gate values must increase");
        // The head commits at once, so a pending sparse mapping is resolved
        // on the host first.
        if (sparseEventValue && impl_->sparseEvent.signaledValue < sparseEventValue &&
            ![impl_->sparseEvent waitUntilSignaledValue:sparseEventValue
                                               timeoutMS:kSparseMapTimeoutMilliseconds]) {
            failBeforeCommit("sparse mapping did not finish before a gated command");
        }
        sparseEventValue = 0;
        uint64_t highest = impl_->chainHighest.load();
        while (highest < gates->waitValue &&
               !impl_->chainHighest.compare_exchange_weak(highest, gates->waitValue)) {
        }
    }
    ticketState->wallStart = wallStart;
    ticketState->sparseEventValue = sparseEventValue;
    if (sparseEventValue) {
        // Keep the queue dependency explicit; the CPU resolves it before commit.
        [command encodeWaitForEvent:impl_->sparseEvent value:sparseEventValue];
    }
    // Encoders can remain autoreleased after their command has completed.
    // The serving loop is long-lived, so bound their temporary ownership to
    // encoding; the command retains everything needed for GPU execution.
    const size_t headCount = impl_->chunkHeadDispatches;
    const bool chunked = !streamed && !gates && headCount && !sparseEventValue &&
                         prepared.size() > 2 * headCount;
    // A trailing command needs a synchronous commit (no pending sparse map).
    const bool withTrailing = trailing && !sparseEventValue;
    const bool waitTrailing = impl_->trailingUnordered;
    if (withTrailing && !impl_->trailFenceIn) {
        impl_->trailFenceIn = [impl_->device newFence];
        impl_->trailFenceOut = [impl_->device newFence];
    }
    if (gates && streamed) {
        ticketState->head = streamed->head;
        ticketState->gated = true;
    } else if (gates) {
        // Head: the dispatches before the signal, committed now; it raises the
        // chain event once they finish.
        id<MTLCommandBuffer> head = [impl_->queue commandBuffer];
        if (!head) failBeforeCommit("unable to create Metal head command buffer");
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = [head computeCommandEncoder];
            if (!encoder) failBeforeCommit("unable to create Metal compute encoder");
            try {
                if (waitTrailing) [encoder waitForFence:impl_->trailFenceOut];
                Impl::encodeDispatches(encoder, prepared, 0, gates->signalBefore);
                [encoder updateFence:impl_->chunkFence];
                [encoder endEncoding];
            } catch (...) {
                impl_->asyncState->releaseSubmission(ticketState->sequence);
                throw;
            }
        }
        [head encodeSignalEvent:impl_->chainEvent value:gates->signalValue];
        if (impl_->asyncState->stopping.stop_requested())
            failBeforeCommit("Metal backend stopped before command submission");
        ticketState->head = head;
        ticketState->gated = true;
        [head commit];
        hostPhase("head", static_cast<long long>(ticketState->sequence));
    }
    if (gates) {
        // Tail: the middle dispatches, the host gate, then the rest.
        const size_t middleBegin = streamed ? headEnd : gates->signalBefore;
        const bool middle = gates->waitBefore > middleBegin;
        @autoreleasepool {
            if (middle) {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) failBeforeCommit("unable to create Metal compute encoder");
                try {
                    [encoder waitForFence:impl_->chunkFence];
                    Impl::encodeDispatches(encoder, prepared, middleBegin, gates->waitBefore);
                    [encoder updateFence:impl_->gateFence];
                    [encoder endEncoding];
                } catch (...) {
                    impl_->asyncState->releaseSubmission(ticketState->sequence);
                    throw;
                }
            }
            [command encodeWaitForEvent:impl_->chainEvent value:gates->waitValue];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) failBeforeCommit("unable to create Metal compute encoder");
            try {
                [encoder waitForFence:middle ? impl_->gateFence : impl_->chunkFence];
                Impl::encodeDispatches(encoder, prepared, gates->waitBefore, prepared.size());
                if (withTrailing) [encoder updateFence:impl_->trailFenceIn];
                [encoder endEncoding];
            } catch (...) {
                impl_->asyncState->releaseSubmission(ticketState->sequence);
                throw;
            }
        }
    } else {
        if (chunked) {
            // Untracked heap resources: the fence orders the tail after the head.
            if (!impl_->chunkFence) impl_->chunkFence = [impl_->device newFence];
            id<MTLCommandBuffer> head = [impl_->queue commandBuffer];
            if (!head) failBeforeCommit("unable to create Metal head command buffer");
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = [head computeCommandEncoder];
                if (!encoder) failBeforeCommit("unable to create Metal compute encoder");
                try {
                    if (waitTrailing) [encoder waitForFence:impl_->trailFenceOut];
                    Impl::encodeDispatches(encoder, prepared, 0, headCount);
                    [encoder updateFence:impl_->chunkFence];
                    [encoder endEncoding];
                } catch (...) {
                    impl_->asyncState->releaseSubmission(ticketState->sequence);
                    throw;
                }
            }
            if (impl_->asyncState->stopping.stop_requested())
                failBeforeCommit("Metal backend stopped before command submission");
            ticketState->head = head;
            [head commit];
            hostPhase("head", static_cast<long long>(ticketState->sequence));
        }
        if (streamed) ticketState->head = streamed->head;
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) {
                failBeforeCommit("unable to create Metal compute encoder");
            }
            try {
                if (chunked || streamed) [encoder waitForFence:impl_->chunkFence];
                else if (waitTrailing) [encoder waitForFence:impl_->trailFenceOut];
                const size_t tailBegin = chunked ? headCount : headEnd;
                // SPLASH_SEAM_SIBLING: from the first sibling's partner on, a
                // concurrent encoder with a buffer barrier before every
                // non-sibling, so each sibling overlaps only its partner (and
                // earlier siblings of it); every other boundary keeps the
                // serial encoder's order. Resources are untracked, so a fence
                // orders the serial part before the concurrent one.
                size_t concurrentFrom = prepared.size();
                for (size_t index = tailBegin + 1; index < prepared.size(); ++index) {
                    if (prepared[index].source->sibling) {
                        concurrentFrom = index - 1;
                        break;
                    }
                }
                Impl::encodeDispatches(encoder, prepared, tailBegin, concurrentFrom);
                if (concurrentFrom < prepared.size()) {
                    if (!impl_->siblingFence) impl_->siblingFence = [impl_->device newFence];
                    if (!impl_->siblingFence) failBeforeCommit("unable to create Metal sibling fence");
                    [encoder updateFence:impl_->siblingFence];
                    [encoder endEncoding];
                    encoder = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    if (!encoder) failBeforeCommit("unable to create Metal concurrent compute encoder");
                    [encoder waitForFence:impl_->siblingFence];
                    for (size_t index = concurrentFrom; index < prepared.size(); ++index) {
                        if (index > concurrentFrom && !prepared[index].source->sibling)
                            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        Impl::encodeDispatches(encoder, prepared, index, index + 1);
                    }
                }
                if (withTrailing) [encoder updateFence:impl_->trailFenceIn];
                [encoder endEncoding];
            } catch (...) {
                impl_->asyncState->releaseSubmission(ticketState->sequence);
                throw;
            }
        }
    }

    impl_->trailingUnordered = false;

    // Driver callbacks only complete the ticket. Device-wide memory telemetry
    // is sampled on the host before submission and when consuming the result.
    std::shared_ptr<BackendAsyncState> observer = impl_->asyncState;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completedCommand) {
        ticketState->finishCommand(completedCommand);
    }];
    impl_->sampleDeviceMemory();
    id<MTLSharedEvent> event = impl_->sparseEvent;
    const bool pendingMap =
        sparseEventValue && event.signaledValue < sparseEventValue;
    const double mapWaitStart = steadySeconds();
    if (pendingMap) {
        observer->mapWaitStarted.store(mapWaitStart, std::memory_order_relaxed);
        observer->mapWaitEvent.store(sparseEventValue, std::memory_order_release);
    }
    afterMetalEvent(event, sparseEventValue, kSparseMapTimeoutMilliseconds,
        [command, event, observer, ticketState, sparseEventValue,
         pendingMap, mapWaitStart, wallStart](bool signaled) {
            if (pendingMap) {
                const double waited = steadySeconds() - mapWaitStart;
                observer->lastMapWaitSeconds.store(waited, std::memory_order_relaxed);
                raisePeak(observer->maxMapWaitSeconds, waited);
                observer->mapWaitEvent.store(0, std::memory_order_release);
            }
            if (observer->stopping.stop_requested()) {
                ticketState->finish({}, "Metal backend stopped before command submission");
                return;
            }
            if (!signaled || !observer->healthy.load(std::memory_order_acquire)) {
                std::ostringstream message;
                message << "sparse mapping dependency failed before Metal command "
                        << ticketState->sequence << ": event " << sparseEventValue
                        << ", signaled " << event.signaledValue;
                if (!signaled)
                    message << ", wait exceeded "
                            << kSparseMapTimeoutMilliseconds << " ms";
                CommandTiming timing;
                timing.wallSeconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wallStart).count();
                ticketState->finish(timing, message.str());
                return;
            }
            if (!observer->commitSubmission(ticketState->sequence, command,
                    [weakTicket = std::weak_ptr(ticketState)](id<MTLCommandBuffer> completed) {
                        if (auto ticket = weakTicket.lock()) ticket->finishCommand(completed);
                    })) {
                ticketState->finish({}, "Metal backend stopped before command submission");
                return;
            }
        }, observer->stopping.get_token());
    hostPhase("commit", static_cast<long long>(ticketState->sequence));
    impl_->pendingSparseEventValue = 0;
    CommandTicket ticket(std::move(ticketState));
    // SPLASH_DRAFT_AHEAD: built by the caller only after the command is
    // committed, so building it never delays the GPU start of the command.
    if (!withTrailing || command.status == MTLCommandBufferStatusNotEnqueued)
        return ticket;
    const std::span<const ComputeDispatch> trailingDispatches = trailing();
    if (trailingDispatches.empty())
        return ticket;
    std::vector<PreparedDispatch> trailingPrepared =
        Impl::prepareDispatches(trailingDispatches, impl_->accounting.get());
    impl_->resolvePipelines(trailingPrepared);
    id<MTLCommandBuffer> trail = [impl_->queue commandBuffer];
    if (!trail) {
        impl_->markUnhealthy("unable to create Metal trailing command buffer");
        return ticket;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [trail computeCommandEncoder];
        if (!encoder) {
            impl_->markUnhealthy("unable to create Metal compute encoder");
            return ticket;
        }
        [encoder waitForFence:impl_->trailFenceIn];
        Impl::encodeDispatches(encoder, trailingPrepared, 0, trailingPrepared.size());
        [encoder updateFence:impl_->trailFenceOut];
        [encoder endEncoding];
    }
    // The command buffer retains the Metal buffers; this keeps their
    // allocation accounting alive until the GPU is done with them.
    auto retainedTrail =
        std::make_shared<std::vector<std::shared_ptr<MetalAllocation>>>();
    for (const ComputeDispatch &dispatch : trailingDispatches)
        for (const BufferBinding &binding : dispatch.buffers)
            retainedTrail->push_back(binding.buffer.impl_->allocation);
    dispatch_semaphore_t trailDone = dispatch_semaphore_create(0);
    [trail addCompletedHandler:^(id<MTLCommandBuffer> done) {
        retainedTrail->clear();
        static const bool gapLog = std::getenv("SPLASH_GPU_GAP_LOG") != nullptr;
        if (gapLog)
            std::fprintf(stderr, "gpu_ahead %.9f %.9f\n", done.GPUStartTime,
                         done.GPUEndTime);
        if (done.status != MTLCommandBufferStatusCompleted)
            observer->markUnhealthy("Metal trailing command failed");
        dispatch_semaphore_signal(trailDone);
    }];
    [trail commit];
    impl_->trailing = trail;
    impl_->trailingDone = trailDone;
    impl_->trailingUnordered = true;
    trailingCommitted = true;
    return ticket;
}

bool MetalBackend::streamHead(std::span<const ComputeDispatch> head,
                              uint64_t signalValue) {
    checkOperation();
    if (head.empty() || impl_->dispatchProfiling) return false;
    hostPhase("stream");
    std::vector<PreparedDispatch> prepared =
        Impl::prepareDispatches(head, impl_->accounting.get());
    hostPhase("valid");
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    if (impl_->streamed) throw MetalBackendError("a streamed head is already pending");
    // The head commits at once: with a sparse mapping still pending, the
    // submission encodes as usual instead.
    if (impl_->pendingSparseEventValue) {
        if (impl_->sparseEvent.signaledValue < impl_->pendingSparseEventValue) return false;
        impl_->pendingSparseEventValue = 0;
    }
    impl_->resolvePipelines(prepared);
    if (!impl_->chunkFence) impl_->chunkFence = [impl_->device newFence];
    if (signalValue) {
        if (!impl_->chainEvent) impl_->chainEvent = [impl_->device newSharedEvent];
        if (!impl_->gateFence) impl_->gateFence = [impl_->device newFence];
        if (!impl_->chainEvent || !impl_->gateFence)
            throw MetalBackendError("unable to create Metal chain gate objects");
        if (signalValue <= impl_->chainEvent.signaledValue)
            throw MetalBackendError("chain gate values must increase");
    }
    if (!impl_->chunkFence) throw MetalBackendError("unable to create Metal chunk fence");

    Impl::StreamedHead streamed;
    streamed.ticket = std::make_shared<CommandTicket::State>();
    streamed.ticket->backend = impl_->asyncState;
    for (const ComputeDispatch &dispatch : head) {
        for (const BufferBinding &binding : dispatch.buffers) {
            const auto &allocation = binding.buffer.impl_->allocation;
            if (streamed.retained.insert(allocation.get()).second)
                streamed.ticket->retainedAllocations.push_back(allocation);
        }
    }
    const uint64_t sequence = impl_->asyncState->beginSubmission(head.size());
    streamed.ticket->sequence = sequence;
    const auto fail = [&](std::string message) {
        impl_->markUnhealthy(message);
        impl_->asyncState->releaseSubmission(sequence);
        throw MetalBackendError(std::move(message));
    };
    streamed.wallStart = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    if (!command) fail("unable to create Metal head command buffer");
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) fail("unable to create Metal compute encoder");
        try {
            if (impl_->trailingUnordered) [encoder waitForFence:impl_->trailFenceOut];
            Impl::encodeDispatches(encoder, prepared, 0, prepared.size());
            [encoder updateFence:impl_->chunkFence];
            [encoder endEncoding];
        } catch (...) {
            impl_->asyncState->releaseSubmission(sequence);
            throw;
        }
    }
    if (signalValue) [command encodeSignalEvent:impl_->chainEvent value:signalValue];
    if (impl_->asyncState->stopping.stop_requested())
        fail("Metal backend stopped before command submission");
    [command commit];
    hostPhase("head", static_cast<long long>(sequence));
    streamed.head = command;
    streamed.count = head.size();
    streamed.gated = signalValue != 0;
    impl_->streamed = std::move(streamed);
    return true;
}

bool MetalBackend::streamedHeadPending() const noexcept {
    std::lock_guard lock(impl_->commandMutex);
    return impl_->streamed.has_value();
}

void MetalBackend::abandonStreamedHead() noexcept {
    std::optional<Impl::StreamedHead> streamed;
    {
        std::lock_guard lock(impl_->commandMutex);
        streamed = std::exchange(impl_->streamed, std::nullopt);
    }
    if (!streamed) return;
    // Its buffers stay retained until the GPU is done with the head (bounded
    // like the command watchdog).
    const double deadline =
        steadySeconds() + impl_->asyncState->commandWatchdog.timeoutSeconds();
    while (streamed->head.status < MTLCommandBufferStatusCompleted &&
           steadySeconds() < deadline)
        usleep(100);
    if (streamed->head.status < MTLCommandBufferStatusCompleted)
        impl_->markUnhealthy("streamed head did not finish after its build failed");
    impl_->asyncState->releaseSubmission(streamed->ticket->sequence);
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    uint64_t deviceCurrent =
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed);
    const uint64_t pendingUnmaps =
        impl_->pendingUnmapCount.load(std::memory_order_acquire);
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        deviceCurrent,
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->sparseVirtualBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->sparseResidentBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->peakSparseResidentBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->peakResidentBytes.load(std::memory_order_relaxed),
        kPlacementSparsePageBytes,
        pendingUnmaps,
        impl_->completedUnmaps.load(std::memory_order_relaxed),
        impl_->lastUnmapSeconds.load(std::memory_order_relaxed),
        impl_->maxUnmapSeconds.load(std::memory_order_relaxed),
        pendingUnmaps
            ? std::max(0.0, steadySeconds() -
                                impl_->pendingUnmapIssuedSeconds.load(
                                    std::memory_order_relaxed))
            : 0.0,
        impl_->asyncState->mapWaitEvent.load(std::memory_order_acquire),
        impl_->asyncState->mapWaitEvent.load(std::memory_order_acquire)
            ? std::max(0.0, steadySeconds() -
                impl_->asyncState->mapWaitStarted.load(std::memory_order_relaxed))
            : 0.0,
        impl_->asyncState->lastMapWaitSeconds.load(std::memory_order_relaxed),
        impl_->asyncState->maxMapWaitSeconds.load(std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    {
        // A completed unmap releases its heap here without ever waiting
        // behind an active encode or mapping call.
        std::unique_lock lock(impl_->commandMutex, std::try_to_lock);
        if (lock.owns_lock())
            static_cast<void>(impl_->reapSparseUnmapsLocked());
    }
    impl_->sampleDeviceMemory();
    return memoryStats();
}

uint64_t MetalBackend::submissionCount() const noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    return impl_->asyncState->nextSequence;
}

size_t MetalBackend::pipelineCount() const noexcept {
    std::lock_guard lock(impl_->commandMutex);
    return impl_->pipelines.count;
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
    if (impl_->pendingUnmapCount.load(std::memory_order_acquire)) {
        static_cast<void>(sparseUnmapPending());
        impl_->ensureHealthy();
    }
}

bool MetalBackend::needsHealthCheck() const noexcept {
    return impl_->asyncState->hasActiveSubmission() ||
           impl_->pendingUnmapCount.load(std::memory_order_acquire) != 0;
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

}  // namespace splash::metal
