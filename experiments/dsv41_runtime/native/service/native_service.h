#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sage_native {
struct Request { int64_t sequence=0, layer=0; std::array<int64_t,6> ids{}; };
// Implemented by the CUDA translation unit with HOST system-scope atomics.
// These methods must never enter Python or call the CUDA runtime.
struct MailboxPort {
    virtual ~MailboxPort() = default;
    virtual Request poll() const = 0;
    virtual void commit(int64_t) = 0;
    virtual int cancelled() const = 0;
    virtual void cancel() = 0;
    virtual bool paused() const = 0;
};
struct Record {
    std::string key;
    int64_t file_offset=0, bytes=0;
    std::string sha256;
    std::array<int64_t,3> bits{}, markers{};
    std::array<int64_t,9> offsets{};
    int layer=0, expert=0;
};
struct Entry { std::string key; int64_t offset=0, size=0, generation=0; };
using Extents = std::vector<std::pair<int64_t,int64_t>>;
struct Region { Extents free; std::map<int64_t,int64_t> allocated; };
struct Snapshot {
    std::vector<Entry> entries;  // Exact Python OrderedDict order.
    std::map<int,Region> regions;
    int64_t generation=0, resident_bytes=0, last_request=0;
};
using Stats = std::map<std::string,double>;
// Region helpers are CPU-only and also used by the standalone local check.
int64_t allocate(Region&, int64_t bytes);
void release(Region&, int64_t offset);
void validate_region(const Region&, int64_t base, int64_t size);

class Service {
public:
    Service(std::shared_ptr<MailboxPort>, void* arena_cpu, uint64_t arena_gpu,
            int64_t capacity, int64_t* table, int64_t* used, int fd,
            std::vector<Record>, std::map<int,int64_t> quotas,
            int64_t reserve, int slots);
    ~Service();
    Service(const Service&)=delete;
    Service& operator=(const Service&)=delete;
    void import_state(Snapshot);
    Snapshot export_state() const;
    void start();  // Call while mailbox is paused, then mailbox.resume(epoch).
    bool stop(int timeout_ms, bool cancel=false);
    bool is_alive() const;
    Stats stats() const;
    std::string error() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace sage_native
