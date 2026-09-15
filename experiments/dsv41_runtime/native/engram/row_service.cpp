// Bounded native row service for graph host callbacks; no CUDA or Python calls.
// Design reference: MiaAI-Lab's graph-compatible Engram row callback. This
// implementation reads this lab's original separate weight/scale extents.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint64_t width=256, scales=8, row_bytes=264, max_rows=8192;
thread_local std::string last_error;
struct Store;
struct Work { Store* store; const int64_t* ids; uint8_t* weights; uint8_t* scales; uint64_t count; };

struct Store {
    int wf=-1, sf=-1;
    uint64_t wo, so, rows, lo, hi, capacity;
    std::vector<uint8_t> cache;
    std::vector<int64_t> keys;
    std::mutex stripes[64], mutex, caller;
    std::condition_variable ready, done;
    std::vector<std::thread> threads;
    bool stopping=false;
    uint64_t generation=0, finished=0;
    Work* job=nullptr;
    std::atomic<uint64_t> next{0}, hits{0}, misses{0}, reads{0}, evictions{0}, lookups{0};
    std::atomic<int> failure{0};

    static int duplicate(int fd, uint64_t offset, uint64_t n, uint64_t bytes) {
        struct stat st{};
        if(fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<0 ||
           offset>uint64_t(st.st_size) || n>(uint64_t(st.st_size)-offset)/bytes)
            throw std::runtime_error("invalid original tensor extent");
        int result=fcntl(fd,F_DUPFD_CLOEXEC,0);
        if(result<0) throw std::runtime_error("could not duplicate original tensor descriptor");
        return result;
    }
    Store(int w,int s,uint64_t woff,uint64_t soff,uint64_t n,uint64_t start,uint64_t end,uint64_t cap,uint64_t workers):
        wo(woff),so(soff),rows(n),lo(start),hi(end),capacity(cap) {
        if(!n || start>=end || end>n || !cap || cap>131072 || !workers || workers>4)
            throw std::runtime_error("row service exceeds qualified bounds");
        try {
            wf=duplicate(w,wo,n,width); sf=duplicate(s,so,n,scales);
            cache.resize(cap*row_bytes); keys.assign(cap,-1);
            for(uint64_t i=0;i<workers;++i) threads.emplace_back([this]{loop();});
        } catch(...) { shutdown(); throw; }
    }
    ~Store(){shutdown();}
    void shutdown() {
        { std::lock_guard<std::mutex> guard(mutex); stopping=true; }
        ready.notify_all();
        for(auto& t:threads) if(t.joinable()) t.join();
        threads.clear();
        if(wf>=0) close(wf); if(sf>=0) close(sf); wf=sf=-1;
    }
    bool read_exact(int fd,uint64_t offset,uint8_t* out,uint64_t bytes) {
        uint64_t got=0;
        while(got<bytes) {
            ssize_t n=pread(fd,out+got,bytes-got,off_t(offset+got));
            if(n<0 && errno==EINTR) continue;
            if(n<=0) {failure.store(1);return false;}
            got+=uint64_t(n); reads.fetch_add(1);
        }
        return true;
    }
    void row(Work* work,uint64_t i) {
        int64_t id=work->ids[i];
        uint8_t* w=work->weights+i*width; uint8_t* s=work->scales+i*scales;
        if(id<0 || uint64_t(id)<lo || uint64_t(id)>=hi) {
            std::memset(w,0,width);std::memset(s,0,scales);return;
        }
        const uint64_t slot=uint64_t(id)%capacity;
        {
            std::lock_guard<std::mutex> guard(stripes[slot%64]);
            if(keys[slot]==id) {
                std::memcpy(w,&cache[slot*row_bytes],width);
                std::memcpy(s,&cache[slot*row_bytes+width],scales);hits.fetch_add(1);return;
            }
        }
        uint8_t data[row_bytes];
        if(!read_exact(wf,wo+uint64_t(id)*width,data,width) ||
           !read_exact(sf,so+uint64_t(id)*scales,data+width,scales)) return;
        misses.fetch_add(1);
        std::memcpy(w,data,width);std::memcpy(s,data+width,scales);
        {
            std::lock_guard<std::mutex> guard(stripes[slot%64]);
            if(keys[slot]>=0 && keys[slot]!=id) evictions.fetch_add(1);
            std::memcpy(&cache[slot*row_bytes],data,row_bytes);keys[slot]=id;
        }
    }
    void loop() {
        uint64_t seen=0;
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
            ready.wait(lock,[&]{return stopping || generation!=seen;});
            if(stopping)return;
            seen=generation; Work* current=job;lock.unlock();
            try {
                for(uint64_t i;(i=next.fetch_add(1))<current->count;) {
                    if(failure.load())break;
                    row(current,i);
                }
            } catch(...) {failure.store(2);}
            lock.lock();++finished;
            if(finished==threads.size())done.notify_one();
        }
    }
    int lookup(Work* work) {
        std::lock_guard<std::mutex> serial(caller);
        if(!work || work->store!=this || work->count>max_rows ||
           (work->count && (!work->ids || !work->weights || !work->scales)))return 3;
        std::unique_lock<std::mutex> lock(mutex);
        if(stopping || failure.load())return 4;
        job=work;next.store(0);finished=0;++generation;ready.notify_all();
        done.wait(lock,[&]{return finished==threads.size();});job=nullptr;
        lookups.fetch_add(1);return failure.load();
    }
};
}
extern "C" {
void* sg_create(int w,int s,uint64_t wo,uint64_t so,uint64_t n,uint64_t lo,uint64_t hi,uint64_t cap,uint64_t workers) {
    try {return new Store(w,s,wo,so,n,lo,hi,cap,workers);}
    catch(const std::exception& e){last_error=e.what();return nullptr;}
    catch(...){last_error="unknown row service construction failure";return nullptr;}
}
const char* sg_error(){return last_error.c_str();}
int sg_lookup_test(void* p) {
    auto* work=static_cast<Work*>(p);
    if(!work || !work->store)return 3;
    try{return work->store->lookup(work);}catch(...){return 5;}
}
void sg_lookup(void* p) {
    if(sg_lookup_test(p)) {
        std::fputs("SAGE Engram row callback failed; refusing stale output\n",stderr);
        std::abort();
    }
}
void sg_stats(void* p,uint64_t* out) {
    auto* s=static_cast<Store*>(p);
    out[0]=s->hits.load();out[1]=s->misses.load();out[2]=s->reads.load();
    out[3]=s->evictions.load();out[4]=s->lookups.load();out[5]=s->failure.load();
    out[6]=s->capacity;out[7]=s->threads.size();
}
void sg_destroy(void* p){delete static_cast<Store*>(p);}
}
