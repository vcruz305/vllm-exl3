#include "native_service.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>

namespace sage_native {
namespace {
constexpr int64_t ALIGN=4096, MAX_RECORD=64LL<<20, MAX_ARENA=80LL<<30;
void check(bool ok, const std::string& msg) { if(!ok) throw std::runtime_error(msg); }
using Clock=std::chrono::steady_clock;
struct Cancelled : std::exception {};
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
struct Buffer {
    void* data=nullptr; size_t size=0;
    explicit Buffer(size_t n):size(n) {
        data=mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        check(data!=MAP_FAILED,"Native staging mmap failed");
    }
    ~Buffer() { if(data && data!=MAP_FAILED) munmap(data,size); }
    Buffer(const Buffer&)=delete;
};
}  // namespace
int64_t allocate(Region& r,int64_t size) {
    check(size>0 && size%ALIGN==0,"Unaligned native allocation");
    for(auto it=r.free.begin();it!=r.free.end();++it) if(it->second>=size) {
        auto offset=it->first;
        if(it->second==size) r.free.erase(it);
        else {it->first+=size;it->second-=size;}
        check(r.allocated.emplace(offset,size).second,"Duplicate allocation");
        return offset;
    }
    return -1;
}
void release(Region& r,int64_t offset) {
    auto it=r.allocated.find(offset);check(it!=r.allocated.end(),"Unknown allocation release");
    r.free.emplace_back(offset,it->second);r.allocated.erase(it);
    std::sort(r.free.begin(),r.free.end()); Extents merged;
    for(auto e:r.free) {
        if(!merged.empty() && merged.back().first+merged.back().second==e.first) merged.back().second+=e.second;
        else merged.push_back(e);
    }
    r.free=std::move(merged);
}
void validate_region(const Region& r,int64_t base,int64_t size) {
    check(base>=0 && size>0 && base%ALIGN==0 && size%ALIGN==0,"Invalid region quota");
    check(std::is_sorted(r.free.begin(),r.free.end()),"Free extents must retain sorted order");
    Extents all=r.free; for(auto e:r.allocated) all.push_back(e);
    std::sort(all.begin(),all.end());int64_t end=base;
    for(auto e:all) {
        check(e.first==end && e.second>0 && e.second%ALIGN==0 && e.second<=base+size-end,"Region gap/overlap/bounds");
        end+=e.second;
    }
    check(end==base+size,"Incomplete region accounting");
}

struct Service::Impl {
    std::shared_ptr<MailboxPort> port;
    uint8_t* arena; uint64_t gpu; int64_t capacity,*table,*used,reserve;
    int fd=-1,slots; void* crypto=nullptr;
    using Hash=unsigned char*(*)(const unsigned char*,size_t,unsigned char*);
    Hash hash=nullptr;
    std::map<std::string,Record> records;
    std::map<int,int64_t> quotas;
    Snapshot state;
    std::thread thread;
    std::atomic<bool> stopping{false};
    mutable std::mutex lifecycle, telemetry;
    std::condition_variable done_cv;
    bool done=true, seeded=false;
    Stats counters;
    std::string failure;
    std::string fatal_path;
    int64_t live_staging=0;
    Impl(std::shared_ptr<MailboxPort> p,void* a,uint64_t g,int64_t c,int64_t* t,int64_t* u,
         int source_fd,std::vector<Record> input,std::map<int,int64_t> q,int64_t r,int n):
         port(std::move(p)),arena(static_cast<uint8_t*>(a)),gpu(g),capacity(c),table(t),used(u),reserve(r),slots(n),quotas(std::move(q)) {
        if(const char* directory=std::getenv("SAGE_TELEMETRY_DIR"))
            if(*directory)fatal_path=std::string(directory)+"/native-service-fatal.json";
        check(port && arena && table && used && uint64_t(arena)==gpu,"Exact shared system arena aliases required");
        check(c>0 && c<=MAX_ARENA && c%ALIGN==0 && r==(8LL<<30) && n>=1 && n<=2,"Admitted arena/reserve/staging bounds required");
        check(source_fd>=0,"Open expert bank descriptor required");
#ifdef O_DIRECT
        auto flags=fcntl(source_fd,F_GETFL);check(flags>=0 && (flags&O_DIRECT),"Native service requires existing O_DIRECT bank fd");
#else
        throw std::runtime_error("Native service requires Linux O_DIRECT");
#endif
        int64_t total=0;
        for(auto e:quotas) {check(e.first>=0 && e.first<40 && e.second>0 && e.second%ALIGN==0 && e.second<=c-total,"Invalid per-layer quota");total+=e.second;}
        check(total==c,"Original exact layer partition required");
        int64_t bank_end=0;std::set<int> layers;
        std::sort(input.begin(),input.end(),[](const Record& a,const Record& b){return a.file_offset<b.file_offset;});
        for(auto rec:input) {
            auto colon=rec.key.find(':');check(colon!=std::string::npos,"Invalid expert key");
            rec.layer=std::stoi(rec.key.substr(0,colon));rec.expert=std::stoi(rec.key.substr(colon+1));
            check(rec.key==std::to_string(rec.layer)+":"+std::to_string(rec.expert) && rec.layer>=0 && rec.layer<40 && rec.expert>=0 && rec.expert<384,"Noncanonical expert key");
            check(rec.file_offset>=bank_end && rec.file_offset%ALIGN==0 && rec.bytes>0 && rec.bytes<=MAX_RECORD && rec.bytes%ALIGN==0,
                  "Bounded nonoverlapping aligned bank records required");
            check(rec.file_offset<=INT64_MAX-rec.bytes,"Expert file offset overflow");
            bank_end=rec.file_offset+rec.bytes;layers.insert(rec.layer);
            check(rec.sha256.size()==64 && rec.sha256.find_first_not_of("0123456789abcdef")==std::string::npos,"Invalid record SHA256");
            for(auto k:rec.bits) check(k>=2 && k<=8,"Original K2-K8 bits required");
            for(auto off:rec.offsets) check(off>=0 && off%256==0 && off<rec.bytes,"Tensor pointer outside record");
            for(auto off:rec.markers) check(off>=0 && off%256==0 && off<=rec.bytes-4,"Marker outside record");
            check(quotas.count(rec.layer) && quotas.at(rec.layer)>=12*rec.bytes,"Complete-batch fragmentation bound failed");
            check(records.emplace(rec.key,std::move(rec)).second,"Duplicate expert");
        }
        check(!records.empty() && layers.size()==quotas.size(),"Every partition must own records");
        // A fixture may select a subset of the already validated full bank.
        // Original ExpertStore remains authoritative for manifest/model identity.
        struct stat info{};check(fstat(source_fd,&info)==0 && S_ISREG(info.st_mode) && info.st_size>=bank_end,"Record exceeds bank file extent");
        // Resolve before execution; no Python imports or crypto loader on workers.
        for(const char* name:{"libcrypto.so.3","libcrypto.so.1.1","libcrypto.so"}) {crypto=dlopen(name,RTLD_NOW|RTLD_LOCAL);if(crypto)break;}
        check(crypto!=nullptr,"System libcrypto unavailable");
        hash=reinterpret_cast<Hash>(dlsym(crypto,"SHA256"));
        if(!hash) {dlclose(crypto);crypto=nullptr;throw std::runtime_error("System SHA256 unavailable");}
        fd=fcntl(source_fd,F_DUPFD_CLOEXEC,3);
        if(fd<0) {dlclose(crypto);crypto=nullptr;throw std::runtime_error("Cannot duplicate expert bank fd");}
    }
    ~Impl() {if(fd>=0)::close(fd);if(crypto)dlclose(crypto);}
    void add(const std::string& key,double value=1) {std::lock_guard<std::mutex> lock(telemetry);counters[key]+=value;}
    void staging(int64_t delta) {
        std::lock_guard<std::mutex> lock(telemetry);live_staging+=delta;
        counters["staging_bytes"]=live_staging;
        counters["peak_staging_bytes"]=std::max(counters["peak_staging_bytes"],double(live_staging));
    }
    void guard() {
        std::ifstream f("/proc/meminfo");check(bool(f),"Cannot read system memory guard");
        std::string line,key;int64_t available=-1,total=-1,free=-1,value;
        while(std::getline(f,line)) {std::istringstream in(line);if(!(in>>key>>value))continue;
            if(key=="MemAvailable:")available=value*1024;
            if(key=="SwapTotal:")total=value*1024;
            if(key=="SwapFree:")free=value*1024;
        }
        check(available>=reserve && total>=0 && total==free,"Native cache reserve/no-swap guard failed");
    }
    void check_cancel() {if(stopping.load(std::memory_order_acquire) || port->cancelled())throw Cancelled();}
    void write_fatal_receipt(const std::string& text) noexcept {
        if(fatal_path.empty())return;
        int out=-1;
        try {
            // ASCII-only escaping keeps even arbitrary errno/exception bytes
            // valid JSON. Bound the receipt without truncating the error latch.
            static const char hex[]="0123456789abcdef";std::string escaped;
            for(size_t i=0;i<std::min(text.size(),size_t(2048));++i) {
                auto ch=static_cast<unsigned char>(text[i]);
                if(ch=='"' || ch=='\\') {escaped.push_back('\\');escaped.push_back(ch);}
                else if(ch<32 || ch>=127) {escaped+="\\u00";escaped.push_back(hex[ch>>4]);escaped.push_back(hex[ch&15]);}
                else escaped.push_back(ch);
            }
            const auto body=std::string("{\"kind\":\"native_service_fatal\",\"pid\":")+std::to_string(getpid())+
                ",\"error\":\""+escaped+"\"}\n";
            out=::open(fatal_path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
            if(out<0)return; // Preserve an existing first-error receipt.
            size_t done=0;
            while(done<body.size()) {
                auto n=::write(out,body.data()+done,body.size()-done);
                if(n<0 && errno==EINTR)continue;
                if(n<=0)break;
                done+=size_t(n);
            }
            ::close(out);
        } catch(...) {if(out>=0)::close(out);}
    }
    void latch_error(const std::string& text) {
        bool first=false;
        {std::lock_guard<std::mutex> lock(telemetry);if(failure.empty()) {failure=text;counters["failures"]++;first=true;}}
        if(first)write_fatal_receipt(text);
        port->cancel();
    }
    std::shared_ptr<Buffer> fetch(const Record& rec) {
        try {return fetch_checked(rec);}
        catch(const Cancelled&) {throw;}
        catch(const std::exception& e) {latch_error(rec.key+": "+e.what());throw;}
        catch(...) {latch_error(rec.key+": unknown native read failure");throw;}
    }
    std::shared_ptr<Buffer> fetch_checked(const Record& rec) {
        check_cancel();guard();auto start=Clock::now();
        struct Active {
            Impl* p;
            explicit Active(Impl* p):p(p){
                std::lock_guard<std::mutex> lock(p->telemetry);p->counters["active_reads"]++;
                p->counters["peak_active_reads"]=std::max(p->counters["peak_active_reads"],p->counters["active_reads"]);
            }
            ~Active(){p->add("active_reads",-1);}
        } active(this);
        auto raw=new Buffer(rec.bytes);staging(rec.bytes);
        auto buffer=std::shared_ptr<Buffer>(raw,[this](Buffer* p){auto n=p->size;delete p;staging(-int64_t(n));});
        add("reads_started");
        int64_t got=0;
        while(got<rec.bytes) {
            check_cancel();ssize_t n=::pread(fd,static_cast<uint8_t*>(buffer->data)+got,rec.bytes-got,rec.file_offset+got);
            if(n<0 && errno==EINTR)continue;
            check(n>0,"Native expert short/failed direct read (errno="+std::to_string(errno)+")");got+=n;
            check(got==rec.bytes || got%ALIGN==0,"Unaligned partial direct read");
        }
        add("read_syscall_seconds",seconds(start));add("bytes_read",got);add("reads");
        unsigned char digest[32];check(hash(static_cast<const unsigned char*>(buffer->data),rec.bytes,digest)!=nullptr,"SHA256 failed");
        static const char hex[]="0123456789abcdef";std::string actual;actual.reserve(64);
        for(auto b:digest) {actual.push_back(hex[b>>4]);actual.push_back(hex[b&15]);}
        check(actual==rec.sha256,"Expert record SHA256 mismatch: "+rec.key);
        for(auto off:rec.markers) {int32_t marker;std::memcpy(&marker,static_cast<uint8_t*>(buffer->data)+off,4);check(marker==-2082680531,"MUL1 marker mismatch: "+rec.key);}
        check_cancel();return buffer;
    }
    auto find_entry(const std::string& key) {return std::find_if(state.entries.begin(),state.entries.end(),[&](const Entry& e){return e.key==key;});}
    void evict(size_t i) {
        auto entry=state.entries.at(i);auto& rec=records.at(entry.key);auto row=table+(rec.layer*384+rec.expert)*13;
        row[0]=1;std::fill(row+1,row+13,0);
        release(state.regions.at(rec.layer),entry.offset);state.resident_bytes-=entry.size;
        state.entries.erase(state.entries.begin()+i);add("evictions");
    }
    void load(const Record& rec,const std::shared_ptr<Buffer>& data,const std::set<std::string>& protected_keys) {
        guard();auto& region=state.regions.at(rec.layer);auto offset=allocate(region,rec.bytes);
        while(offset<0) {
            size_t victim=state.entries.size();std::pair<int64_t,std::string> best;
            for(size_t i=0;i<state.entries.size();++i) {auto& e=state.entries[i];auto& r=records.at(e.key);
                if(r.layer!=rec.layer || protected_keys.count(e.key))continue;
                auto score=std::make_pair(used[r.layer*384+r.expert],e.key);
                if(victim==state.entries.size() || score<best) {victim=i;best=std::move(score);}
            }
            check(victim<state.entries.size(),"No safe native cache region");evict(victim);offset=allocate(region,rec.bytes);
        }
        bool inserted=false;
        try {
            check_cancel();auto start=Clock::now();std::memcpy(arena+offset,data->data,rec.bytes);add("mapped_cpu_copy_seconds",seconds(start));check_cancel();
            Entry entry{rec.key,offset,rec.bytes,state.generation+1};state.entries.push_back(entry);inserted=true;
            state.generation++;state.resident_bytes+=rec.bytes;
            auto row=table+(rec.layer*384+rec.expert)*13;
            for(int i=0;i<3;++i)row[1+i]=rec.bits[i];
            for(int i=0;i<9;++i)row[4+i]=gpu+offset+rec.offsets[i];row[0]=2;
            add("loads");add("misses");
            std::lock_guard<std::mutex> lock(telemetry);
            counters["resident_bytes"]=state.resident_bytes;
            counters["peak_resident_bytes"]=std::max(counters["peak_resident_bytes"],double(state.resident_bytes));
        } catch(...) {if(!inserted)release(region,offset);throw;}
    }
    void transaction(const Request& req) {
        check(req.sequence>state.last_request && req.layer>=0 && req.layer<40 && !port->paused(),"Stale request or invalid decode ownership");
        std::set<std::string> protected_keys;std::vector<const Record*> missing;
        for(auto id:req.ids) {
            auto key=std::to_string(req.layer)+":"+std::to_string(id);auto it=records.find(key);
            if(it!=records.end() && protected_keys.insert(key).second && find_entry(key)==state.entries.end())missing.push_back(&it->second);
        }
        check(!missing.empty(),"Request has no owned missing expert");
        // At most two futures/buffers including the one being copied. The
        // coordinator waits natively; an I/O hang never requires a Python join.
        std::vector<std::future<std::shared_ptr<Buffer>>> jobs(missing.size());size_t next=0;
        auto submit=[&](size_t i){
            auto rec=missing[i];jobs[i]=std::async(std::launch::async,[this,rec]{return fetch(*rec);});
            std::lock_guard<std::mutex> lock(telemetry);counters["pending"]++;
            counters["peak_pending"]=std::max(counters["peak_pending"],counters["pending"]);
        };
        try {
            while(next<missing.size() && next<size_t(slots))submit(next++);
            for(size_t i=0;i<missing.size();++i) {
                auto start=Clock::now();
                while(jobs[i].wait_for(std::chrono::milliseconds(5))!=std::future_status::ready)check_cancel();
                auto data=jobs[i].get();add("mapped_read_wait_seconds",seconds(start));
                load(*missing[i],data,protected_keys);data.reset();add("pending",-1);if(next<missing.size())submit(next++);
            }
        } catch(const Cancelled&) {
            if(!stopping.load(std::memory_order_acquire))
                latch_error("GPU mailbox cancelled while waiting for native I/O: "+std::to_string(port->cancelled()));
            throw;
        }
        catch(const std::exception& e) {
            // Latch/cancel BEFORE future destruction waits for another read.
            latch_error(e.what());throw;
        } catch(...) {
            latch_error("Unknown native load failure");throw;
        }
        for(auto& key:protected_keys)check(find_entry(key)!=state.entries.end(),"Incomplete native expert batch");
        check_cancel();state.last_request=req.sequence;add("requests");add("publications");
        {std::lock_guard<std::mutex> lock(telemetry);counters["sequence"]=req.sequence;}
        // Release acknowledgement is the final publication action. Everything
        // below this point is CPU-private; the GPU may immediately advance.
        port->commit(req.sequence);
    }
    void run() noexcept {
        try {
            while(!stopping.load(std::memory_order_acquire)) {
                if(auto code=port->cancelled()) {
                    if(!stopping.load(std::memory_order_acquire))latch_error("Unexpected GPU mailbox cancellation: "+std::to_string(code));
                    break;
                }
                auto req=port->poll();
                if(req.sequence)transaction(req);
                else std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        } catch(const Cancelled&) {
            // Requested stop/cancellation is distinct from a read/hash failure.
            if(!stopping.load(std::memory_order_acquire))
                latch_error("GPU mailbox cancelled during native miss: "+std::to_string(port->cancelled()));
            else port->cancel();
        } catch(const std::exception& e) {
            latch_error(e.what());
        } catch(...) {
            latch_error("Unknown native service failure");
        }
        {std::lock_guard<std::mutex> lock(telemetry);counters["pending"]=0;}
        {std::lock_guard<std::mutex> lock(lifecycle);done=true;}done_cv.notify_all();
    }
};
Service::Service(std::shared_ptr<MailboxPort> p,void* a,uint64_t g,int64_t c,int64_t* t,int64_t* u,int fd,
    std::vector<Record> records,std::map<int,int64_t> quotas,int64_t reserve,int slots):
    impl(new Impl(std::move(p),a,g,c,t,u,fd,std::move(records),std::move(quotas),reserve,slots)) {}
Service::~Service() {
    // A successfully joined service has no more mailbox work. In particular,
    // destruction must not touch a mailbox its owner has already retired.
    if(impl->thread.joinable() && !stop(100,true)) {
        // I/O cannot safely be forcibly interrupted. Leak the worker's complete
        // CPU state; bridge additionally retains CUDA-backed pointer owners.
        impl->thread.detach();(void)impl.release();
    }
}
void Service::import_state(Snapshot input) {
    auto& p=*impl;std::lock_guard<std::mutex> lock(p.lifecycle);
    check(!p.thread.joinable() && p.done && p.port->paused(),"Import requires stopped service and paused mailbox");
    check(error().empty() && !p.port->cancelled(),"Cannot import after service failure/cancellation");
    check(input.generation>=0 && input.last_request>=0 && input.resident_bytes>=0 && input.resident_bytes<=p.capacity,"Invalid cache counters");
    check(input.regions.size()==p.quotas.size(),"Snapshot partition mismatch");
    int64_t base=0,sum=0;std::set<std::string> keys;std::map<int,std::map<int64_t,int64_t>> occupied;
    for(auto q:p.quotas) {validate_region(input.regions.at(q.first),base,q.second);base+=q.second;}
    for(auto e:input.entries) {
        auto& rec=p.records.at(e.key);check(keys.insert(e.key).second && e.size==rec.bytes && e.generation>0 && e.generation<=input.generation,"Invalid entry identity");
        check(occupied[rec.layer].emplace(e.offset,e.size).second,"Overlapping entry allocation");sum+=e.size;
        auto row=p.table+(rec.layer*384+rec.expert)*13;check(row[0]==2,"Resident table state mismatch");
        for(int i=0;i<3;++i)check(row[1+i]==rec.bits[i],"Resident bits mismatch");
        for(int i=0;i<9;++i)check(uint64_t(row[4+i])==p.gpu+e.offset+rec.offsets[i],"Resident pointer mismatch");
    }
    for(auto q:p.quotas)check(occupied[q.first]==input.regions.at(q.first).allocated,"Entry/region ownership mismatch");
    for(auto& pair:p.records) {auto& r=pair.second;check(p.table[(r.layer*384+r.expert)*13]==(keys.count(r.key)?2:1),"Owned table identity mismatch");}
    check(sum==input.resident_bytes,"Resident accounting mismatch");p.state=std::move(input);p.seeded=true;
    {std::lock_guard<std::mutex> stats_lock(p.telemetry);p.counters["resident_bytes"]=sum;p.counters["peak_resident_bytes"]=std::max(p.counters["peak_resident_bytes"],double(sum));}
}
Snapshot Service::export_state() const {
    auto& p=*impl;std::lock_guard<std::mutex> lock(p.lifecycle);
    check(!p.thread.joinable() && p.done && p.port->paused(),"Export requires stopped service and paused mailbox");
    check(error().empty() && !p.port->cancelled(),"Cannot export failed/cancelled cache as reusable state");return p.state;
}
void Service::start() {
    auto& p=*impl;std::lock_guard<std::mutex> lock(p.lifecycle);
    check(p.seeded && !p.thread.joinable() && p.done && p.port->paused(),"Start requires imported paused ownership");
    check(error().empty() && !p.port->cancelled(),"Cannot restart failed/cancelled native service");
    p.stopping.store(false,std::memory_order_release);p.done=false;p.seeded=false;
    try {p.thread=std::thread([ptr=&p]{ptr->run();});}catch(...) {p.done=true;p.seeded=true;throw;}
}
bool Service::stop(int timeout_ms,bool cancel) {
    auto& p=*impl;check(timeout_ms>=0 && timeout_ms<=60000,"Bounded native join required");
    std::unique_lock<std::mutex> lock(p.lifecycle);
    if(!cancel)check(p.port->paused(),"Normal stop requires GPU drain and mailbox.pause first");
    p.stopping.store(true,std::memory_order_release);
    if(cancel)p.port->cancel();
    if(!p.thread.joinable())return true;
    if(!p.done_cv.wait_for(lock,std::chrono::milliseconds(timeout_ms),[&p]{return p.done;}))return false;
    lock.unlock();p.thread.join();return true;
}
Stats Service::stats() const {auto& p=*impl;std::lock_guard<std::mutex> lock(p.telemetry);return p.counters;}
bool Service::is_alive() const {auto& p=*impl;std::lock_guard<std::mutex> lock(p.lifecycle);return !p.done;}
std::string Service::error() const {auto& p=*impl;std::lock_guard<std::mutex> lock(p.telemetry);return p.failure;}
}  // namespace sage_native
