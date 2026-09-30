// Persistent single-stream, multi-layer miss mailbox. CPU poll/commit/cancel
// perform no CUDA operations. All mapped allocations outlive stream completion.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda/atomic>
#include <chrono>
#include <thread>
#include <memory>
#include <map>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include "native_service.h"
#include <pybind11/stl.h>

struct Allocation {
    void* cpu=nullptr; void* gpu=nullptr;
    explicit Allocation(size_t bytes) {
        C10_CUDA_CHECK(cudaHostAlloc(&cpu,bytes,cudaHostAllocMapped));
        auto e=cudaHostGetDevicePointer(&gpu,cpu,0);
        if(e!=cudaSuccess) {cudaFreeHost(cpu); cpu=nullptr; C10_CUDA_CHECK(e);}
    }
    ~Allocation() {if(cpu) cudaFreeHost(cpu);}
};

std::vector<at::Tensor> allocate(int64_t bytes) {
    TORCH_CHECK(bytes>0 && bytes<=80LL*1024*1024*1024 && bytes%4096==0,
                "Aligned allocation bounded by the admitted80GiB cache budget");
    int device; C10_CUDA_CHECK(cudaGetDevice(&device));
    auto a=std::make_shared<Allocation>(bytes);
    return {at::from_blob(a->cpu,{bytes},[a](void*){},at::TensorOptions().dtype(at::kByte)),
        at::from_blob(a->gpu,{bytes},[a](void*){},at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA,device)))};
}

// Candidate only. Keep the request/table atomics in the qualified pinned
// allocation; this alternative applies only to the large expert-byte arena.
// System allocation uses CPU page tables when hardware pageable access is
// supported. It is neither cudaMallocManaged nor a pinned CUDA allocation.
struct SystemAllocation {
    void* ptr=nullptr;
    size_t bytes;
    explicit SystemAllocation(size_t count): bytes(count) {
        ptr=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        TORCH_CHECK(ptr!=MAP_FAILED,"System arena mmap failed: ",std::strerror(errno));
    }
    ~SystemAllocation() {if(ptr && ptr!=MAP_FAILED) munmap(ptr,bytes);}
};

std::map<std::string,int64_t> managed_capabilities() {
    int device; C10_CUDA_CHECK(cudaGetDevice(&device));
    std::map<std::string,int64_t> result;
    for(auto attribute: std::vector<std::pair<std::string,cudaDeviceAttr>>{
        {"concurrent_managed",cudaDevAttrConcurrentManagedAccess},
        {"pageable_access",cudaDevAttrPageableMemoryAccess},
        {"host_page_tables",cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"direct_managed_host_access",cudaDevAttrDirectManagedMemAccessFromHost},
        {"host_native_atomics",cudaDevAttrHostNativeAtomicSupported}}) {
        int value; C10_CUDA_CHECK(cudaDeviceGetAttribute(&value,attribute.second,device));
        result[attribute.first]=value;
    }
    return result;
}

std::vector<at::Tensor> allocate_system(int64_t bytes) {
    TORCH_CHECK(bytes>0 && bytes<=80LL*1024*1024*1024 && bytes%4096==0,"Bounded aligned system arena required");
    const auto caps=managed_capabilities();
    TORCH_CHECK(caps.at("concurrent_managed")==1 && caps.at("host_page_tables")==1 &&
                caps.at("pageable_access")==1 &&
                caps.at("host_native_atomics")==1,"Hardware-coherent pageable system access required; no software-coherence fallback");
    int device; C10_CUDA_CHECK(cudaGetDevice(&device));
    auto a=std::make_shared<SystemAllocation>(bytes);
    // The standard pointer-device inference rejects unregistered host memory.
    // These aliases are valid only after the system-page-table checks above.
    // Specify the consuming device without changing or registering the mmap.
    const at::Device gpu_device(at::kCUDA,device);
    return {at::from_blob(a->ptr,{bytes},[a](void*){},at::TensorOptions().dtype(at::kByte).device(at::kCPU)),
        at::from_blob(a->ptr,{bytes},[a](void*){},
            at::TensorOptions().dtype(at::kByte).device(gpu_device),gpu_device)};
}

struct alignas(64) State {
    cuda::atomic<int64_t,cuda::thread_scope_system> request{0},ack{0};
    cuda::atomic<int,cuda::thread_scope_system> cancelled{0};
    int64_t step=0, layer=0, ids[6]{};
    cuda::atomic<int,cuda::thread_scope_system> cpu_mode{1};
};

__device__ void abort_request(State* s,int code) {
    int clear=0;
    s->cancelled.compare_exchange_strong(clear,code,cuda::memory_order_acq_rel);
    asm volatile("trap;");
}

__global__ void gate(State* s,const int64_t* ids,int64_t* table,
                     int64_t* used,int layer,unsigned long long timeout_cycles) {
    if(threadIdx.x) return;
    if(s->cancelled.load(cuda::memory_order_acquire)) {abort_request(s,10);return;}
    if(s->cpu_mode.load(cuda::memory_order_acquire)) {abort_request(s,15);return;}
    table+=layer*384*13; used+=layer*384;
    s->layer=layer;
    const int64_t step=++s->step;
    bool missing=false;
    for(int i=0;i<6;++i) {
        const int64_t id=ids[i];
        s->ids[i]=id;
        // Preserve the qualified routing fixture's invalid/unowned-ID mask.
        if(id>=0 && id<384 && table[id*13]) {
            used[id]=step;
            if(table[id*13]==1) missing=true;
        }
    }
    if(!missing) return;
    // Prior expert kernels on this same stream have completed. No table or
    // expert bytes may change until this release has been observed by the CPU.
    s->request.store(step,cuda::memory_order_release);
    const auto start=clock64();
    while(s->ack.load(cuda::memory_order_acquire)!=step) {
        if(s->cancelled.load(cuda::memory_order_acquire)) {abort_request(s,11);return;}
        if(clock64()-start>timeout_cycles) {abort_request(s,12);return;}
        __nanosleep(256);
    }
    if(s->cancelled.load(cuda::memory_order_acquire)) {abort_request(s,13);return;}
    for(int i=0;i<6;++i) {
        const int64_t id=s->ids[i];
        if(id>=0 && id<384 && table[id*13]==1) {abort_request(s,14);return;}
    }
    // Subsequent kernels on this stream observe the complete CPU publication.
    __threadfence_system();
}

class Mailbox : public sage_native::MailboxPort {
    std::shared_ptr<Allocation> allocation;
    State *host,*gpu;
    at::Tensor table_cpu,table_gpu,used_cpu,used_gpu;
    cudaStream_t owner;
    int device;
    bool closed=false;
    // Main-thread lifecycle only. Workers access only the existing State atomics.
    bool native_attached=false,native_active=false;
    unsigned long long timeout;
public:
    Mailbox(at::Tensor tc,at::Tensor tg,at::Tensor uc,at::Tensor ug,int64_t cycles):
        table_cpu(tc),table_gpu(tg),used_cpu(uc),used_gpu(ug) {
        TORCH_CHECK(cycles>=1000000 && cycles<=10000000000LL,"Bounded GPU deadline required");
        timeout=cycles;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp p; C10_CUDA_CHECK(cudaGetDeviceProperties(&p,device));
        int native=0; C10_CUDA_CHECK(cudaDeviceGetAttribute(&native,cudaDevAttrHostNativeAtomicSupported,device));
        TORCH_CHECK(p.major==12 && p.minor==1 && p.integrated && p.canMapHostMemory && native,
                    "Qualified coherent GB10 required");
        TORCH_CHECK(!tc.is_cuda() && !uc.is_cuda() && tg.is_cuda() && ug.is_cuda(),"Mapped aliases required");
        TORCH_CHECK(tg.get_device()==device && ug.get_device()==device,"Metadata must use the owning device");
        TORCH_CHECK(tc.scalar_type()==at::kLong && tg.scalar_type()==at::kLong && uc.scalar_type()==at::kLong && ug.scalar_type()==at::kLong,"INT64 metadata required");
        TORCH_CHECK(tc.numel()==40*384*13 && tg.numel()==40*384*13 && uc.numel()==40*384 && ug.numel()==40*384,"Fixed metadata geometry required");
        TORCH_CHECK(tc.is_contiguous() && tg.is_contiguous() && uc.is_contiguous() && ug.is_contiguous(),"Contiguous metadata required");
        void* alias; C10_CUDA_CHECK(cudaHostGetDevicePointer(&alias,tc.data_ptr(),0));
        TORCH_CHECK(alias==tg.data_ptr(),"Table is not the same mapped allocation");
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&alias,uc.data_ptr(),0));
        TORCH_CHECK(alias==ug.data_ptr(),"Use stamps are not the same mapped allocation");
        owner=at::cuda::getCurrentCUDAStream(device);
        allocation=std::make_shared<Allocation>(sizeof(State));
        host=new(allocation->cpu) State; gpu=static_cast<State*>(allocation->gpu);
    }
    void enqueue(int layer,at::Tensor ids) {
        TORCH_CHECK(!closed && layer>=0 && layer<40,"Live owned layer required");
        TORCH_CHECK(!native_attached || native_active,"Native decode service must own the mailbox before enqueue");
        TORCH_CHECK(!host->cpu_mode.load(cuda::memory_order_acquire),"CPU cache ownership must end before GPU dispatch");
        TORCH_CHECK(at::cuda::getCurrentCUDAStream(device)==owner,"One owning CUDA stream required");
        TORCH_CHECK(ids.is_cuda() && ids.get_device()==device && ids.scalar_type()==at::kLong && ids.numel()==6 && ids.is_contiguous(),"Six contiguous GPU INT64 IDs required");
        gate<<<1,32,0,owner>>>(gpu,ids.data_ptr<int64_t>(),table_gpu.data_ptr<int64_t>(),used_gpu.data_ptr<int64_t>(),layer,timeout);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    std::vector<int64_t> poll(int milliseconds) {
        TORCH_CHECK(!closed && milliseconds>=0 && milliseconds<=100,"Bounded live poll required");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(milliseconds);
        do {
            if(host->cancelled.load(cuda::memory_order_acquire)) return {-1};
            const auto req=host->request.load(cuda::memory_order_acquire);
            if(req!=host->ack.load(cuda::memory_order_acquire))
                return {req,host->layer,host->ids[0],host->ids[1],host->ids[2],host->ids[3],host->ids[4],host->ids[5]};
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } while(std::chrono::steady_clock::now()<deadline);
        return {};
    }
    sage_native::Request poll() const override {
        sage_native::Request result;
        if(closed || host->cancelled.load(cuda::memory_order_acquire)) return result;
        const auto req=host->request.load(cuda::memory_order_acquire);
        if(req!=host->ack.load(cuda::memory_order_acquire)) {
            result.sequence=req; result.layer=host->layer;
            for(int i=0;i<6;++i) result.ids[i]=host->ids[i];
        }
        return result;
    }
    bool paused() const override {return !closed && host->cpu_mode.load(cuda::memory_order_acquire);}
    int cancelled() const override {return closed ? 1 : host->cancelled.load(cuda::memory_order_acquire);}
    int64_t* native_table() {return table_cpu.data_ptr<int64_t>();}
    int64_t* native_used() {return used_cpu.data_ptr<int64_t>();}
    void attach_native() {TORCH_CHECK(!closed && paused() && !native_attached,"One native service per paused mailbox");native_attached=true;}
    void detach_native() {TORCH_CHECK(!native_active,"Cannot detach active native service");native_attached=false;}
    void set_native_active(bool value) {native_active=value;}
    void commit(int64_t seq) override {
        TORCH_CHECK(!closed && !host->cancelled.load(cuda::memory_order_acquire),"Cancelled publication");
        TORCH_CHECK(seq>0 && host->request.load(cuda::memory_order_acquire)==seq && host->ack.load(cuda::memory_order_acquire)!=seq,"Stale or duplicate publication");
        host->ack.store(seq,cuda::memory_order_release);
    }
    int64_t pause() {
        TORCH_CHECK(!closed,"Mailbox closed");
        C10_CUDA_CHECK(cudaStreamSynchronize(owner));
        TORCH_CHECK(host->request.load(cuda::memory_order_acquire)==host->ack.load(cuda::memory_order_acquire),"Unfinished GPU request");
        host->cpu_mode.store(1,cuda::memory_order_release);
        return host->step;
    }
    void resume(int64_t epoch) {
        TORCH_CHECK(!closed && host->cpu_mode.load(cuda::memory_order_acquire),"CPU ownership required");
        TORCH_CHECK(!native_attached || native_active,"Start native service before resuming mailbox");
        C10_CUDA_CHECK(cudaStreamSynchronize(owner));
        TORCH_CHECK(epoch>=host->step && epoch<1000000000000000000LL,"Monotonic host use epoch required");
        TORCH_CHECK(!host->cancelled.load(cuda::memory_order_acquire),"Cancelled service");
        host->step=epoch;
        host->cpu_mode.store(0,cuda::memory_order_release);
    }
    void rebind() {
        TORCH_CHECK(!closed && host->cpu_mode.load(cuda::memory_order_acquire),"Paused live mailbox required for stream transfer");
        TORCH_CHECK(!native_active,"Join native ownership before stream transfer");
        C10_CUDA_CHECK(cudaStreamSynchronize(owner));
        TORCH_CHECK(host->request.load(cuda::memory_order_acquire)==host->ack.load(cuda::memory_order_acquire),"Unfinished request during stream transfer");
        int current_device; C10_CUDA_CHECK(cudaGetDevice(&current_device));
        TORCH_CHECK(current_device==device,"Stream transfer changed CUDA device");
        owner=at::cuda::getCurrentCUDAStream(device);
    }
    void cancel() override {
        if(!closed) {
            int clear=0;
            host->cancelled.compare_exchange_strong(clear,1,cuda::memory_order_acq_rel);
        }
    }
    int cancellation_code() {return closed ? -1 : host->cancelled.load(cuda::memory_order_acquire);}
    void close() {
        if(closed) return;
        TORCH_CHECK(!native_attached,"Retire native service before releasing mailbox storage");
        // CPU producer must already be joined. An error here deliberately keeps
        // storage alive; a failed CUDA context is terminated by the owned worker.
        C10_CUDA_CHECK(cudaStreamSynchronize(owner));
        closed=true; host->~State(); allocation.reset();
        table_cpu=at::Tensor();table_gpu=at::Tensor();
        used_cpu=at::Tensor();used_gpu=at::Tensor();
    }
    ~Mailbox() {
        // Python service explicitly closes after joining. As a final guard,
        // never free mailbox storage while a gate may still be running.
        if(!closed) {cancel(); cudaStreamSynchronize(owner);}
    }
};

namespace py=pybind11;
class NativeService {
    std::shared_ptr<Mailbox> mailbox;
    at::Tensor cpu_arena,gpu_arena;
    std::unique_ptr<sage_native::Service> service;
    bool closed=false;
    sage_native::Stats final_stats;
    std::string final_error;
public:
    NativeService(std::shared_ptr<Mailbox> mb,at::Tensor cpu,at::Tensor gpu,int fd,
                  py::list input,std::map<int,int64_t> quotas,int64_t reserve,int slots):
                  mailbox(std::move(mb)),cpu_arena(cpu),gpu_arena(gpu) {
        TORCH_CHECK(mailbox && mailbox->paused(),"Native service setup requires paused mailbox");
        TORCH_CHECK(!cpu.is_cuda() && gpu.is_cuda() && cpu.scalar_type()==at::kByte && gpu.scalar_type()==at::kByte,
                    "CPU/GPU byte arena aliases required");
        TORCH_CHECK(cpu.is_contiguous() && gpu.is_contiguous() && cpu.numel()==gpu.numel() && cpu.data_ptr()==gpu.data_ptr(),
                    "Exact same system arena required");
        std::vector<sage_native::Record> records;
        for(auto item:input) {
            auto row=py::cast<py::tuple>(item);TORCH_CHECK(row.size()==7,"Record descriptor must have seven fields");
            sage_native::Record r;
            r.key=row[0].cast<std::string>();r.file_offset=row[1].cast<int64_t>();r.bytes=row[2].cast<int64_t>();
            r.sha256=row[3].cast<std::string>();r.bits=row[4].cast<std::array<int64_t,3>>();
            r.offsets=row[5].cast<std::array<int64_t,9>>();r.markers=row[6].cast<std::array<int64_t,3>>();
            records.push_back(std::move(r));
        }
        mailbox->attach_native();
        try {
            service=std::make_unique<sage_native::Service>(mailbox,cpu.data_ptr(),reinterpret_cast<uint64_t>(gpu.data_ptr()),
                cpu.numel(),mailbox->native_table(),mailbox->native_used(),fd,std::move(records),std::move(quotas),reserve,slots);
        } catch(...) {mailbox->detach_native();throw;}
    }
    ~NativeService() {
        if(!service)return;
        bool joined=false;
        try {joined=service->stop(100,true);}catch(...) {}
        if(!joined) {
            // Intentionally unreclaimed until process exit. A detached/hung
            // syscall may still read/copy into these exact mapped allocations.
            struct Retained {std::shared_ptr<Mailbox> mb;at::Tensor cpu,gpu;};
            (void)new Retained{mailbox,cpu_arena,gpu_arena};
        } else {
            mailbox->set_native_active(false);mailbox->detach_native();
        }
        service.reset();  // CPU-only destructor performs another bounded join.
    }
    void import_state(py::list entries,py::dict regions,int64_t generation,int64_t resident,int64_t last_request) {
        TORCH_CHECK(!closed,"Native service closed");sage_native::Snapshot state;
        state.generation=generation;state.resident_bytes=resident;state.last_request=last_request;
        for(auto item:entries) {
            auto row=py::cast<py::tuple>(item);TORCH_CHECK(row.size()==4,"Entry must have four fields");
            state.entries.push_back({row[0].cast<std::string>(),row[1].cast<int64_t>(),row[2].cast<int64_t>(),row[3].cast<int64_t>()});
        }
        for(auto item:regions) {
            auto row=py::cast<py::dict>(item.second);sage_native::Region region;
            region.free=row["free"].cast<sage_native::Extents>();
            for(auto extent:row["allocated"].cast<sage_native::Extents>())
                TORCH_CHECK(region.allocated.emplace(extent).second,"Duplicate allocated extent");
            state.regions.emplace(py::cast<int>(item.first),std::move(region));
        }
        service->import_state(std::move(state));
    }
    py::dict export_state() {
        TORCH_CHECK(!closed,"Native service closed");auto s=service->export_state();py::dict result,regions;py::list entries;
        for(auto& e:s.entries)entries.append(py::make_tuple(e.key,e.offset,e.size,e.generation));
        for(auto& pair:s.regions) {
            py::dict r;r["free"]=py::cast(pair.second.free);sage_native::Extents allocated;
            for(auto e:pair.second.allocated)allocated.push_back(e);
            r["allocated"]=py::cast(allocated);regions[py::int_(pair.first)]=r;
        }
        result["entries"]=entries;result["regions"]=regions;result["generation"]=s.generation;
        result["resident_bytes"]=s.resident_bytes;result["last_request"]=s.last_request;result["stats"]=py::cast(service->stats());
        return result;
    }
    void start() {
        TORCH_CHECK(!closed,"Native service closed");service->start();mailbox->set_native_active(true);
    }
    bool stop(int timeout_ms,bool cancel) {
        if(closed)return true;
        bool joined=service->stop(timeout_ms,cancel);if(joined)mailbox->set_native_active(false);return joined;
    }
    bool close(int timeout_ms) {
        if(closed)return true;
        if(!stop(timeout_ms,true))return false;
        // Retire the core while State is still alive. Python is now permitted
        // to call mailbox.close even if a worker-view retains this wrapper.
        final_stats=service->stats();final_error=service->error();
        service.reset();mailbox->detach_native();closed=true;return true;
    }
    bool is_alive() const {return closed ? false : service->is_alive();}
    sage_native::Stats stats() const {return closed ? final_stats : service->stats();}
    std::string error() const {return closed ? final_error : service->error();}
};
void bind_native_service(py::module_& m) {
    py::class_<NativeService>(m,"NativeService")
        .def(py::init<std::shared_ptr<Mailbox>,at::Tensor,at::Tensor,int,py::list,std::map<int,int64_t>,int64_t,int>(),
             py::arg("mailbox"),py::arg("cpu_arena"),py::arg("gpu_arena"),py::arg("fd"),py::arg("records"),
             py::arg("layer_quotas"),py::arg("reserve_bytes")=8LL*1024*1024*1024,py::arg("slots")=2)
        .def("import_state",&NativeService::import_state,py::arg("entries"),py::arg("regions"),py::arg("generation"),py::arg("resident_bytes"),py::arg("last_request"))
        .def("export_state",&NativeService::export_state)
        .def("start",&NativeService::start,py::call_guard<py::gil_scoped_release>())
        .def("stop",&NativeService::stop,py::arg("timeout_ms")=15000,py::arg("cancel")=false,py::call_guard<py::gil_scoped_release>())
        .def("close",&NativeService::close,py::arg("timeout_ms")=15000,py::call_guard<py::gil_scoped_release>())
        .def("is_alive",&NativeService::is_alive)
        .def("stats",&NativeService::stats)
        .def("error",&NativeService::error);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("allocate",&allocate);
    m.def("managed_capabilities",&managed_capabilities);
    m.def("allocate_system",&allocate_system);
    pybind11::class_<Mailbox,std::shared_ptr<Mailbox>>(m,"Mailbox")
        .def(pybind11::init<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>())
        .def("enqueue",&Mailbox::enqueue,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("poll",static_cast<std::vector<int64_t>(Mailbox::*)(int)>(&Mailbox::poll),pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("commit",&Mailbox::commit)
        .def("pause",&Mailbox::pause,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("resume",&Mailbox::resume,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("rebind",&Mailbox::rebind,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("cancel",&Mailbox::cancel)
        .def("cancellation_code",&Mailbox::cancellation_code)
        .def("close",&Mailbox::close,pybind11::call_guard<pybind11::gil_scoped_release>());
    bind_native_service(m);
}
