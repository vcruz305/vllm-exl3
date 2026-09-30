#include "native_service.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace sage_native;
int main() {
    constexpr int64_t page=4096,base=128*page;
    Region r{{{base,12*page}}, {}};
    validate_region(r,base,12*page);
    for(int i=0;i<12;++i)assert(allocate(r,page)==base+i*page);
    assert(allocate(r,page)==-1);
    // Non-zero per-layer bases, unsorted releases, and merged holes must match
    // the Python allocator without moving any still-resident expert pointer.
    release(r,base+3*page);release(r,base+page);release(r,base+2*page);
    validate_region(r,base,12*page);
    assert(allocate(r,3*page)==base+page);
    auto before=r;
    try {release(r,base+2*page);assert(false);}catch(const std::runtime_error&){}
    assert(r.free==before.free && r.allocated==before.allocated);
    for(auto e:before.allocated)release(r,e.first);
    validate_region(r,base,12*page);
    assert(r.free==Extents({{base,12*page}}));
    Region overlap{{{base,12*page}},{{base,page}}};
    try {validate_region(overlap,base,12*page);assert(false);}catch(const std::runtime_error&){}
    Region gap{{{base,page},{base+2*page,10*page}}, {}};
    try {validate_region(gap,base,12*page);assert(false);}catch(const std::runtime_error&){}
    std::cout<<"native region ownership checks passed\n";
}
