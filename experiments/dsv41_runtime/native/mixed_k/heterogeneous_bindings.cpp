#include <torch/extension.h>
#include "heterogeneous_moe.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe", &sage_heterogeneous_moe,
          "Internal packed MUL1 mixed-K expert dispatch; caller validates descriptor values and pointer leases");
}
