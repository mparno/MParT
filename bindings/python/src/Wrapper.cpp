#include "CommonPybindUtilities.h"

using namespace mpart::binding;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);
    MultiIndexWrapper(m);
    MapOptionsWrapper(m);
    ConditionalMapBaseWrapper(m);
    MapFactoryWrapper(m);
    
}