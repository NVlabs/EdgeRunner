#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <meto/engine_clers.h>
#include <meto/engine_lr.h>
#include <meto/engine_lr_absco.h>

namespace py = pybind11;

PYBIND11_MODULE(_meto, m) {
    // Engine_CLERS
    py::class_<Engine_CLERS>(m, "Engine_CLERS")
        .def(py::init<int, bool>())
        .def("encode", &Engine_CLERS::encode)
        .def("decode", &Engine_CLERS::decode);
  
    // Engine_LR
    py::class_<Engine_LR>(m, "Engine_LR")
        .def(py::init<int, bool>())
        .def("encode", &Engine_LR::encode)
        .def("decode", &Engine_LR::decode);
    
    // Engine_LR_ABSCO
    py::class_<Engine_LR_ABSCO>(m, "Engine_LR_ABSCO")
        .def(py::init<int, bool>())
        .def("encode", &Engine_LR_ABSCO::encode)
        .def("decode", &Engine_LR_ABSCO::decode);
}