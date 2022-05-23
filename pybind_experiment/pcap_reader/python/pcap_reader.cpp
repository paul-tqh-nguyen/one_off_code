#include "../cpp/include/pcap_reader.hpp"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mcl {

PYBIND11_MODULE(pcap_reader, m) {
  m.doc() = "PCAP -> NumPy library";
    
  py::class_<PCAPReader::PCAPReader>(m, "PCAPReader")
    .def(py::init<>())
    .def("f", []()  -> py::array_t<double> {
		// Allocate and initialize some data; make this big so
		// we can see the impact on the process memory use:
		constexpr size_t size = 100*1000*1000;
		double *foo = new double[size];
		for (size_t i = 0; i < size; i++) {
		  foo[i] = (double) i;
		}

		// Create a Python object that will free the allocated
		// memory when destroyed:
		py::capsule free_when_done(foo, [](void *f) {
						  double *foo = reinterpret_cast<double *>(f);
						  std::cerr << "Element [0] = " << foo[0] << "\n";
						  std::cerr << "freeing memory @ " << f << "\n";
						  delete[] foo;
						});

		return py::array_t<double>(
					   {100, 1000, 1000}, // shape
					   {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
					   foo, // the data pointer
					   free_when_done); // numpy array references this parent
	      })
    .def("get_array", py::overload_cast<>( &PCAPReader::PCAPReader::get_array, py::const_));
}
}
