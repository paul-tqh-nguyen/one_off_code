
#include <Python.h>
//#include "../tibs-main/tibs-main.cpp"

static PyObject* func(PyObject* self, PyObject* args) {
  int a, b;
  
  if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
    return NULL;
  }
  
  return Py_BuildValue("i", (a + b) / 2);
}

static PyObject* version(PyObject* self) {
  return Py_BuildValue("s", "Version 0.1");
}
 
static PyMethodDef tibsCodeGenMethods[] = {
    {"func", func, METH_VARARGS, "TODO fill this in."},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef tibsCodeGenerationModule = {
	PyModuleDef_HEAD_INIT,
	"tibs_code_generator",
	"Tibs Code Generation Module",
	-1,
	tibsCodeGenMethods
};

PyMODINIT_FUNC PyInit_tibs_code_generator(void)
{
    return PyModule_Create(&tibsCodeGenerationModule);
}


