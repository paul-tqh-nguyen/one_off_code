
#include "Tibs/TibsDialect.h"
#include "Tibs/TibsOps.h"

using namespace mlir;
using namespace mlir::tibs;

/****************/
/* Tibs dialect */
/****************/

void TibsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tibs/TibsOps.cpp.inc"
      >();
}
