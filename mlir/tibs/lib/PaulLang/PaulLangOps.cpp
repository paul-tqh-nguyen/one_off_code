
#include "Tibs/TibsOps.h"
#include "Tibs/TibsDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "Tibs/TibsOps.cpp.inc"

using namespace mlir;
using namespace mlir::tibs;

void mlir::tibs::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
