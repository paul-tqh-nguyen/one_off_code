#ifndef TIBS_TIBSOPS_H
#define TIBS_TIBSOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Tibs/TibsOps.h.inc"

#endif // TIBS_TIBSOPS_H
