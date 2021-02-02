//===- PaulLangOps.cpp - PaulLang dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PaulLang/PaulLangOps.h"
#include "PaulLang/PaulLangDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "PaulLang/PaulLangOps.cpp.inc"

using namespace mlir;
using namespace mlir::paullang;

void mlir::paullang::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
