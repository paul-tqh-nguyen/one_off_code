//===- PauLangPasses.h - PauLang dialect passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PAULLANG_PAULLANGPASSES_H
#define PAULLANG_PAULLANGPASSES_H

namespace mlir {
  class Pass; // TODO why can't we use #include "mlir/Pass/Pass.h" ?
  
  namespace paullang {
    std::unique_ptr<Pass> createLowerToAffinePass();
    std::unique_ptr<Pass> createLowerToLLVMPass();
  }
  
}

#endif // PAULLANG_PAULLANGPASSES_H
