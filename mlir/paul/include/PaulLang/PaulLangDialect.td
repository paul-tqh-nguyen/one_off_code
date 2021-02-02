//===- PaulLangDialect.td - PaulLang dialect ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PAULLANG_DIALECT
#define PAULLANG_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PaulLang dialect definition.
//===----------------------------------------------------------------------===//

def PaulLang_Dialect : Dialect {
    let name = "paullang";
    let summary = "An out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::paullang";
}

//===----------------------------------------------------------------------===//
// Base paullang operation definition.
//===----------------------------------------------------------------------===//

class PaulLang_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<PaulLang_Dialect, mnemonic, traits>;

#endif // PAULLANG_DIALECT
