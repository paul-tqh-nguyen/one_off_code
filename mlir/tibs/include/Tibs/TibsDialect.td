
#ifndef TIBS_DIALECT
#define TIBS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tibs dialect definition.
//===----------------------------------------------------------------------===//

def Tibs_Dialect : Dialect {
    let name = "tibs";
    let summary = "Tibs MLIR dialect."; // TODO update this documentation
    let description = [{
        Tibs MLIR dialect.
    }];
    let cppNamespace = "::mlir::tibs";
}

//===----------------------------------------------------------------------===//
// Base tibs operation definition.
//===----------------------------------------------------------------------===//

class Tibs_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Tibs_Dialect, mnemonic, traits>;

#endif // TIBS_DIALECT
