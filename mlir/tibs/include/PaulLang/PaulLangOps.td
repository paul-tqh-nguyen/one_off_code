
#ifndef TIBS_OPS
#define TIBS_OPS

include "TibsDialect.td"
include "TibsPasses.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tibs operations.
//===----------------------------------------------------------------------===//

def Tibs_FooOp : Tibs_Op<"foo", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "constant";
    let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = tibs.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def ConstantOp : Tibs_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = tibs.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];
  
  let arguments = (ins F64ElementsAttr:$value);
  
  let results = (outs F64Tensor);
  
  let builders = [
    OpBuilderDAG<(ins "double":$value)>
  ];

}

def PrintOp : Tibs_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // let arguments = (ins F64Tensor:$input);
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReturnOp : Tibs_Op<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        tibs.return %0 : tensor<2xf64>
      }
    ```
  }];

  let arguments = (ins Variadic<F64Tensor>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilderDAG<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def TransposeOp : Tibs_Op<"transpose", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "transpose";
  let description = [{
  Transpose description.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
  
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}

#endif // TIBS_OPS
