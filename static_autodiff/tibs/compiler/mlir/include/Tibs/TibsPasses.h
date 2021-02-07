
#ifndef TIBS_TIBSPASSES_H
#define TIBS_TIBSPASSES_H

namespace mlir {
  class Pass; // TODO why can't we use #include "mlir/Pass/Pass.h" ?
  
  namespace tibs {
    std::unique_ptr<Pass> createLowerToAffinePass();
    std::unique_ptr<Pass> createLowerToLLVMPass();
  }
  
}

#endif // TIBS_TIBSPASSES_H
