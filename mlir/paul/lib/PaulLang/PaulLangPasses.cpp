//===- PaulLangPasses.cpp - PaulLang passes -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PaulLang/PaulLangDialect.h"
#include "PaulLang/PaulLangPasses.h"
#include "PaulLang/PaulLangOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"

#include <iostream>
#include <boost/type_index.hpp>

using namespace mlir;
using namespace mlir::paullang;

//===----------------------------------------------------------------------===//
// PaulLang canonicalization passes.
//===----------------------------------------------------------------------===//

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<mlir::paullang::TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context) : OpRewritePattern<mlir::paullang::TransposeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(mlir::paullang::TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    
    // std::cout << "\n";
    // std::cout << "mlir::LogicalResult matchAndRewrite(mlir::paullang::TransposeOp op, mlir::PatternRewriter &rewriter) const override: " << "\n";
    
    mlir::Value transposeInput = op.getOperand();
    
    mlir::paullang::TransposeOp transposeInputOp = transposeInput.getDefiningOp<mlir::paullang::TransposeOp>();

    // using namespace boost::typeindex;
    // std::cout << "type_id_with_cvr<decltype(op)>().pretty_name(): " << type_id_with_cvr<decltype(op)>().pretty_name() << "\n";
    
    if (!transposeInputOp) {
      // std::cout << "Failure."<< "\n";
      // std::cout << "\n";
      return mlir::failure();
    }
    
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    // std::cout << "Success." << "\n";
    // std::cout << "\n";
    return mlir::success();
  }
};

void mlir::paullang::TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context) {
  // std::cout << "void mlir::paullang::TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context): " << "\n";
  results.insert<SimplifyRedundantTranspose>(context);
}

// TODO get DRRs working
// void mlir::paullang::TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context) {
//   std::cout << "void mlir::paullang::TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context): " << "\n";
//   results.insert<mlir::paullang::TransposeTransposeOptPattern, DoubleTransposeOptPattern, FoldDoubleTransposeOptPattern>(context);
// }

//===----------------------------------------------------------------------===//
// PaulLang->Affine Lowering Helpers
//===----------------------------------------------------------------------===//

static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter) {
  mlir::AllocOp alloc = rewriter.create<mlir::AllocOp>(loc, type);
  
  mlir::Block *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());
  
  mlir::DeallocOp dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  
  return alloc;
}

//===----------------------------------------------------------------------===//
// PaulLang->Affine Constant Operation Lowering Rewrite Pattern
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<mlir::paullang::ConstantOp> {
  using OpRewritePattern<mlir::paullang::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(paullang::ConstantOp op, PatternRewriter &rewriter) const final {
    // std::cout << "Calling matchAndRewrite(paullang::ConstantOp op, PatternRewriter &rewriter): " << "\n";
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    mlir::TensorType tensorType = op.getType().cast<TensorType>();
    MemRefType memRefType = convertTensorToMemRef(tensorType);
    Value alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    llvm::ArrayRef<long int> valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    bool hasRankZero = valueShape.empty();
    if (hasRankZero) {
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    } else {
      for (long int i : llvm::seq<int64_t>(0, *std::max_element(valueShape.begin(), valueShape.end()))) {
    	constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
      }
    }

    SmallVector<Value, 2> indices;
    llvm::mapped_iterator<mlir::DenseElementsAttr::AttributeElementIterator, mlir::FloatAttr (*)(mlir::Attribute), mlir::FloatAttr> valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements =
      [&](uint64_t dimension) {
	if (dimension == valueShape.size()) {
	  mlir::FloatAttr constantOpValue = *valueIt++;
	  mlir::ConstantOp constantOp = rewriter.create<mlir::ConstantOp>(loc, constantOpValue);
	  rewriter.create<AffineStoreOp>(loc, constantOp, alloc, llvm::makeArrayRef(indices));
	  return;
	}

	for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
	  indices.push_back(constantIndices[i]);
	  storeElements(dimension + 1);
	  indices.pop_back();
	}
      };

    storeElements(0);

    rewriter.replaceOp(op, alloc);
    
    // using namespace boost::typeindex;
    // std::cout << "valueIt type: " << type_id_with_cvr<decltype(valueIt)>().pretty_name() << "\n";
    // std::cout << "*valueIt type: " << type_id_with_cvr<decltype(*valueIt)>().pretty_name() << "\n";
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PaulLang->Standard Return Operation Lowering Rewrite Pattern
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<mlir::paullang::ReturnOp> {
  using OpRewritePattern<mlir::paullang::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::paullang::ReturnOp op, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PaulLang->Affine lowering passes.
//===----------------------------------------------------------------------===//

namespace {
  
  struct PaulLangToAffineLoweringPass : public PassWrapper<PaulLangToAffineLoweringPass, FunctionPass> {
    
    void getDependentDialects(DialectRegistry &registry) const override {
      // std::cout << "Calling getDependentDialects(DialectRegistry &registry)" << "\n";
      registry.insert<AffineDialect, StandardOpsDialect>();
    }
  
    void runOnFunction() final;
  };
  
} // end anonymous namespace.

void PaulLangToAffineLoweringPass::runOnFunction() {
  // std::cout << "Calling PaulLangToAffineLoweringPass::runOnFunction()" << "\n";
  mlir::FuncOp function = getFunction();
  
  // using namespace boost::typeindex;
  // std::cout << "type_id_with_cvr<decltype(function.getName())>().pretty_name(): " << type_id_with_cvr<decltype(function.getName())>().pretty_name() << "\n";
  
  if (function.getName() != "paulMLIRFunc") {
    return;
  }
  
  if (function.getNumArguments() || function.getType().getNumResults()) {
    function.emitError("expected '' to have 0 inputs and 0 results");
    return signalPassFailure();
  }

  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<AffineDialect, StandardOpsDialect>();

  target.addIllegalDialect<mlir::paullang::PaulLangDialect>();
  target.addLegalOp<mlir::paullang::PrintOp>();
  //target.addLegalOp<mlir::paullang::ReturnOp>(); // TODO remove this

  OwningRewritePatternList patterns;
  patterns.insert<ConstantOpLowering, ReturnOpLowering >(&getContext()); // TODO add other op lowerings

  mlir::LogicalResult conversionStatus = applyPartialConversion(getFunction(), target, std::move(patterns));
  if (failed(conversionStatus)) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::paullang::createLowerToAffinePass() {
  // std::cout << "Calling mlir::paullang::createLowerToAffinePass(): " << "\n";
  return std::make_unique<PaulLangToAffineLoweringPass>();
}

//===----------------------------------------------------------------------===//
// PaulLang->LLVM Print Operation Lowering Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
  
  class PrintOpLowering : public ConversionPattern {
    
  public:
    explicit PrintOpLowering(MLIRContext *context) : ConversionPattern(mlir::paullang::PrintOp::getOperationName(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override {
      auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
      auto memRefShape = memRefType.getShape();
      auto loc = op->getLoc();

      ModuleOp parentModule = op->getParentOfType<ModuleOp>(); // TODO printout this type to add an explicit namespace

      FlatSymbolRefAttr printfRef = getOrInsertPrintf(rewriter, parentModule);
      mlir::Value formatSpecifierCst = getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
      mlir::Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

      SmallVector<Value, 4> loopIvs;
      for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
	auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0); // TODO add explicit namespace to ConstantIndexOp
	auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]); // TODO add explicit namespace to ConstantIndexOp
	auto step = rewriter.create<ConstantIndexOp>(loc, 1);
	auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
	for (Operation &nested : *loop.getBody()) {
	  rewriter.eraseOp(&nested);
	}
	loopIvs.push_back(loop.getInductionVar());

	rewriter.setInsertionPointToEnd(loop.getBody());

	if (i != e - 1) {
	  rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32), newLineCst);
	}
	rewriter.create<scf::YieldOp>(loc);
	rewriter.setInsertionPointToStart(loop.getBody());
      }
      
      mlir::paullang::PrintOp printOp = cast<mlir::paullang::PrintOp>(op);
      auto elementLoad = rewriter.create<LoadOp>(loc, printOp.input(), loopIvs); // TODO add explicit namespace to LoadOp
      rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32), ArrayRef<Value>({formatSpecifierCst, elementLoad}));
      
      rewriter.eraseOp(op);
      
      return success();
    }

  private:
    
    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module) {
      
      auto *context = module.getContext();
      
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
	return SymbolRefAttr::get("printf", context);
      }

      auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(context);
      auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(context);
      auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy, true);
      
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
      return SymbolRefAttr::get("printf", context);
    }
    
    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module) {
      LLVM::GlobalOp global;
      
      if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
	OpBuilder::InsertionGuard insertGuard(builder);
	builder.setInsertionPointToStart(module.getBody());
	auto type = LLVM::LLVMType::getArrayTy(LLVM::LLVMType::getInt8Ty(builder.getContext()), value.size());
	global = builder.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, builder.getStringAttr(value));
      }
      
      mlir::Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
      mlir::Value cst0 = builder.create<LLVM::ConstantOp>(loc, LLVM::LLVMType::getInt64Ty(builder.getContext()), builder.getIntegerAttr(builder.getIndexType(), 0));
      return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMType::getInt8PtrTy(builder.getContext()), globalPtr, llvm::ArrayRef<Value>({cst0, cst0}));
    }
  };
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PaulLang->LLVM lowering passes.
//===----------------------------------------------------------------------===//

namespace {
  
  struct PaulLangToLLVMLoweringPass : public PassWrapper<PaulLangToLLVMLoweringPass, OperationPass<ModuleOp>> { // TODO add explicit namespace to ModuleOp
    
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    
    void runOnOperation() final;
    
  };
  
} // end anonymous namespace

void PaulLangToLLVMLoweringPass::runOnOperation() {
  
  std::cout << "////PaulLangToLLVMLoweringPass::runOnOperation" << std::endl;
  
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>(); // TODO add explicit namespaces
  
  LLVMTypeConverter typeConverter(&getContext());
  
  OwningRewritePatternList patterns;
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns); // TODO Why doesn't this convert the standard MLIR "func" to LLVM IR?
  // std::cout << "&(typeConverter.getContext()): " << &(typeConverter.getContext()) << "\n";
  // std::cout << "(typeConverter.getDialect()): " << (typeConverter.getDialect()) << "\n";
  // std::cout << "&(typeConverter.getOptions()): " << &(typeConverter.getOptions()) << "\n";
  // std::cout << "typeConverter.getOptions().useBarePtrCallConv: " << typeConverter.getOptions().useBarePtrCallConv << "\n";
  // std::cout << "typeConverter.getOptions().emitCWrappers: " << typeConverter.getOptions().emitCWrappers << "\n";
  // std::cout << "typeConverter.getOptions().indexBitwidth: " << typeConverter.getOptions().indexBitwidth << "\n";
  // std::cout << "typeConverter.getOptions().useAlignedAlloc: " << typeConverter.getOptions().useAlignedAlloc << "\n";
  // //std::cout << "typeConverter.getOptions().dataLayout: " << typeConverter.getOptions().dataLayout << "\n";

  patterns.insert<PrintOpLowering>(&getContext());

  mlir::ModuleOp module = getOperation();
  mlir::LogicalResult fullConversionStatus = mlir::applyFullConversion(module, target, std::move(patterns));

  // std::cout << "BEFORE module->dump()" << std::endl;
  // module->dump();
  
  std::cout << "////failed(fullConversionStatus): " << failed(fullConversionStatus) << "\n";
  
  if (failed(fullConversionStatus)) {
    signalPassFailure();
  }
  
  // std::cout << "AFTER module->dump()" << std::endl;
  // module->dump();
  
  std::cout << "\n\n\n" << std::endl;
}

std::unique_ptr<mlir::Pass> mlir::paullang::createLowerToLLVMPass() {
  return std::make_unique<PaulLangToLLVMLoweringPass>();
}
