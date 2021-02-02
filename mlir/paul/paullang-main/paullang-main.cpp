//===- paullang-main.cpp ----------------------------------------*- C++ -*-===//
//
// Main driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

//#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "PaulLang/PaulLangDialect.h"
#include "PaulLang/PaulLangOps.h"
#include "PaulLang/PaulLangPasses.h"

#include <iostream>
#include <boost/type_index.hpp>
#include <tuple>

void generateModule(mlir::MLIRContext &context, mlir::ModuleOp &theModule) {
  
  // Misc. MLIR Initializations
  // mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::paullang::PaulLangDialect>();
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  
  // Create an MLIR module
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  // mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  
  // Create a location
  std::string fileName = "/tmp/non_existant_file.fake";
  int lineNumber = 12;
  int columnNumber = 34;
  mlir::Location location = builder.getFileLineColLoc(builder.getIdentifier(fileName), lineNumber, columnNumber);
  std::cout << "location.dump(): \n";
  location.dump();
  std::cout << "\n";

  /* // This function causes problems because the tutorial doesn't go over how to lower function args
  { // transpose_transpose function
    
    // Create empty function
    unsigned long int numberOfArgs = 1;
    llvm::ArrayRef<int64_t> inputArrayShape({2,3});
    mlir::RankedTensorType inputTensorType = mlir::RankedTensorType::get(inputArrayShape, builder.getF64Type());
    llvm::SmallVector<mlir::Type, 4> arg_types(numberOfArgs, inputTensorType); // TODO Figure out what the 4 denotes
    mlir::FunctionType func_type = builder.getFunctionType(arg_types, llvm::None);
    llvm::StringRef funcName("transpose_transpose");
    mlir::FuncOp funcOp = mlir::FuncOp::create(location, funcName, func_type);

    // Add function body
    mlir::Block &entryBlock = *funcOp.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Add first transpose operation
    llvm::MutableArrayRef<mlir::BlockArgument> inputArgs = entryBlock.getArguments();
    mlir::BlockArgument inputArg = inputArgs[0];
    mlir::Value firstTransposeOperation = builder.create<mlir::paullang::TransposeOp>(location, inputArg);

    // Add second transpose operation
    mlir::Value secondTransposeOperation = builder.create<mlir::paullang::TransposeOp>(location, firstTransposeOperation);
    
    // Add return statement to function body
    mlir::paullang::ReturnOp returnOperation;
    builder.create<mlir::paullang::ReturnOp>(location, secondTransposeOperation);
  
    // Add functions to module
    theModule.push_back(funcOp);
  }
  //*/

  { // Main Function
    
    // Create empty function
    unsigned long int numberOfArgs = 0;
    llvm::ArrayRef<int64_t> inputArrayShape({2,3});
    mlir::RankedTensorType inputTensorType = mlir::RankedTensorType::get(inputArrayShape, builder.getF64Type());
    llvm::SmallVector<mlir::Type, 4> arg_types(numberOfArgs, inputTensorType); // TODO Figure out what the 4 denotes
    mlir::FunctionType func_type = builder.getFunctionType(arg_types, llvm::None);
    llvm::StringRef funcName("paulMLIRFunc");
    mlir::FuncOp funcOp = mlir::FuncOp::create(location, funcName, func_type);

    // Add function body
    mlir::Block &entryBlock = *funcOp.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Add constant array initialization to function body
    std::vector<double> constantData = {1, 2, 3, 4, 5, 6};
    llvm::ArrayRef<int64_t> constantArrayShape({2,3});
    mlir::Type constantElementType = builder.getF64Type();
    mlir::RankedTensorType constantDataType = mlir::RankedTensorType::get(constantArrayShape, constantElementType);
    mlir::DenseElementsAttr constantDataAttribute = mlir::DenseElementsAttr::get(constantDataType, llvm::makeArrayRef(constantData));
    mlir::RankedTensorType constantTensorType = mlir::RankedTensorType::get(constantArrayShape, builder.getF64Type());
    mlir::Value constantOperation = builder.create<mlir::paullang::ConstantOp>(location, constantTensorType, constantDataAttribute);

    // Add print to function body
    builder.create<mlir::paullang::PrintOp>(location, constantOperation);

    // Add emtpy return statement to function body
    mlir::paullang::ReturnOp returnOperation;
    builder.create<mlir::paullang::ReturnOp>(location);
  
    // Add function to module
    theModule.push_back(funcOp);
  }

  return;
}

/*
void runPassesAndThenPartiallyLower() {
  
  std::cout << "===== runPassesAndThenPartiallyLower start =====" << "\n\n" << "\n\n";
  
  mlir::MLIRContext context;
  mlir::ModuleOp theModule;
  generateModule(context, theModule);

  // Dump Original MLIR
  std::cout << "Original MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();

  // Pass Application Utilities
  mlir::PassManager pm(&context);
  std::function<void()> runPassManager =
    [&]() {
      mlir::LogicalResult pmRunStatus = pm.run(theModule);
      if (mlir::failed(pmRunStatus)) {
	std::cout << "Pass manager run failed." << "\n";
      }
      return;
    };
  
  // Apply canonicalization passes
  pm.addPass(mlir::createCanonicalizerPass());
  runPassManager();
  std::cout << "\n";
  std::cout << "Canonicalized MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();
  std::cout << "\n";

  // Lower to Affine Dialect
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(mlir::paullang::createLowerToAffinePass());
  // optPM.addPass(mlir::createCanonicalizerPass());
  runPassManager();
  std::cout << "\n";
  std::cout << "Lowered (without second canonicalization) MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();
  std::cout << "\n";

  // Second Pass
  runPassManager();
  std::cout << "\n";
  std::cout << "Second Canonicalization Pass MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();
  std::cout << "\n";
  
  std::cout << "===== runPassesAndThenPartiallyLower end =====" << "\n\n" << "\n\n";
  
  return;
}
//*/

void runAllPasses() {
  
  std::cout << "===== runAllPasses start =====" << "\n\n" << "\n\n";
  
  mlir::MLIRContext context;
  mlir::ModuleOp theModule;
  generateModule(context, theModule);

  // Original MLIR
  std::cout << "Original MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();

  // Pass Application Utilities
  mlir::PassManager pm(&context);
  std::function<void()> runPassManager =
    [&]() {
      mlir::LogicalResult pmRunStatus = pm.run(theModule);
      if (mlir::failed(pmRunStatus)) {
	std::cout << "Pass manager run failed." << "\n";
      }
      return;
    };
  
  // Apply canonicalization passes
  pm.addPass(mlir::createCanonicalizerPass());
  // Lower to Affine Dialect
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(mlir::paullang::createLowerToAffinePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  // Lower to LLVM
  pm.addPass(mlir::paullang::createLowerToLLVMPass());

  // Final MLIR
  runPassManager();
  std::cout << "\n";
  std::cout << "Final MLIR:" << "\n";
  std::cout << "\n";
  theModule->dump();
  std::cout << "\n";
  
  std::cout << "===== runAllPasses end =====" << "\n\n" << "\n\n";
  
  return;
}

int main(int argc, char **argv) {

  // runPassesAndThenPartiallyLower();
  runAllPasses();
  
  // using namespace boost::typeindex;
  // std::cout << "type_id_with_cvr<decltype(theModule)>().pretty_name(): " << type_id_with_cvr<decltype(theModule)>().pretty_name() << "\n";

  return 0;
}
