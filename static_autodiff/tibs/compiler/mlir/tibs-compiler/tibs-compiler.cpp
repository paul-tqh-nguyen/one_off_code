
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Support/TargetSelect.h"

#include "Tibs/TibsDialect.h"
#include "Tibs/TibsOps.h"
#include "Tibs/TibsPasses.h"

#include <iostream>
#include <boost/type_index.hpp>
#include <tuple>

// TODO remove unneeded libraries above

/*********************/
/* C++ Functionality */
/*******((************/

// TODO rename this section ^^

class ModuleGenerator {

private:
  void intializePasses() {
    // canonicalization passes
    pm.addPass(mlir::createCanonicalizerPass());
    // Lower to Affine Dialect
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::tibs::createLowerToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    // Lower to LLVM
    pm.addPass(mlir::tibs::createLowerToLLVMPass());
    return;
  }
  
public:

  // TODO make these private
  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  mlir::PassManager pm;
  
  ModuleGenerator()
    :
    context(),
    builder(&context),
    theModule(mlir::ModuleOp::create(builder.getUnknownLoc())),
    pm(&context)
  {
    context.getOrLoadDialect<mlir::tibs::TibsDialect>();
    intializePasses();
    return;
  }

  void dumpModule() {
    theModule->dump();
    return;
  }

  void runPassManager () {
    mlir::LogicalResult pmRunStatus = pm.run(theModule);
    if (mlir::failed(pmRunStatus)) {
      std::cout << "Pass manager run failed." << std::endl;
    }
    return;
  }

  std::unique_ptr<mlir::ExecutionEngine>& compileModule() {
      
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    unsigned optLevel = 3;
    unsigned sizeLevel = 0;
    llvm::TargetMachine* targetMachine = nullptr;
    std::function<llvm::Error(llvm::Module *)> optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);

    llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp, llvm::LLVMContext &)> llvmModuleBuilder = nullptr;
    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine = mlir::ExecutionEngine::create(theModule, llvmModuleBuilder, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    std::unique_ptr<mlir::ExecutionEngine> &engine = maybeEngine.get();
  
    return engine;
  }

  void compileAndExecuteModule() {

    std::unique_ptr<mlir::ExecutionEngine> &engine = compileModule();
    
    llvm::Error invocationResult = engine->invoke("tibsMLIRFunc");
    if (invocationResult) {
      std::cout << "Compilation failed" << std::endl;
      return;
    }
  
    return;
  }
  
  void generateModule() { // TODO get rid of this
    
    // Create a location
    std::string fileName = "/tmp/non_existant_file.fake";
    int lineNumber = 12;
    int columnNumber = 34;
    mlir::Location location = builder.getFileLineColLoc(builder.getIdentifier(fileName), lineNumber, columnNumber);
    std::cout << "location.dump(): \n";
    location.dump();
    std::cout << "\n";

    { // Main Function
    
      // Create empty function
      unsigned long int numberOfArgs = 0;
      llvm::ArrayRef<int64_t> inputArrayShape({2,3});
      mlir::RankedTensorType inputTensorType = mlir::RankedTensorType::get(inputArrayShape, builder.getF64Type());
      llvm::SmallVector<mlir::Type, 4> arg_types(numberOfArgs, inputTensorType); // TODO Figure out what the 4 denotes
      mlir::FunctionType func_type = builder.getFunctionType(arg_types, llvm::None);
      llvm::StringRef funcName("tibsMLIRFunc");
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
      mlir::Value constantOperation = builder.create<mlir::tibs::ConstantOp>(location, constantTensorType, constantDataAttribute);

      // Add print to function body
      builder.create<mlir::tibs::PrintOp>(location, constantOperation);

      // Add emtpy return statement to function body
      mlir::tibs::ReturnOp returnOperation;
      builder.create<mlir::tibs::ReturnOp>(location);
  
      // Add function to module
      theModule.push_back(funcOp);
    }

    return;
  }
};

/***********************/
/* Python Entry Points */
/***********************/

extern "C" void runAllPasses() {
  
  std::cout << "===== runAllPasses start =====" << "\n\n" << "\n\n";

  ModuleGenerator moduleGenerator;
  moduleGenerator.generateModule();

  // Original MLIR
  std::cout << "Original MLIR:" << "\n";
  std::cout << "\n";
  moduleGenerator.dumpModule();

  // Final MLIR
  moduleGenerator.runPassManager();
  std::cout << "\n";
  std::cout << "Final MLIR:" << "\n";
  std::cout << "\n";
  moduleGenerator.dumpModule();
  std::cout << "\n";
  
  std::cout << "===== runAllPasses end =====" << "\n\n" << "\n\n";
  
  std::cout << "===== compile LLVM start =====" << "\n\n" << "\n\n";
  
  moduleGenerator.compileAndExecuteModule();
  
  std::cout << "===== compile LLVM end =====" << "\n\n" << "\n\n";
  
  return;
}
