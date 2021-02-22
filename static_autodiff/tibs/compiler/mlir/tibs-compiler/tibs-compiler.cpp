
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

/*******************/
/* ModuleGenerator */
/*******************/

class ModuleGenerator {

private:
  
  void intializePasses() {
    // Canonicalization Passes
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
    // TODO remove these
    // python3 setup.py build ; python3 -c "import tibs ; print(tibs.compiler.ModuleGenerator().dump_module())"
    context.getOrLoadDialect<mlir::tibs::TibsDialect>();
    intializePasses();
    return;
  }
};

extern "C" void* newModuleGenerator() {
  ModuleGenerator* modGen = new ModuleGenerator();
  void* resultPointer = static_cast<void*>(modGen);
  return resultPointer;
}

extern "C" void deleteModuleGenerator(void* modGenVoidPointer) {
  ModuleGenerator* modGen = static_cast<ModuleGenerator*>(modGenVoidPointer);
  delete modGen;
  return;
}

extern "C" void dumpModule(void* modGen) {
  mlir::ModuleOp &theModule = static_cast<ModuleGenerator*>(modGen)->theModule;
  theModule->dump();
  return;
}

extern "C" bool runPassManager(void* modGenVoidPointer) {
  ModuleGenerator* modGen = static_cast<ModuleGenerator*>(modGenVoidPointer);
  mlir::PassManager &pm = modGen->pm;
  mlir::ModuleOp &theModule = modGen->theModule;
  mlir::LogicalResult pmRunStatus = pm.run(theModule);
  bool successStatus = not mlir::failed(pmRunStatus);
  return successStatus;
}

extern "C" bool compileAndExecuteModule(void* modGenVoidPointer) {
  ModuleGenerator* modGen = static_cast<ModuleGenerator*>(modGenVoidPointer);
  mlir::ModuleOp &theModule = modGen->theModule;
  
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  static const unsigned optLevel = 3;
  static const unsigned sizeLevel = 0;
  llvm::TargetMachine* targetMachine = nullptr;
  std::function<llvm::Error(llvm::Module *)> optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);

  llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp, llvm::LLVMContext &)> llvmModuleBuilder = nullptr;
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine = mlir::ExecutionEngine::create(theModule, llvmModuleBuilder, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  std::unique_ptr<mlir::ExecutionEngine> &engine = maybeEngine.get();
    
  llvm::Error invocationResult = engine->invoke("tibsMLIRFunc");
  bool successStatus = not static_cast<bool>(invocationResult);

  return successStatus;
}

extern "C" void* newLocation(void* modGenVoidPointer, const char* fileNameCharaters, int lineNumber, int columnNumber) {
  ModuleGenerator* modGenPointer = static_cast<ModuleGenerator*>(modGenVoidPointer);
  mlir::OpBuilder &builder = modGenPointer->builder;
  std::string fileName(fileNameCharaters);
  mlir::Location location = builder.getFileLineColLoc(builder.getIdentifier(fileName), lineNumber, columnNumber);
  mlir::Location* copiedLocation = new mlir::Location(location);
  void* resultPointer = static_cast<void*>(copiedLocation);
  return resultPointer;
}

extern "C" void deleteLocation(void* unkownTypePointer) {
  mlir::Location* location = static_cast<mlir::Location*>(unkownTypePointer);
  delete location;
  return;
}

extern "C" void generateModule(void* modGenVoidPointer, void* locationVoidPointer) {
  // TODO get rid of this function
  ModuleGenerator* modGen = static_cast<ModuleGenerator*>(modGenVoidPointer);
  mlir::OpBuilder &builder = modGen->builder;
  mlir::ModuleOp &theModule = modGen->theModule;
  
  mlir::Location *location = static_cast<mlir::Location*>(locationVoidPointer);
  
  { // Main Function

    // Create empty function
    unsigned long int numberOfArgs = 0;
    llvm::ArrayRef<int64_t> inputArrayShape({2,3});
    mlir::RankedTensorType inputTensorType = mlir::RankedTensorType::get(inputArrayShape, builder.getF64Type());
    llvm::SmallVector<mlir::Type, 4> arg_types(numberOfArgs, inputTensorType); // TODO Figure out what the 4 denotes
    mlir::FunctionType func_type = builder.getFunctionType(arg_types, llvm::None);
    llvm::StringRef funcName("tibsMLIRFunc");
    mlir::FuncOp funcOp = mlir::FuncOp::create(*location, funcName, func_type);

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
    mlir::Value constantOperation = builder.create<mlir::tibs::ConstantOp>(*location, constantTensorType, constantDataAttribute);
    
    // Add print to function body
    builder.create<mlir::tibs::PrintOp>(*location, constantOperation);
    
    // Add emtpy return statement to function body
    mlir::tibs::ReturnOp returnOperation;
    builder.create<mlir::tibs::ReturnOp>(*location);
    
    // Add function to module
    theModule.push_back(funcOp);
  }
  
  return;
}
