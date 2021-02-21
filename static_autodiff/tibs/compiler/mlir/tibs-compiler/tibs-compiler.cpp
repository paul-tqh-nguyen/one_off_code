
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
/*********************/

// TODO rename this section ^^

class ModuleGenerator {

private:
  
  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  mlir::PassManager pm;
  
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

  void dumpModule() const {
    theModule->dump();
    return;
  }

  bool runPassManager () {
    mlir::LogicalResult pmRunStatus = pm.run(theModule);
    bool successStatus = not mlir::failed(pmRunStatus);
    return successStatus;
  }

  bool compileAndExecuteModule() const {
      
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
  
  void generateModule() { // TODO get rid of this
        
    // Create a location
    std::string fileName = "/tmp/non_existant_file.fake";
    int lineNumber = 12;
    int columnNumber = 34;
    mlir::Location location = builder.getFileLineColLoc(builder.getIdentifier(fileName), lineNumber, columnNumber);
    
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

extern "C" void runAllPasses() { // TODO remove this
  
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

extern "C" void generateModule(void* unkown_type_pointer) { // TODO get rid of this
  static_cast<ModuleGenerator*>(unkown_type_pointer)->generateModule();
  return;
}

///////////////////////////////////////////////////////////////////////////// TODO get rid of the above

extern "C" void* newModuleGenerator() {
  ModuleGenerator* modGen = new ModuleGenerator();
  void* result_pointer = static_cast<void*>(modGen);
  return result_pointer;
}

extern "C" void deleteModuleGenerator(void* unkown_type_pointer) {
  ModuleGenerator* modGen = static_cast<ModuleGenerator*>(unkown_type_pointer);
  delete modGen;
  return;
}

extern "C" void dumpModule(void* modGen) {
  static_cast<ModuleGenerator*>(modGen)->dumpModule();
  return;
}

extern "C" bool runPassManager(void* modGen) {
  return static_cast<ModuleGenerator*>(modGen)->runPassManager();
}

extern "C" bool compileAndExecuteModule(void* modGen) {
  return static_cast<ModuleGenerator*>(modGen)->compileAndExecuteModule();
}
