
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "Tibs/TibsDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register tibs translations here.

  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
