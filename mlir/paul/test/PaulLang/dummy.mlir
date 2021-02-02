// RUN: paullang-opt %s | paullang-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = paullang.foo %{{.*}} : i32
        %res = paullang.foo %0 : i32
        return
    }
}
