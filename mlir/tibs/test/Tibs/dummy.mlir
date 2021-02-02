// RUN: tibs-opt %s | tibs-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = tibs.foo %{{.*}} : i32
        %res = tibs.foo %0 : i32
        return
    }
}
