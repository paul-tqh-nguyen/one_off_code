module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 440 : i32}} {
  func @main() {
    %0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %1 = "tf.Const"() {value = dense<-1.41421354> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.Const"() {value = dense<2.82842708> : tensor<f32>} : () -> tensor<f32>
    %3 = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf.resource<tensor<1xf32>>>
    %5 = "tf.VarIsInitializedOp"(%4) {device = ""} : (tensor<!tf.resource<tensor<1xf32>>>) -> tensor<i1>
    "tf.AssignVariableOp"(%4, %0) {device = ""} : (tensor<!tf.resource<tensor<1xf32>>>, tensor<1xf32>) -> ()
    %6 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf.resource<tensor<2x1xf32>>>
    %7 = "tf.VarIsInitializedOp"(%6) {device = ""} : (tensor<!tf.resource<tensor<2x1xf32>>>) -> tensor<i1>
    %8 = "tf.RandomUniform"(%3) {_class = ["loc:@dense/kernel"], device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x1xf32>
    %9 = "tf.Mul"(%8, %2) {_class = ["loc:@dense/kernel"], device = ""} : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %10 = "tf.AddV2"(%9, %1) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    "tf.AssignVariableOp"(%6, %10) {device = ""} : (tensor<!tf.resource<tensor<2x1xf32>>>, tensor<2x1xf32>) -> ()
    return
  }
}