func @hello_integers() {
  %chain = tfrt.new.chain

  // Create an integer containing 42.
  %forty_two = tfrt.constant.i32 42

  // Print 42.
  tfrt.print.i32 %forty_two, %chain

  tfrt.return
}