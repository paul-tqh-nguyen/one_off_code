func @hello() {
  %chain = tfrt.new.chain

  // Create a string containing "hello world" and store it in %hello.
  %hello = "tfrt_test.get_string"() { string_attr = "hello world" } : () -> !tfrt.string

  // Print the string in %hello.
  "tfrt_test.print_string"(%hello, %chain) : (!tfrt.string, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
