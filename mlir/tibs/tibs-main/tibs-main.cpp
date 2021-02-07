
#include <iostream>
#include <boost/type_index.hpp>
#include "tibs-shared.hpp"

int main(int argc, char **argv) {
  
  runAllPasses();
  
  // using namespace boost::typeindex;
  // std::cout << "type_id_with_cvr<decltype(theModule)>().pretty_name(): " << type_id_with_cvr<decltype(theModule)>().pretty_name() << "\n";

  return 0;
}
