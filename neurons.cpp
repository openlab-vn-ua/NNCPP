// Main test app

#include <nnjs.hpp>

#include <nnjs_unit_test.hpp>
#include <nnjs_xor_sample.hpp>

#include <iostream>

int main()
{
  if (!NN::Test::runUnitTests())
  {
    std::cout << "Units tests FAILED" << "\n";
  }
  else
  {
    std::cout << "Units tests passed OK" << "\n";
  }

  NN::Demo::sampleXorNetwork();

  return 0;
}

