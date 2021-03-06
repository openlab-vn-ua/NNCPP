// Simple Neural Network toolkit
// Open Source Software under MIT License
// [Main test app]

#include <nnjs.hpp>

#include <nnjs_unit_test.hpp>
#include <nnjs_xor_sample.hpp>
#include <nnjs_ocr_sample.hpp>

#include <iostream>

int main()
{
  if (!NN::Test::runUnitTests())
  {
    std::cout << "Units tests: FAILED" << "\n";
  }
  else
  {
    std::cout << "Units tests: passed OK" << "\n";
  }

  bool isOk;

  isOk = NN::Demo::sampleXorNetwork();
  std::cout << "XOR Training result:" << (isOk ? "OK" : "FAIL") << "\n";

  isOk = NN::Demo::sampleXorNetwork2();
  std::cout << "XOR2 Training result:" << (isOk ? "OK" : "FAIL") << "\n";

  isOk = NN::Demo::sampleOcrNetwork();
  std::cout << "OCR Training result:" << (isOk ? "OK" : "FAIL") << "\n";

  return 0;
}

