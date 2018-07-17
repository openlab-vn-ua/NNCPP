// Simple Neural Network toolkit
// Open Source Software under MIT License
// [Unit test]

#include "nnjs_unit_test.hpp"

#include <cmath>
#include <vector>
#include <functional>
#include <nnjs.hpp>

// Simple Neural Network toolkit
// [Unit test]

// Require nnjs.js

// Test case(s)
// ---------------------------------

#define CONSOLE_LOG(...) // Nothing to do yet

namespace NN { namespace Test {

#define TEST_DEFAULT_EPS 0.0001

static bool isFloatAlmostEqual(double a, double b, double eps = NAN)
{
  if (std::isnan(eps)) { eps = TEST_DEFAULT_EPS; }
  double dif = a-b;
  if (dif < 0) { dif = -dif; }
  return (dif <= eps);
}

static bool isFloatListAlmostEqual(const std::vector<double> &a, const std::vector<double> &b, double eps = NAN)
{
  if (a.size() != b.size()) { return false; }
  size_t count = a.size();
  for (size_t i = 0; i < count; i++)
  {
    if (!isFloatAlmostEqual(a[i],b[i],eps)) { return false; }
  }
  return true;
}

static bool doUnitTest1()
{
  // Test case based on
  // http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
  // Note: For some reason original test uses division by S' instead of S' multiplication during train calculation 
  // That is why we use DIV_IN_TRAIN=true here to keep clculation consistent with the original good explaied test (basides the trick with "/")
  // Mode DIV_IN_TRAIN=true intended to use for this test only, during production work we use is as false

  bool isOk = true;

  bool ODT = DIV_IN_TRAIN;

  NN::DIV_IN_TRAIN = true;

  auto addInput = [](BaseNeuron *to, BaseNeuron *input, double weight)
  {
    auto target = dynamic_cast<ProcNeuron*>(to);
    if (target != NULL)
    {
      target->addInput(input, weight);
    }
  };

  auto NTE = [](BaseNeuron *source)
  {
    auto target = dynamic_cast<ProcNeuronTrainee*>(source);
    if (target == NULL)
    {
      throw "Invalid operation: Cannot convert BaseNeuron to ProcNeuronTrainee";
    }

    return(target);
  };

  auto IN  = new NN::Layer(2, NN::TheNeuronFactory<NN::InputNeuron>{});

  auto L1  = new NN::Layer(3, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); 
  //L1->addInputAll(IN);
  addInput(L1->neurons[0], IN->neurons[0], 0.8);
  addInput(L1->neurons[0], IN->neurons[1], 0.2);
  addInput(L1->neurons[1], IN->neurons[0], 0.4);
  addInput(L1->neurons[1], IN->neurons[1], 0.9);
  addInput(L1->neurons[2], IN->neurons[0], 0.3);
  addInput(L1->neurons[2], IN->neurons[1], 0.5);

  auto OUT = new NN::Layer(1, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); 
  //OUT->addInputAll(L1);
  addInput(OUT->neurons[0], L1->neurons[0], 0.3);
  addInput(OUT->neurons[0], L1->neurons[1], 0.5);
  addInput(OUT->neurons[0], L1->neurons[2], 0.9);

  NN::Network NET;
  
  NET.addLayer(IN);
  NET.addLayer(L1);
  NET.addLayer(OUT);

  std::vector<double> DATA {1, 1}; // Input
  std::vector<double> TARG { 0 }; // Expected output

  auto CALC = NN::doProc(NET, DATA)[0]; // Actual output

  if (!isFloatAlmostEqual(CALC,0.7743802720529458))
  {
    isOk = false;
    CONSOLE_LOG("FAIL: Result", CALC);
  }
 
  // Adjust Output layer

  auto OSME = TARG[0] - CALC;

  if (!isFloatAlmostEqual(OSME,-0.7743802720529458))
  {
    isOk = false;
    CONSOLE_LOG("FAIL: output sum margin of error", OSME);
  }

  auto DOS = NN::getDeltaOutputSum(NTE(OUT->neurons[0]), OSME);
  if (!isFloatAlmostEqual(DOS, -0.13529621033156358))
  {
    CONSOLE_LOG("FAIL: delta output sum", DOS); // How much sum have to be adjusted
  }

  auto &pOut = NTE(OUT->neurons[0])->inputs; // Pre-output layer (L1)
  auto DWS = NN::getDeltaWeights(NTE(OUT->neurons[0]), DOS);

  if (!isFloatListAlmostEqual(DWS, std::vector<double> { -0.1850689045809531, -0.1721687291239315, -0.19608871636883077 }))
  {
    CONSOLE_LOG("FAIL: delta weights", DWS); // How much w of prev neurons have to be adjusted
  }

  NTE(OUT->neurons[0])->initNewWeights();
  NTE(OUT->neurons[0])->addNewWeightsDelta(DWS);

  auto NWS = NTE(OUT->neurons[0])->nw;

  if (!isFloatListAlmostEqual(NWS, std::vector<double> { 0.11493109541904689, 0.3278312708760685, 0.7039112836311693 }))
  {
    CONSOLE_LOG("FAIL: new weights", NWS); // New w of output
  }

  // calclulate how to change outputs of prev layer (DOS for each neuton of prev layer)
  // DOS is delta output sum for this neuron

  auto DHS = NN::getDeltaHiddenSums(NTE(OUT->neurons[0]), DOS);

  if (!isFloatListAlmostEqual(DHS, std::vector<double> { -0.08866949824511623, -0.045540261294143396, -0.032156856991522986 }))
  {
    CONSOLE_LOG("FAIL: delta hidden sums", DHS); // array of DOS for prev layer
  }

  // Proc the hidden layer

  std::vector<std::vector<double>> DWSL1; // = [];
  std::vector<std::vector<double>> NWSL1; // = [];

  for (size_t i = 0; i < pOut.size(); i++)
  {
    DWSL1.push_back(NN::getDeltaWeights(NTE(L1->neurons[i]), DHS[i]));
    NTE(L1->neurons[i])->initNewWeights(); // would work this way since only one output neuron (so will be called once for each hidden neuron)
    NTE(L1->neurons[i])->addNewWeightsDelta(DWSL1[i]);
    NWSL1.push_back(NTE(L1->neurons[i])->nw);
  }

  if (!isFloatListAlmostEqual(DHS, std::vector<double> { -0.08866949824511623, -0.045540261294143396, -0.032156856991522986 }))
  {
    CONSOLE_LOG("FAIL: delta hidden sums", DHS); // array of DOS for prev layer
  }

  if (!isFloatListAlmostEqual(DWSL1[0], std::vector<double> {-0.08866949824511623 , -0.08866949824511623 }) ||
      !isFloatListAlmostEqual(DWSL1[1], std::vector<double> {-0.045540261294143396, -0.045540261294143396}) ||
      !isFloatListAlmostEqual(DWSL1[2], std::vector<double> {-0.032156856991522986, -0.032156856991522986}))
  {
    CONSOLE_LOG("FAIL: delta weights L1", [DWSL1]); // array of DOS for prev layer
  }

  if (!isFloatListAlmostEqual(NWSL1[0], std::vector<double> {0.7113305017548838, 0.11133050175488378}) ||
      !isFloatListAlmostEqual(NWSL1[1], std::vector<double> {0.3544597387058566, 0.8544597387058567 }) ||
      !isFloatListAlmostEqual(NWSL1[2], std::vector<double> {0.267843143008477 , 0.467843143008477  }))
  {
    CONSOLE_LOG("FAIL: new weights L1", [NWSL1]); // array of NW for prev layer
  }

  // assign

  NTE(OUT->neurons[0])->applyNewWeights();

  for (size_t i = 0; i < pOut.size(); i++)
  {
    NTE(L1->neurons[i])->applyNewWeights();
  }

  auto CALC2 = NN::doProc(NET, DATA)[0]; // Actual output

  if (!isFloatAlmostEqual(CALC2,0.6917258326007417))
  {
    isOk = false;
    CONSOLE_LOG("FAIL: Result after adjust", CALC2); // should be 0.6917258326007417
  }

  NN::DIV_IN_TRAIN = ODT;

  return isOk;
}

static bool doUnitTestRNG1()
{
  bool isOk = true;
  Random TRNG(42);
  if (isOk) { isOk = (705894     == TRNG.next()); }
  if (isOk) { isOk = (1126542223 == TRNG.next()); }
  if (isOk) { isOk = (1579310009 == TRNG.next()); }
  if (isOk) { isOk = (565444343  == TRNG.next()); }
  if (isOk) { isOk = (807934826  == TRNG.next()); }
  return(isOk);
}

static bool doUnitTestRNG2()
{
  bool isOk = true;
  Random TRNG(42);
  if (isOk) { isOk = isFloatAlmostEqual(0.0003287070433876543 , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.5245871017916008    , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.7354235320681926    , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.26330554044182      , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.3762239710206389    , TRNG.nextFloat()); }
  return(isOk);
}

#define TEST_RNG_MAX_COUNT 1000000

static int32_t getTestRNGCountSeed()
{ 
  return(42); 
}

static bool doUnitTestRNG3()
{
  bool isOk = true;
  Random TRNG(getTestRNGCountSeed());
  double r;
  for (int i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.nextFloat();
    if (r < 0) { isOk = false; break; }
    if (r >= 1.0) { isOk = false; break; }
  }
  if (!isOk) { CONSOLE_LOG("FAIL", r); }
  return(isOk);
}

static bool doUnitTestRNG4()
{
  bool isOk = true;
  Random TRNG(getTestRNGCountSeed());
  double r;
  for (int i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat();
    if (r < 0) { isOk = false; break; }
    if (r > 1.0) { isOk = false; break; }
  }
  if (!isOk) { CONSOLE_LOG("FAIL", r); }
  return(isOk);
}

static bool doUnitTestRNG5()
{
  bool isOk = true;
  Random TRNG(getTestRNGCountSeed());
  double r;
  double TMAX = TRNG.RAND_MAX_VALUE / 64.0;
  int cmax = 0;
  for (int i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat(TMAX);
    if (r < 0) { isOk = false; break; }
    if (r > TMAX) { isOk = false; break; }
    if (r == TMAX) { cmax++; }
  }
  if (!isOk) { CONSOLE_LOG("FAIL", r); }
  //if (isOk) { if (cmax <= 0) { isOk = false; CONSOLE_LOG("FAIL: no max found"); } }
  return(isOk);
}

static bool doUnitTestRNG6()
{
  bool isOk = true;
  Random TRNG(getTestRNGCountSeed());
  double r;
  double TMIN = 3333;
  double TMAX = 5555;
  int cmin = 0;
  int cmax = 0;
  for (int i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat(TMIN, TMAX);
    if (r < TMIN) { isOk = false; break; }
    if (r > TMAX) { isOk = false; break; }
    if (r == TMIN) { cmin++; }
    if (r == TMAX) { cmax++; }
  }
  if (!isOk) { CONSOLE_LOG("FAIL", r); }
  //if (isOk) { if (cmax <= 0) { isOk = false; CONSOLE_LOG("FAIL: no max found"); } }
  //if (isOk) { if (cmin <= 0) { isOk = false; CONSOLE_LOG("FAIL: no min found"); } }
  return(isOk);
}


//extern
bool runUnitTests()
{
  std::vector<std::function<bool()>> TESTS 
  {
    &doUnitTest1, 
    &doUnitTestRNG1, 
    &doUnitTestRNG2, 
    &doUnitTestRNG3, 
    &doUnitTestRNG4, 
    &doUnitTestRNG5, 
    &doUnitTestRNG6, 
  };

  auto count = TESTS.size();
  auto failed = 0;
  for (size_t i = 0; i < count; i++)
  {
    auto test = TESTS[i];
    if (!test())
    {
      failed++;
      CONSOLE_LOG("UNIT "+i+" failed");
    }
  }

  if (failed == 0)
  {
    CONSOLE_LOG("UNIT TESTS OK "+count+"");
  }
  else
  {
    CONSOLE_LOG("UNIT TESTS FAILED "+failed+" of "+count+"");
  }

  return(failed == 0);
}

} } // NN::Test

