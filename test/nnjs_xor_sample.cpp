// Simple Neural Network toolkit
// Open Source Software under MIT License
// [Xor demo network]

#include "nnjs_xor_sample.hpp"
#include <nnjs.hpp>

#include "nnjs_console.hpp"
#include "nnjs_console_training_stat.hpp"

#include <ctime>

namespace NN { namespace Demo {

//extern
bool sampleXorNetwork()
{
  if (true)
  {
    auto seed = time(NULL) % 0x7FFF0000 + 1;
    NN::Internal::PRNG.setSeed(seed);
    console::log("sampleOcrNetwork", "seed=", seed);
  }

  auto IN  = new NN::Layer(2, NN::TheNeuronFactory<NN::InputNeuron>{}); IN->addNeuron(new NN::BiasNeuron());
  auto L1  = new NN::Layer(2, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L1->addNeuron(new NN::BiasNeuron());
  L1->addInputAll(IN);
  auto OUT = new NN::Layer(1, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{});
  OUT->addInputAll(L1);

  NN::Network NET; // = [IN, L1, OUT];
  NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);

  std::vector<std::vector<double>> DATAS{ std::vector<double>{ 1, 1 }, std::vector<double>{ 1, 0 }, std::vector<double>{ 0, 1 }, std::vector<double>{ 0, 0 } }; // = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  std::vector<std::vector<double>> TARGS{                       { 0 },                       { 1 },                       { 1 },                       { 0 } }; // = [    [0],    [1],    [1],    [0]];

  auto result = NN::doTrain(NET, DATAS, TARGS, -1, -1, &NN::TrainingProgressReporterConsole(1000));

  return(result);
}

bool sampleXorNetwork2()
{
  if (true)
  {
    auto seed = time(NULL) % 0x7FFF0000 + 1;
    NN::Internal::PRNG.setSeed(seed);
    console::log("sampleOcrNetwork2", "seed=", seed);
  }

  auto IN  = new NN::Layer(2, NN::TheNeuronFactory<NN::InputNeuron>{}); IN->addNeuron(new NN::BiasNeuron());
  auto L1  = new NN::Layer(3, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L1->addNeuron(new NN::BiasNeuron());
  L1->addInputAll(IN);
  auto L2  = new NN::Layer(3, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L2->addNeuron(new NN::BiasNeuron());
  L2->addInputAll(L1);
  auto OUT = new NN::Layer(1, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{});
  OUT->addInputAll(L2);

  NN::Network NET; // = [IN, L1, L2, OUT];
  NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(L2); NET.addLayer(OUT);

  std::vector<std::vector<double>> DATAS{ std::vector<double>{ 1, 1 }, std::vector<double>{ 1, 0 }, std::vector<double>{ 0, 1 }, std::vector<double>{ 0, 0 } }; // = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  std::vector<std::vector<double>> TARGS{                       { 0 },                       { 1 },                       { 1 },                       { 0 } }; // = [    [0],    [1],    [1],    [0]];

  auto result = NN::doTrain(NET, DATAS, TARGS, -1, -1, &NN::TrainingProgressReporterConsole(1000));

  return(result);
}

} } // NN:Demo
