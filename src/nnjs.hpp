#ifndef NNJS_HPP
#define NNJS_HPP

// Simple Neural Network toolkit
// [CORE]
// Open Source Software under MIT License
// Based on nnjs.js JavaScrtipt NN framework

#include <stdint.h>
#include <cmath>
#include <vector>
#include <assert.h>
#include "prng.hpp"

// Exports: NN namespace

namespace NN {

// Public constants / params

// Use "/" instaed of * during train. Used in unit tests only, should be false on production
extern bool DIV_IN_TRAIN;

// Tools

namespace Internal {

//var PRNG = new Random(1363990714); // would not train on 2+1 -> 2+1 -> 1 configuration
extern Random PRNG; // PRNG = new Random(new Date().getTime());

// Return random float in range [min..max] inclusive
inline double getRandom (double min, double max)
{
  return (PRNG.randFloat(min, max));
}

// Return integer in range [from..limit) :: from-inclusive, limit-exclusive
inline int32_t getRandomInt (int32_t from, int32_t limit)
{
  return ((int)std::floor((PRNG.nextFloat() * (limit - from)) + from)) % limit; // TODO: Verify me
}

// Return integer in range [0, limit) :: 0-inclusive, limit-exclusive
inline int32_t getRandomInt (int32_t limit)
{
  return getRandomInt (0, limit);
}

} // Internal

using namespace NN::Internal;

// Activation functions

inline double S(double x)
{
  return(1.0/(1.0+exp(-x)));
}

inline double SD(double x) //S derivative AKA S' AKA dS/dX
{
  // e^x/(1 + e^x)^2
  return(std::exp(x)/std::pow(1.0+std::exp(x),2));
}

// ProcNeuron types

// ProcNeuron have to have following functions:
// // Proc                                                        Input             Normal             Bias  
// .get()                        - to provide its current result  value of .set     result of .proc    1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links  [ignored]
// .set(inp)                     - to assing input value          assing inp value  [N/A]              [N/A]
// .inputs[] (used for train)    - current input Neurons links    [N/A]             [input link count] [] // empty
// .w[]      (used for train)    - current input Neurons weights  [N/A]             [input link count] [] // empty
// // Construction
// .addInput(ProcNeuron,weight)      - add input link to ProcNeuron       [N/A]             add input link     [ignored]
// .addInputAll(Neurons,weights) - add input link to Neurons      [N/A]             add input links    [ignored]
// // Train
// .getSum()                     - raw sum of all inputs before S [N/A]             sum of .proc       1.0
// .nw[]                         - new input Neurons weights      [N/A]             [input link count] [] // empty
// .initNewWeights()             - init new  weights (.nw) array  [N/A]             copy .w to .nw     [ignored]
// .addNewWeightsDelta(DW)       - adds DW to new  weights (.nw)  [N/A]             add dw to .nw      [ignored]
// .applyNewWeights()            - adds DW to new  weights (.nw)  [N/A]             copy .nw to .w     [ignored]

class BaseNeuron
{
  // Returns "state" value that neuron currently "holds" (state updated by proc() function)
  public: virtual double get() = 0; // abstract

  // Proccess inputs and change state
  public: virtual void proc() { }

  public: virtual ~BaseNeuron() { }
};

class InputNeuron : public BaseNeuron
{
  protected: double out = 0.0;

  public: void set(double value)
  {
    this->out = value;
  }

  public: virtual double get()
  override
  {
    return (this->out);
  }
};

inline double getRandomInitWeight()
{
  return(getRandom(-1, 1));
}

class ProcNeuron : public BaseNeuron
{
  public: std::vector<BaseNeuron*> inputs;
  public: std::vector<double> w;

  protected: double out = 0.0;

  protected: double sum = 0.0; // Used for for training, but kept here to simplify training implemenation

  public: ProcNeuron()
  {
  }

  protected: double calcOutputSum(const std::vector<double> &ins)
  {
    double out = 0;

    size_t count = this->w.size();

    assert(ins.size() == count);

    for (size_t i = 0; i < count; i++)
    {
      out += ins[i] * this->w[i];
    }

    return(out);
  }

  public: void addInput(const BaseNeuron *neuron, double w)
  {
    this->inputs.push_back(const_cast<BaseNeuron *>(neuron));
    this->w.push_back(w);
  }

  public: void addInput(const BaseNeuron *neuron)
  {
    addInput(neuron, getRandomInitWeight());
  }

  public: void addInputAll(const BaseNeuron *(neurons[]), int count, const double weights[] = NULL)
  {
    for (int i = 0; i < count; i++)
    {
      if (weights == NULL)
      {
        addInput(neurons[i]);
      }
      else
      {
        addInput(neurons[i], weights[i]);
      }
    }
  }

  public: void addInputAll(const std::vector<BaseNeuron *> &neurons, const std::vector<double> &weights)
  {
    size_t count = neurons.size();
    size_t weightsCount = weights.size();
    for (size_t i = 0; i < count; i++)
    {
      if (i >= weightsCount)
      {
        addInput(neurons[i]);
      }
      else
      {
        addInput(neurons[i], weights[i]);
      }
    }
  }

  public: void addInputAll(const std::vector<BaseNeuron *> &neurons)
  {
     std::vector<double> dummy_empty_weights;
     addInputAll(neurons, dummy_empty_weights);
  }

  public: virtual void proc() override
  {
    assert(this->inputs.size() == this->w.size());

    std::vector<double> ins;

    size_t count = this->inputs.size();
    for (size_t i = 0; i < count; i++)
    {
      ins.push_back(this->inputs[i]->get());
    }

    this->sum = calcOutputSum(ins);
    this->out = S(this->sum);
  }

  public: virtual double get() override
  {
    return(this->out);
  }
};

class ProcNeuronTrainee : public ProcNeuron
{
  // ProcNeurons extension used for training

  public:    std::vector<double> nw;  // for train

  public: double getSum()
  {
    return(this->sum);
  }

  public: void initNewWeights()
  {
    //this->nw.assign(this->w.begin(), this->w.end());
    this->nw = this->w;
  }

  public: void addNewWeightsDelta(const double dw[])
  {
    size_t count = this->nw.size();
    for (size_t i = 0; i < count; i++)
    {
      this->nw[i] += dw[i];
    }
  }

  public: void addNewWeightsDelta(const std::vector<double> &dw)
  {
    size_t count = this->nw.size();

    assert(count == dw.size());

    for (size_t i = 0; i < count; i++)
    {
      double nwv = this->nw[i] + dw[i];
      this->nw[i] = nwv;
    }
  }

  public: void applyNewWeights()
  {
    //this->w.assign(this->nw.begin(), this->nw.end());
    this->w = this->nw;
  }
};

class BiasNeuron : public BaseNeuron
{
  public: double const BIAS = 1.0;

  public: virtual double get() override
  {
    return(BIAS);
  }
};

class NeuronFactory
{
  public: virtual BaseNeuron *makeNeuron() = 0;
};

template<class T> 
class TheNeuronFactory : public NeuronFactory
{
  public: virtual BaseNeuron *makeNeuron() override { return new T(); };
};

/// This class composes neuron network layer and acts as container for neurons
/// Layer Container "owns" ProcNeuron(s)

class Layer
{
  public: std::vector<BaseNeuron*> neurons;

  public: Layer(int N, NeuronFactory *maker)
  {
    if (N < 0) { N = 0; }
    if (N > 0)
    {
      assert(maker != NULL);
      for (int i = 0; i < N; i++)
      {
        this->addNeuron(maker);
      }
    }
  }

  public: Layer(int N, NeuronFactory &maker)
    : Layer(N, &maker)
  {
  }

  public: Layer()
  {
    // Nothing to do
  }

  public: virtual ~Layer()
  {
    size_t count = this->neurons.size();
    for (size_t i = 0; i < count; i++)
    {
      delete this->neurons[i];
    }
    this->neurons.clear();
  }

  public: BaseNeuron *addNeuron(BaseNeuron *neuron)
  {
    assert(neuron != NULL);
    this->neurons.push_back(neuron);
    return(neuron);
  }

  public: BaseNeuron *addNeuron(NeuronFactory *maker)
  {
    assert(maker != NULL);
    auto neuron = maker->makeNeuron();
    addNeuron(neuron);
    return(neuron);
  }

  public: BaseNeuron *addNeuron(NeuronFactory &maker)
  {
    return this->addNeuron(&maker);
  }

  public: void addInputAll(const Layer *inputLayer)
  {
    size_t count = this->neurons.size();
    for (size_t i = 0; i < count; i++)
    {
      auto neuron = dynamic_cast<ProcNeuron *>(this->neurons[i]);
      if (neuron != NULL)
      {
        neuron->addInputAll(inputLayer->neurons);
      }
    }
  }
};

// Processing functions

inline void doProcNet(std::vector<Layer*> &layers)
{
  // potential optimization:
  // we may start from layer 1 (not 0) to skip input layer [on input/bias neurons proc func is empty]
  // but we will start from 0 because we want support "subnet" case here
  int layersCount = layers.size();
  for (int i = 0; i < layersCount; i++)
  {
    int neuronsCount = layers[i]->neurons.size();
    for (int ii = 0; ii < neuronsCount; ii++)
    {
      layers[i]->neurons[ii]->proc();
    }
  }
}

/// This class composes multiple neuron network layers and acts as container for layers
/// Network Container "owns" Layer(s)

class Network
{
  public: std::vector<Layer*> layers;

  public: virtual ~Network()
  {
    size_t count = layers.size();
    for (size_t i = 0; i < count; i++)
    {
      delete this->layers[i];
    }
    this->layers.clear();
  }

  public: Layer *addLayer(Layer *layer)
  {
    this->layers.push_back(layer);
    return(layer);
  }
};

inline void doProcAssignInput(Network &NET, const std::vector<double> &inputs)
{
  if ((NET.layers.size() <= 0) && (inputs.size() <= 0))
  {
    return; // strange, but assume OK
  }

  assert(NET.layers.size() > 0);

  Layer *LIN = NET.layers[0]; // input layer // Layer *LIN = (*NET.layers.begin());

  // assert(LIN->neurons.size() == inputs.size()); // input layer may have bias neuron(s)

  for (size_t i = 0; i < LIN->neurons.size(); i++)
  {
    auto input = dynamic_cast<InputNeuron*>(LIN->neurons[i]);

    if (input == NULL)
    {
      // Bias ProcNeuron (skip) // Not InputNeuron (skip)
    }
    else
    {
      if (i < inputs.size())
      {
        input->set(inputs[i]);
      }
      else
      {
        input->set(0);
      }
    }
  }
}

inline std::vector<double> doProcGetResult(Network &NET)
{
  std::vector<double> result;

  if (NET.layers.size() <= 0)
  {
    return result; // strange, but assume empty response
  }

  assert(NET.layers.size() > 0);

  Layer *LOUT = NET.layers[NET.layers.size()-1];

  for (size_t i = 0; i < LOUT->neurons.size(); i++)
  {
    result.push_back(LOUT->neurons[i]->get());
  }

  return(result);
}

inline std::vector<double> doProc(Network &NET, const std::vector<double> &inputs)
{
  doProcAssignInput(NET, inputs);

  doProcNet(NET.layers);

  return doProcGetResult(NET);
}

// Training functions

inline double getDeltaOutputSum(ProcNeuronTrainee *outNeuron, double OSME) // OSME = output sum margin of error (AKA Expected - Calculated)
{
  if (outNeuron == NULL) { return NAN; }
  double OS = outNeuron->getSum();
  double DOS = SD(OS) * OSME;
  return(DOS);
}

inline std::vector<double> getDeltaWeights(ProcNeuronTrainee *theNeuron, double DOS) // theNeuron in question, DOS = delta output sum
{
  std::vector<double> DWS;

  if (theNeuron == NULL) { return DWS; } // Empty

  double dw;

  size_t count = theNeuron->inputs.size();
  for (size_t i = 0; i < count; i++)
  {
    if (DIV_IN_TRAIN) { dw = DOS / theNeuron->inputs[i]->get(); } else { dw = DOS * theNeuron->inputs[i]->get(); }
    DWS.push_back(dw);
  }

  return(DWS);
}

inline std::vector<double> getDeltaHiddenSums(ProcNeuronTrainee *theNeuron, double DOS) // theNeuron in question, DOS = delta output sum
{
  std::vector<double> DHS;

  if (theNeuron == NULL) { return DHS; } // Empty

  double ds;

  size_t count = theNeuron->inputs.size();
  for (size_t i = 0; i < count; i++)
  {
    auto input = dynamic_cast<ProcNeuronTrainee*>(theNeuron->inputs[i]);
    if (input == NULL)
    {
      ds = NAN; // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
    }
    else // look like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
    {
      if (DIV_IN_TRAIN) { ds = DOS / theNeuron->w[i] * SD(input->getSum()); } else { ds = DOS * theNeuron->w[i] * SD(input->getSum()); }
    }

    DHS.push_back(ds);
  }

  return(DHS);
}

// Train function

inline void doTrainStepProcPrevLayer(std::vector<BaseNeuron*> &LOUT, std::vector<double> &DOS, int layerIndex)
{
  // Addjust previous layer(s)
  // LOUT[neurons count] = current level (its new weights already corrected)
  // DOS[neurons count]  = delta output sum for each neuron on in current level
  // layerIndex = current later index, where 0 = input layer

  assert(LOUT.size() == DOS.size());

  if (layerIndex <= 1)
  {
    return; // previous layer is an input layer, so skip any action
  }

  for (size_t i = 0; i < LOUT.size(); i++)
  {
    auto neuron = dynamic_cast<ProcNeuronTrainee *>(LOUT[i]);

    if (neuron != NULL)
    {
      auto &LP = neuron->inputs;

      auto DOHS = getDeltaHiddenSums(neuron, DOS[i]);

      assert(LP.size() == DOHS.size());

      for (size_t ii = 0; ii < LP.size(); ii++)
      {
        auto input = dynamic_cast<ProcNeuronTrainee *>(LP[ii]);
        if (input != NULL)
        {
          auto DW = getDeltaWeights(input, DOHS[ii]);
          input->addNewWeightsDelta(DW);
        }
      }

      doTrainStepProcPrevLayer(LP, DOHS, layerIndex-1);
    }
  }
};

inline void doTrainStep(Network &NET, const std::vector<double> &DATA, const std::vector<double> &TARG, double SPEED)
{
  // NET=network, DATA=input, TARG=expeted
  // CALC=calculated output (will be calculated)
  // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

  if (std::isnan(SPEED) ||  SPEED <= 0.0) { SPEED = 1.0; } // 0.1???

  auto CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

  for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
  {
    int iicount = NET.layers[i]->neurons.size();
    for (int ii = 0; ii < iicount; ii++)
    {
      auto neuron = dynamic_cast<ProcNeuronTrainee *>(NET.layers[i]->neurons[ii]);
      if (neuron != NULL)
      {
        neuron->initNewWeights(); // prepare
      }
    }
  }

  // Output layer (special handling)

  auto &LOUT = NET.layers[NET.layers.size()-1]->neurons;

  std::vector<double> OSME; // output sum margin of error (AKA Expected - Calculated) for each output
  std::vector<double> DOS ; // delta output sum for each output neuron
  std::vector<std::vector<double>> DOIW; // delta output neuron input weights each output neuron

  for (size_t i = 0; i < LOUT.size(); i++)
  {
    auto neuron = dynamic_cast<ProcNeuronTrainee *>(LOUT[i]);
    OSME.push_back((TARG[i] - CALC[i]) * SPEED);
    DOS.push_back(getDeltaOutputSum(neuron, OSME[i])); // will handle neuron=NULL case
    DOIW.push_back(getDeltaWeights(neuron, DOS[i])); // will handle neuron=NULL case
    if (neuron != NULL)
    {
      neuron->addNewWeightsDelta(DOIW[i]);
    }
  }

  // proc prev layers

  doTrainStepProcPrevLayer(LOUT, DOS, NET.layers.size()-1);

  for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
  {
    int iicount = NET.layers[i]->neurons.size();
    for (int ii = 0; ii < iicount; ii++)
    {
      auto neuron = dynamic_cast<ProcNeuronTrainee *>(NET.layers[i]->neurons[ii]);
      if (neuron != NULL)
      {
        neuron->applyNewWeights(); // adjust
      }
    }
  }
}

class BaseTraningResutChecker
{
  // Function checks if training is done
  // DATAS is a list of source data sets
  // TARGS is a list of target data sets
  // CALCS is a list of result data sets
  public: virtual bool isTraningDone(const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS) = 0;
};

const double DEFAULT_EPS = 0.1;

inline bool isResultMatchSimpleFunc(const std::vector<double> &TARG, const std::vector<double> &CALC, double eps = NAN)
{
  if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5 // just in case

  auto isResultItemMatch = [eps](double t, double c)
  {
    if (std::abs(t-c) < eps) { return(true); }
    return(false);
  };

  assert(TARG.size() == CALC.size());

  for (size_t ii = 0; ii < TARG.size(); ii++)
  {
    if (!isResultItemMatch(TARG[ii], CALC[ii]))
    {
      return(false);
    }
  }

  return(true);
}

class EpsTraningResutChecker : public BaseTraningResutChecker
{
  protected: double eps = DEFAULT_EPS;

  public: EpsTraningResutChecker()
  {
    this->eps = DEFAULT_EPS;
  }

  public: EpsTraningResutChecker(double eps)
  {
    if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5
    this->eps = eps;
  }

  public: bool isTraningDone(const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS)
  {
    assert(TARGS.size() == CALCS.size());

    for (size_t s = 0; s < TARGS.size(); s++)
    {
      if (!isResultMatchSimpleFunc(TARGS[s], CALCS[s], this->eps))
      {
        return(false);
      }
    }

    return(true);
  }

  public: virtual bool isTraningDone(const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS)
  override
  {
    assert(DATAS.size() == TARGS.size());
    assert(TARGS.size() == CALCS.size());
    return(isTraningDone(TARGS, CALCS));
  }
};

class TrainProgress
{
  public: class TrainingArgs
  {
    public:
    Network   &NET; 
    const      std::vector<std::vector<double>> &DATAS;
    const      std::vector<std::vector<double>> &TARGS;
    double     SPEED;
    int        maxStepsCount;

    TrainingArgs(Network &NET, const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, double SPEED, int maxStepsCount)
      : NET(NET), DATAS(DATAS), TARGS(TARGS), SPEED(SPEED), maxStepsCount(maxStepsCount)
    {
    }
  };

  public: class TrainingStep
  {
    public:
    const      std::vector<std::vector<double>> &CALCS;
    int        stepIndex;

    TrainingStep(const std::vector<std::vector<double>> &CALCS, int stepIndex)
      : CALCS(CALCS), stepIndex(stepIndex)
    {
    }
  };

  public: virtual void onTrainingBegin(TrainingArgs *args) { }
  public: virtual bool onTrainingStep (TrainingArgs *args, TrainingStep *step) { return true; } // return false to abort training
  public: virtual void onTrainingEnd  (TrainingArgs *args, bool isOk) { }
};

const int    DEFAULT_TRAIN_COUNT    = 50000;
const double DEFAULT_TRAINING_SPEED = 0.125;

inline bool doTrain(Network &NET, const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, double SPEED = -1, int MAX_N = -1, TrainProgress *progressReporter = NULL, BaseTraningResutChecker *isTrainDoneFunc = NULL)
{
  if (MAX_N < 0)       { MAX_N = DEFAULT_TRAIN_COUNT; }
  if (SPEED < 0)       { SPEED = DEFAULT_TRAINING_SPEED; }

  EpsTraningResutChecker isTrainDoneDefaultFunc;

  if (isTrainDoneFunc == NULL) { isTrainDoneFunc = &isTrainDoneDefaultFunc; }

  TrainProgress::TrainingArgs trainArgs(NET, DATAS, TARGS, SPEED, MAX_N);

  if (progressReporter != NULL) { progressReporter->onTrainingBegin(&trainArgs); }

  bool isDone = false;
  for (int n = 0; (n < MAX_N) && (!isDone); n++)
  {
    std::vector<std::vector<double>> CALCS;
    for (size_t s = 0; s < DATAS.size(); s++)
    {
      CALCS.push_back(doProc(NET, DATAS[s])); // Fill output
    }

    if (progressReporter != NULL)
    {
      TrainProgress::TrainingStep step(CALCS, n);

      if (!progressReporter->onTrainingStep(&trainArgs, &step))
      {
        // Abort training
        progressReporter->onTrainingEnd(&trainArgs, false);
        return(false);
      }
    }

    isDone = isTrainDoneFunc->isTraningDone(DATAS, TARGS, CALCS);

    if (!isDone)
    {
      for (size_t s = 0; s < DATAS.size(); s++)
      {
        doTrainStep(NET, DATAS[s], TARGS[s], SPEED);
      }
    }
  }

  if (progressReporter != NULL)
  { 
    progressReporter->onTrainingEnd(&trainArgs, isDone);
  }

  return(isDone);
}

// Some internals

/*
NCore.Internal = {};
NCore.Internal.PRNG = PRNG;
NCore.Internal.getRandom = getRandom;
NCore.Internal.getRandomInt = getRandomInt;
NCore.Internal.getDeltaOutputSum  = getDeltaOutputSum;
NCore.Internal.getDeltaWeights    = getDeltaWeights;
NCore.Internal.getDeltaHiddenSums = getDeltaHiddenSums;

// Exports

NCore.ProcNeuron = ProcNeuron;
NCore.InputNeuron = InputNeuron;
NCore.BiasNeuron = BiasNeuron;

NCore.Layer   = Layer;

NCore.doProc  = doProc;
NCore.doTrain = doTrain;
*/

// Aux

inline bool isResultBatchMatchSimpleFunc(const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS, double eps)
{
  EpsTraningResutChecker checker(eps);
  return(checker.isTraningDone(TARGS, CALCS));
}

} // NN

#endif
