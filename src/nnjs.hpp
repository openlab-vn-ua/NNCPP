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

// Tools

namespace Internal {

inline Random* getPRNG()
{
  //var PRNG = new Random(1363990714); // would not train on 2+1 -> 2+1 -> 1 configuration
  static Random PRNG; // PRNG = new Random(new Date().getTime());
  return(&PRNG);
}

// Return random float in range [min..max] inclusive
inline double getRandom (double min, double max)
{
  return (getPRNG()->randFloat(min, max));
}

// Return integer in range [from..limit) :: from-inclusive, limit-exclusive
inline int32_t getRandomInt (int32_t from, int32_t limit)
{
  int32_t valcount = limit - from;
  return ((int)std::floor((getPRNG()->nextFloat() * valcount) + from)) % limit; // TODO: Verify me
}

// Return integer in range [0, limit) :: 0-inclusive, limit-exclusive
inline int32_t getRandomInt (int32_t limit)
{
  return getRandomInt (0, limit);
}

class NonAssignable // derive from this to prevent copy of move
{
  private: NonAssignable(NonAssignable const&) { }
  private: NonAssignable& operator=(NonAssignable const&) { }
  public:  NonAssignable() {}
};

const double NaN = std::numeric_limits<double>::quiet_NaN(); // NAN

} // Internal

using namespace NN::Internal;

// Activation : Base

/// Activation function provider (Should be stateless)
class ActFunc
{
  /// Activation function
  public: virtual double S(double x) const = 0; 
};

/// Activation function provider for training  (Should be stateless)
class ActFuncTrainee : public ActFunc
{
  /// Activation function derivative
  public: virtual double SD(double x) const = 0; 
};

// Activation : Sigmoid

class CalcMathSigmoid // static provider
{
  public: static double S(double x)  { return(1.0 / (1.0 + exp(-x))); } // S is sigmoid function // DEF: y=1/(1+exp(-x))
  public: static double SD(double x) { return(std::exp(x) / std::pow(1.0 + std::exp(x), 2)); } // S derivative AKA S' AKA dS/dX // DEF: y=e^x/(1 + e^x)^2
};

class ActFuncSigmoid : public ActFunc
{
  public: virtual double S(double x) const override  { return CalcMathSigmoid::S(x); }
  public: static ActFuncSigmoid* getInstance() { static ActFuncSigmoid s; return &s; }
};

class ActFuncSigmoidTrainee : public ActFuncTrainee
{
  public: virtual double S(double x)  const override { return CalcMathSigmoid::S(x);  }
  public: virtual double SD(double x) const override { return CalcMathSigmoid::SD(x); }
  public: static ActFuncSigmoidTrainee* getInstance() { static ActFuncSigmoidTrainee s; return &s; }
};

// Activation : RELU

class CalcMathRELU // static provider
{
  public: static double S(double x)  { return (x < 0) ? 0.0 : x; }
  public: static double SD(double x) { return (x < 0) ? 0.0 : 1.0; }
};

class ActFuncRELU : public ActFunc
{
  public: virtual double S(double x) const override { return CalcMathRELU::S(x); }
  public: static ActFuncRELU* getInstance() { static ActFuncRELU s; return &s; }
};

class ActFuncRELUTrainee : public ActFuncTrainee
{
  public: virtual double S(double x)  const override { return CalcMathRELU::S(x); }
  public: virtual double SD(double x) const override { return CalcMathRELU::SD(x); }
  public: static ActFuncRELUTrainee* getInstance() { static ActFuncRELUTrainee s; return &s; }
};

// Activation : LRELU

const double CoreMathLRELUDefLeak = 0.001; // [0.0..1.0)

class CoreMathLRELU // core provider
{
  public: double leak = CoreMathLRELUDefLeak;
  public: CoreMathLRELU(double leak = CoreMathLRELUDefLeak) : leak(leak) { }
  public: double S(double x)  const { return (x < 0) ? x * leak : x; }
  public: double SD(double x) const { return (x < 0) ? leak : 1.0; }
};

class ActFuncLRELU : public ActFunc
{
  protected: CoreMathLRELU core;
  public: ActFuncLRELU(double leak = CoreMathLRELUDefLeak) : core(leak) { }
  public: virtual double S(double x) const override { return core.S(x); }
  public: static ActFuncLRELU* getInstance() { static ActFuncLRELU s; return &s; }
  public: static ActFuncLRELU* newInstance(double leak = CoreMathLRELUDefLeak) { return new ActFuncLRELU(leak); }
};

class ActFuncLRELUTrainee : public ActFuncTrainee
{
  protected: CoreMathLRELU core;
  public: ActFuncLRELUTrainee(double leak = CoreMathLRELUDefLeak) : core(leak) { }
  public: virtual double S(double x)  const override { return core.S(x); }
  public: virtual double SD(double x) const override { return core.SD(x); }
  public: static ActFuncLRELUTrainee* getInstance() { static ActFuncLRELUTrainee s; return &s; }
  public: static ActFuncLRELUTrainee* newInstance(double leak = CoreMathLRELUDefLeak) { return new ActFuncLRELUTrainee(leak); }
};

// Activation : LLRELU

const double CoreMathLLRELUDefLeak = 0.001; // [0.0..1.0)

class CoreMathLLRELU // core provider
{
  public: double nleak = CoreMathLLRELUDefLeak;
  public: double pleak = CoreMathLLRELUDefLeak;
  public: CoreMathLLRELU(double leak = CoreMathLLRELUDefLeak) : nleak(leak), pleak(leak) { }
  public: CoreMathLLRELU(double nleak, double pleak) : nleak(nleak), pleak(pleak) { }
  public: double S(double x)  const { return (x < 0) ? x * nleak : (x <= 1.0) ? x : 1.0 + (x-1.0) * pleak; }
  public: double SD(double x) const { return (x < 0) ? nleak : (x <= 1.0) ? 1.0 : pleak; }
};

class ActFuncLLRELU : public ActFunc
{
  protected: CoreMathLLRELU core;
  public: ActFuncLLRELU(double leak = CoreMathLLRELUDefLeak) : core(leak) { }
  public: ActFuncLLRELU(double nleak, double pleak) : core(nleak, pleak) { }
  public: virtual double S(double x) const override { return core.S(x); }
  public: static ActFuncLLRELU* getInstance() { static ActFuncLLRELU s; return &s; }
  public: static ActFuncLLRELU* newInstance(double leak = CoreMathLLRELUDefLeak) { return new ActFuncLLRELU(leak); }
};

class ActFuncLLRELUTrainee : public ActFuncTrainee
{
  protected: CoreMathLLRELU core;
  public: ActFuncLLRELUTrainee(double leak = CoreMathLLRELUDefLeak) : core(leak) { }
  public: ActFuncLLRELUTrainee(double nleak, double pleak) : core(nleak, pleak) { }
  public: virtual double S(double x)  const override { return core.S(x); }
  public: virtual double SD(double x) const override { return core.SD(x); }
  public: static ActFuncLLRELUTrainee* getInstance() { static ActFuncLLRELUTrainee s; return &s; }
  public: static ActFuncLLRELUTrainee* newInstance(double leak = CoreMathLLRELUDefLeak) { return new ActFuncLLRELUTrainee(leak); }
  public: static ActFuncLLRELUTrainee* newInstance(double nleak, double pleak) { return new ActFuncLLRELUTrainee(nleak, pleak); }
};

// Activation : Tanh

class CalcMathTanh // static provider
{
  public: static double S(double x)  { return std::tanh(x); }
  public: static double SD(double x) { return 1.0-std::pow(std::tanh(x),2.0); }
};

class ActFuncTanh : public ActFunc
{
  public: virtual double S(double x) const override { return CalcMathTanh::S(x); }
  public: static ActFuncTanh* getInstance() { static ActFuncTanh s; return &s; }
};

class ActFuncTanhTrainee : public ActFuncTrainee
{
  public: virtual double S(double x)  const override { return CalcMathTanh::S(x); }
  public: virtual double SD(double x) const override { return CalcMathTanh::SD(x); }
  public: static ActFuncTanhTrainee* getInstance() { static ActFuncTanhTrainee s; return &s; }
};

// Default Activation Functions

inline ActFunc*        getDefActFunc()        { return ActFuncSigmoid::getInstance(); }
inline ActFuncTrainee* getDefActFuncTrainee() { return ActFuncSigmoidTrainee::getInstance(); }

// Neuron types

// Neuron have to have following functions:
// // Base                                                        Input             Normal               Bias  
// .get()                        - to provide its current result  value of .set     result of .proc      1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links    [ignored]
// // Input
// .set(inp)                     - to assing input value          assing inp value  [N/A]                [N/A]
// // Proc
// .inputs[] (used for train)    - current input Neurons links    [N/A]             [input link count]   [N/A]
// .w[]      (used for train)    - current input Neurons weights  [N/A]             [input link count]   [N/A]
// // Construction
// .addInput(Neuron,weight)      - add input link to Neuron       [N/A]             add input link       [N/A]
// .addInputAll(Neurons,weights) - add input link to Neurons      [N/A]             add input links      [N/A]
// // Trainee
// .getSum()                     - raw sum of all inputs before S [N/A]             sum of .proc         [N/A]
// .nw[]                         - new input Neurons weights      [N/A]             [input link count]   [N/A]
// .initTrainStep()              - init new  weights (.nw) array  [N/A]             .nw[]=.w[], .dos=0   [N/A]
// .addNewWeightsDelta(DW)       - adds DW to new  weights (.nw)  [N/A]             .nw[] += DW[]        [N/A]
// .applyNewWeights()            - adds DW to new  weights (.nw)  [N/A]             .w[]=.nw[]           [N/A]
// // Trainee // FAST
// .dos                          - accumulated delta out sum      [N/A]             accumulated dos      [N/A]
// .addDeltaOutputSum(ddos)      - increments dos for neuron      [N/A]             .dos+ddos            [N/A]
// .getDeltaOutputSum()          - return dos for neuron          [N/A]             return .dos          [N/A]

// Base neuron
// Base class for all neurons
class BaseNeuron : protected NonAssignable
{
  // Returns "state" value that neuron currently "holds" (state updated by proc() function)
  public: virtual double get() const = 0; // abstract

  // Proccess inputs and change state
  public: virtual void proc() { }

  public: virtual ~BaseNeuron() { }
};

// InputNeuron
// Always return set value as its output
class InputNeuron : public BaseNeuron
{
  protected: double out = 0.0;

  public: void set(double value)
  {
    this->out = value;
  }

  public: virtual double get() const
  override
  {
    return (this->out);
  }
};

// Proc Neuron
// Neuron that proccess its input inside proc method
class ProcNeuron : public BaseNeuron
{
  public: std::vector<BaseNeuron*> inputs;
  public: std::vector<double> w;

  protected: ActFunc* func = NULL; // just reference, does not "own" the provider

  protected: double out = 0.0;

  protected: double sum = 0.0; // Used for for training, but kept here to simplify training implementation

  public: ProcNeuron(ActFunc* func = NULL) : func(func == NULL ? getDefActFunc() : func) { }

  protected: double getRandomInitWeight()
  {
    return(getRandom(-1, 1));
  }

  protected: double getRandomInitWeightForNNeurons(size_t n)
  {
    // Select input range of w. for n neurons
    // Simple range is -1..1,
    // But we may try - 1/sqrt(n)..1/sqrt(n) to reduce range of w in case of many inputs (so act func will not saturate)
    double wrange = 1;

    if (n > 0)
    {
      wrange = 1.0/std::sqrt(n);
    }

    return (getRandom(-wrange, wrange));
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
        addInput(neurons[i], getRandomInitWeightForNNeurons(count));
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
        addInput(neurons[i], getRandomInitWeightForNNeurons(count));
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

  // Core proccesing
  // Computes output based on input

  protected: double S(double x) { return func->S(x); }

  public: virtual void proc() override
  {
    assert(this->inputs.size() == this->w.size());

    double sum = 0;
    size_t count = this->inputs.size();
    for (size_t i = 0; i < count; i++)
    {
      sum += this->inputs[i]->get() * this->w[i];
    }

    this->sum = sum;
    this->out = S(sum);
  }

  public: virtual double get() const override
  {
    return(this->out);
  }
};

// Proc neuron "trainee"
// Regular proc neuron that extended with training data and functions
class ProcNeuronTrainee : public ProcNeuron
{
  // ProcNeurons extension used for training

  protected: ActFuncTrainee* train;

  public: ProcNeuronTrainee(ActFuncTrainee* func = NULL) : ProcNeuron(func == NULL ? getDefActFuncTrainee() : func)
  {
    train = dynamic_cast<ActFuncTrainee*>(this->func); 
    assert(train != NULL);
  }

  // #region inputGetTrainee cache implemenation

  protected: std::vector<BaseNeuron*>        inputTraineesSrc;
  protected: std::vector<ProcNeuronTrainee*> inputTraineesOut;

  protected: void  inputTraineesInit()
  {
    inputTraineesOut.clear();
    inputTraineesSrc.clear();
    size_t count = this->inputs.size();
    for (size_t i = 0; i < count; i++)
    {
      inputTraineesSrc.push_back(this->inputs[i]);
      inputTraineesOut.push_back(dynamic_cast<ProcNeuronTrainee *>(this->inputs[i]));
    }
  }

  // #endregion

  // Returns input trainee neuron with index i
  public: ProcNeuronTrainee* inputGetTrainee(size_t i)
  {
    assert(i >= 0);
    assert(i < this->inputs.size());

    //return dynamic_cast<ProcNeuronTrainee*>(this->inputs[i]); // slow

    // Cache driven implementation

    if (this->inputs.size() != inputTraineesOut.size())
    {
      inputTraineesInit(); // size changed
    }
    else if (inputTraineesSrc[i] != this->inputs[i])
    {
      inputTraineesInit(); // source value changed
    }

    return inputTraineesOut[i];
  }

  // Main training

  public: double SD(double x) { return train->SD(x); }

  public: std::vector<double> nw; // new weights for train

  public: double getSum()
  {
    return(this->sum);
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

  public: void applyNewWeights() // Replace current weights with new weights
  {
    //this->w.assign(this->nw.begin(), this->nw.end());
    this->w = this->nw;
  }

  protected: void initTrainStepMain()
  {
    //this->nw.assign(this->w.begin(), this->w.end());
    this->nw = this->w;
  }

  // Fast training

  public: double dos = 0.0; // for train (delta output sum)

  public: void addDeltaOutputSum(double ddos)
  {
    this->dos += ddos;
  }

  public: double getDeltaOutputSum()
  {
    return this->dos;
  }

  protected: void initTrainStepFast()
  {
    this->dos = 0.0;
  }

  public: void initTrainStep()
  {
    initTrainStepMain();
    initTrainStepFast();
  }
};

// BiasNeuron
// Always return 1.0 as its output

class BiasNeuron : public BaseNeuron
{
  public: double const BIAS = 1.0;

  public: virtual double get() const override
  {
    return(BIAS);
  }
};

/// Neuron factory
/// Creates neurons when batch creation is used

class NeuronFactory
{
  public: virtual BaseNeuron *makeNeuron() = 0;
};

template<class T> 
class TheNeuronFactory : public NeuronFactory
{
  public: virtual BaseNeuron *makeNeuron() override { return new T(); };
};

template<class T, class A>
class ExtNeuronFactory : public NeuronFactory
{
  protected: A arg;
  public: ExtNeuronFactory(A arg) : arg(arg) { }
  public: virtual BaseNeuron* makeNeuron() override { return new T(arg); };
};

/// Layer
/// Represent a layer of network
/// This class composes neuron network layer and acts as container for neurons
/// Layer Container "owns" Neuron(s)

class Layer : protected NonAssignable
{
  public: std::vector<BaseNeuron*> neurons;

  public: Layer(int N, NeuronFactory *maker)
  {
    if (N > 0)
    {
      assert(maker != NULL);
      this->addNeurons(N, maker);
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

  public: void addNeurons(int N, NeuronFactory *maker)
  {
    if (N <= 0) { return; }
    assert(maker != NULL);
    for (int i = 0; i < N; i++)
    {
      auto neuron = maker->makeNeuron();
      this->addNeuron(neuron);
    }
  }

  public: void addNeurons(int N, NeuronFactory &maker)
  {
    addNeurons(N, &maker);
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
  // but we will start from 0 because we want support "subnet" case here in future
  // start from 1 to skip input layer
  auto layersCount = layers.size();
  for (size_t i = 1; i < layersCount; i++)
  {
    auto neuronsCount = layers[i]->neurons.size();
    for (size_t ii = 0; ii < neuronsCount; ii++)
    {
      layers[i]->neurons[ii]->proc();
    }
  }
}

/// Network
/// This class composes multiple neuron network layers and acts as container for layers
/// Network Container "owns" Layer(s)

class Network : protected NonAssignable
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

// Assign inputs
// The inputs assigned to first layer of the network
// inputs should be same saze as first (input) layer

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

// Get network output result
// resturns array of network's output

inline std::vector<double> doProcGetResult(const Network &NET)
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

// doProc with alternative params format

inline std::vector<double> doProc(Network* NET, const std::vector<double>& inputs)
{
  return doProc(*NET, inputs);
}

inline std::vector<double> doProc(Network* NET, const std::vector<double>* inputs)
{
  return doProc(*NET, *inputs);
}

// Network Stat and Math functionality
// -----------------------------------------------
// Mostly used for train

namespace NetworkStat // static class
{
  // MatchEps

  const double DEFAULT_EPS = 0.1;

  inline double isResultItemMatchEps(double t, double c, double eps) // private
  {
    //if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5
    if (std::abs(t - c) < eps) { return(true); }
    return (false);
  }

  inline bool isResultSampleMatchEps(const std::vector<double> &TARG, const std::vector<double> &CALC, double eps = NAN)
  {
    if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    auto count = TARG.size();

    assert(count == CALC.size());

    for (size_t ii = 0; ii < count; ii++)
    {
      if (!isResultItemMatchEps(TARG[ii], CALC[ii], eps))
      {
        return(false);
      }
    }

    return(true);
  }

  // MatchArgMax

  inline int getMaximumIndexEps(const std::vector<double> &R, double eps)
  {
    // input:  R as vector of floats (usualy 0.0 .. 1.0), eps min comparison difference
    // result: index of maximum value, checking that next maximum is at least eps lower.
    // returns -1 if no such value found (maximums too close)

    if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    int FAIL = -1;

    auto RLen = R.size();

    if (RLen <= 0) { return(FAIL); }
    if (RLen <= 1) { return(0);    }

    // RLen >= 2

    int currMaxIndex;
    int prevMaxIndex;
    if (R[0] > R[1])
    {
      currMaxIndex = 0;
      prevMaxIndex = 1;
    }
    else
    {
      currMaxIndex = 1;
      prevMaxIndex = 0;
    }

    for (size_t i = 2; i < RLen; i++)
    {
      if (R[i] > R[currMaxIndex]) { prevMaxIndex = currMaxIndex; currMaxIndex = static_cast<int>(i); }
    }

    if (std::isnan(eps))
    {
      // reserved for NAN = do not check (not used as for now)
    }
    else
    {
      if (R[currMaxIndex] < eps)
      {
        return(FAIL); // not ever greater than 0, no reason so check another max
      }

      if (std::abs(R[currMaxIndex] - R[prevMaxIndex]) < eps)
      {
        return(FAIL); // maximums too close
      }
    }

    return (currMaxIndex);
  }

  inline int getMaximumIndex(const std::vector<double> &R)
  {
    // input:  R as vector of floats (usualy 0.0 .. 1.0)
    // result: index of maximum value
    // returns -1 if no such value found (vector is empty)

    int FAIL = -1;

    auto RLen = R.size();

    if (RLen <= 0) { return(FAIL); }
    if (RLen <= 1) { return(0);    }

    // Rlen >= 2

    int currMaxIndex = 0;

    for (size_t i = 1; i < RLen; i++)
    {
      if (R[i] > R[currMaxIndex]) { currMaxIndex = static_cast<int>(i); }
    }

    return (currMaxIndex);
  }

  inline bool isResultSampleMatchArgmaxEps(const std::vector<double> &TARG, const std::vector<double> &CALC, double eps)
  {
    if (std::isnan(eps) || eps <= 0.0) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    auto maxIndex = getMaximumIndexEps(CALC, eps);

    if (maxIndex < 0) { return(false); }

    auto count = TARG.size();

    assert(count == CALC.size());

    for (size_t ii = 0; ii < count; ii++)
    {
      if ((TARG[ii] > 0.0) && (ii != maxIndex))
      {
        return(false);
      }
      else if ((TARG[ii] <= 0.0) && (ii == maxIndex))
      {
        return(false);
      }
    }

    return(true);
  }

  inline bool isResultSampleMatchArgmax(const std::vector<double> &TARG, const std::vector<double> &CALC)
  {
    auto maxIndex = getMaximumIndex(CALC);

    if (maxIndex < 0) { return(false); }

    auto count = TARG.size();

    assert(count == CALC.size());

    for (size_t ii = 0; ii < count; ii++)
    {
      if ((TARG[ii] > 0.0) && (ii != maxIndex))
      {
        return(false);
      }
      else if ((TARG[ii] <= 0.0) && (ii == maxIndex))
      {
        return(false);
      }
    }

    return(true);
  }

  // Aggregated error sum (AKA source for simple loss function)

  // Actually, simple loss function on sample is aggregated error sum divided by 2.0.
  // The divisor is need to have a "clean" partial derivative as simple difference
  // in many cases divisor ommited, as different anyway is mutiplied to small number (learning rate), but we define it here just in case

  inline double getResultSampleAggErrorSum(const std::vector<double> &TARG, const std::vector<double> &CALC)
  {
    auto count = TARG.size();

    assert(count == CALC.size());

    double result = 0;

    for (size_t ii = 0; ii < count; ii++)
    {
      double diff = TARG[ii] - CALC[ii];

      result += diff * diff;
    }

    return(result);
  }

  // AggSum to SimpleLoss mutiplier: to be used as loss function (error function), should be mutiplied by 1/2 so derivative will not have 2x in front

  const double AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY = 0.5;

  inline double getResultSampleSimpleLoss(const std::vector<double> &TARG, const std::vector<double> &CALC)
  {
    return getResultSampleAggErrorSum(TARG, CALC) * AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY;
  }

  // Mean Squared error

  inline double getResultSampleMSE(const std::vector<double> &TARG, const std::vector<double> &CALC)
  {
    auto result = getResultSampleAggErrorSum(TARG, CALC);
    auto count = TARG.size();
    assert(count == CALC.size());
    if (count <= 0) { return NaN; }
    return(result / count);
  }

  inline double getResultMSEByAggErrorSum(double sum, size_t sampleSize, size_t samplesCount = 1)
  {
    auto count = samplesCount;
    if (count > 0) { count *= sampleSize; }
    if (count <= 0) { return NaN; }
    return(sum / count);
  }

  // Aggregated error (AKA MSE rooted)

  inline double getResultAggErrorByAggErrorSum(double sum, size_t sampleSize, size_t samplesCount = 1)
  {
    return std::sqrt(getResultMSEByAggErrorSum(sum, sampleSize, samplesCount));
  }

  // Misc

  /// Retuns array with only one index of total item set to SET(=1) and all other as NOTSET(=0):
  /// Example: 0=[1, 0, 0 ...], 1=[0, 1, 0, ...], 2=[0, 0, 1, ...]
  inline std::vector<double> getR1Array(int index, int total, double SET = 1, double NOTSET = 0)
  {

    // if (SET    == null) { SET    = 1; }
    // if (NOTSET == null) { NOTSET = 0; }

    std::vector<double> R; // = [];

    for (auto i = 0; i < total; i++)
    {
      R.push_back(i == index ? SET : NOTSET);
    }

    return(R);
  }

  inline size_t getNetWeightsCount(const Network &NET)
  {
    size_t result = 0;
    size_t layersCount = NET.layers.size();
    for (size_t i = 0; i < layersCount; i++) // TODO: may skip input layer later
    {
      size_t neuronsCount = NET.layers[i]->neurons.size();
      for (size_t ii = 0; ii < neuronsCount; ii++)
      {
        auto neuron = dynamic_cast<ProcNeuron*>(NET.layers[i]->neurons[ii]);
        if (neuron != NULL) // proc neuron
        {
          result += neuron->w.size();
        }
      }
    }
    return result;
  }

  inline size_t getNetNeuronsCount(const Network& NET)
  {
    size_t result = 0;
    size_t layersCount = NET.layers.size();
    for (size_t i = 0; i < layersCount; i++)
    {
      result += NET.layers[i]->neurons.size();
    }
    return result;
  }
}

// Train functions
// -----------------------------------------------
// Do network train

const int    DEFAULT_MAX_EPOCH_COUNT = 50000;
const double DEFAULT_TRAINING_SPEED  = 0.125;

/// Training parameters
class TrainingParams
{
  public: double speed = DEFAULT_TRAINING_SPEED;
  public: int    maxEpochCount = DEFAULT_MAX_EPOCH_COUNT;
  public: bool   fastVerify = false;

  public: TrainingParams() { }
  public: TrainingParams(double speed, int maxEpochCount = -1, bool fastVerify = false)
  {
    if (speed <= 0) { speed = DEFAULT_TRAINING_SPEED; }
    if (maxEpochCount <= 0) { maxEpochCount = DEFAULT_MAX_EPOCH_COUNT; }
    this->speed = speed;
    this->maxEpochCount = maxEpochCount;
    this->fastVerify = fastVerify;
  }
};

/// Data information
class TrainingDatasetInfo
{
  public: size_t trainDatasetSize;
};

/// Generic class for training lifecycle
class TrainingProccessor
{
  /// <summary> Start the whole training session </summary>
  public: virtual void trainStart(Network& NET, TrainingParams &trainingParams, TrainingDatasetInfo &datasetInfo) { }

  /// <summary> Start training eposh (each pass over the whole dataset) </summary>
  public: virtual void trainEposhStart(Network& NET, int eposhIndex) { }

  /// <summary> End training eposh (each pass over the whole dataset) </summary>
  public: virtual void trainEposhEnd(Network& NET, int eposhIndex) { }

  /// <summary> End the whole training session </summary>
  public: virtual void trainEnd(Network& NET, bool isDone) { }
};

/// Base class for training alogorith implementation
class NetworkTrainer : public TrainingProccessor
{
  /// <summary> Do single training step. Returs result of pre-training run with DATA </summary>
  public: virtual std::vector<double> trainBySample(Network& NET, const std::vector<double>& DATA, const std::vector<double>& TARG, double speed) = 0;
};

/// Class checker for training is done
class TrainingDoneChecker : public TrainingProccessor
{
  /// <summary> Check is sample is valid. If all samples are valid, assumed that training is complete </summary>
  public: virtual bool trainSampleIsValid(Network& NET, const std::vector<double>& TARG, const std::vector<double>& CALC) { return false; }

  /// <summary> Check that eposh result is valid. If this function returns true, training assumed complete </summary>
  public: virtual bool trainEpochIsValid(Network& NET, int epochIndex) { return false; }
};

/// Training progress reporter
/// Will be called during traing to report progress
/// May rise traing abort event

class TrainingProgressReporter : public TrainingProccessor
{
  /// Reports or updates stats by sample. if returns false, training will be aborted
  public: virtual bool trainSampleReportAndCheckContinue(Network& NET, const std::vector<double>& DATA, const std::vector<double>& TARG, const std::vector<double>& CALC, int epochIndex, size_t sampleIndex)
  {
    return true;
  }
};

// Training progress void (does nothing) implementation

class TrainingProgressReporterVoid : public TrainingProgressReporter
{
  // Does nothing
};

// Back propagation training alogorithm implementation

class NetworkTrainerBackProp : public NetworkTrainer
{
  // Use "/" instead of * during train. Used in unit tests only, should be false on production
  public: bool DIV_IN_TRAIN = false;

  public: double getDeltaOutputSum(ProcNeuronTrainee* outNeuron, double osme) // osme = output sum margin of error (AKA Expected - Calculated)
  {
    if (outNeuron == NULL) { return NaN; }
    double OS = outNeuron->getSum();
    double DOS = outNeuron->SD(OS) * osme;
    return(DOS);
  }

  public: std::vector<double> getDeltaWeights(ProcNeuronTrainee* theNeuron, double dos) // theNeuron in question, dos = delta output sum
  {
    if (theNeuron == NULL) { return std::vector<double>(); } // Empty

    size_t count = theNeuron->inputs.size();
    std::vector<double> DWS(count); // reserve capacity, so we do not need push_back

    double dw;
    for (size_t i = 0; i < count; i++)
    {
      if (DIV_IN_TRAIN) { dw = dos / theNeuron->inputs[i]->get(); } else { dw = dos * theNeuron->inputs[i]->get(); }
      DWS[i] = dw; // DWS.push_back(dw);
    }

    return(DWS);
  }

  public: std::vector<double> getDeltaHiddenSums(ProcNeuronTrainee* theNeuron, double dos) // theNeuron in question, dos = delta output sum
  {
    if (theNeuron == NULL) { return std::vector<double>(); } // Empty

    size_t count = theNeuron->inputs.size();
    std::vector<double> DHS(count); // reserve capacity, so we do not need push_back

    double ds;
    for (size_t i = 0; i < count; i++)
    {
      auto input = dynamic_cast<ProcNeuronTrainee*>(theNeuron->inputs[i]);
      if (input == NULL)
      {
        ds = NaN; // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
      }
      else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
      {
        if (DIV_IN_TRAIN) { ds = dos / theNeuron->w[i] * input->SD(input->getSum()); } else { ds = dos * theNeuron->w[i] * input->SD(input->getSum()); }
      }

      DHS[i] = ds; // DHS.push_back(ds);
    }

    return(DHS);
  }

  protected: void doTrainStepProcPrevLayer(std::vector<BaseNeuron*> &LOUT, std::vector<double> &DOS, int layerIndex)
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

      if (neuron == NULL) { break; } // Non-trainable neuron

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

  public: virtual std::vector<double> trainBySample(Network &NET, const std::vector<double> &DATA, const std::vector<double> &TARG, double speed) override
  {
    // NET=network, DATA=input, TARG=expeted
    // CALC=calculated output (will be calculated)
    // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

    if (std::isnan(speed) ||  (speed <= 0.0)) { speed = 0.1; } // 1.0 is max

    auto CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

    for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
    {
      auto iicount = NET.layers[i]->neurons.size();
      for (size_t ii = 0; ii < iicount; ii++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee *>(NET.layers[i]->neurons[ii]);
        if (neuron != NULL)
        {
          neuron->initTrainStep(); // prepare
        }
      }
    }

    // Output layer (special handling)

    auto &LOUT = NET.layers[NET.layers.size()-1]->neurons;

    std::vector<double> OSME; // output sum margin of error (AKA Expected - Calculated) for each output
    std::vector<double> DOS ; // delta output sum for each output neuron
    std::vector<std::vector<double>> DOIW; // delta output neuron input weights each output neuron

    // proc output layer

    for (size_t i = 0; i < LOUT.size(); i++)
    {
      auto neuron = dynamic_cast<ProcNeuronTrainee *>(LOUT[i]);
      OSME.push_back((TARG[i] - CALC[i]) * speed);
      DOS.push_back(getDeltaOutputSum(neuron, OSME[i])); // will handle neuron=NULL case
      DOIW.push_back(getDeltaWeights(neuron, DOS[i])); // will handle neuron=NULL case
      if (neuron != NULL)
      {
        neuron->addNewWeightsDelta(DOIW[i]);
      }
    }

    // proc prev layers
    // will apply training back recursively
    // recursion controlled by laterIndex

    doTrainStepProcPrevLayer(LOUT, DOS, static_cast<int>(NET.layers.size()-1));

    for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
    {
      auto iicount = NET.layers[i]->neurons.size();
      for (size_t ii = 0; ii < iicount; ii++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee *>(NET.layers[i]->neurons[ii]);
        if (neuron != NULL)
        {
          neuron->applyNewWeights(); // adjust
        }
      }
    }

    return CALC;
  }
};

// Back propagation training fast alogorithm implementation

class NetworkTrainerBackPropFast : public NetworkTrainer
{
  // Use "/" instead of * during train. Used in unit tests only, should be false on production
  public: bool DIV_IN_TRAIN = false;
  
  protected: void addDeltaOutputSum(ProcNeuronTrainee* outNeuron, double osme) // osme = output sum margin of error (AKA Expected - Calculated) // FAST
  {
    if (outNeuron == NULL) { return; }
    double dos = outNeuron->SD(outNeuron->getSum()) * osme;
    outNeuron->addDeltaOutputSum(dos);
  }

  protected: void addDeltaWeights(ProcNeuronTrainee* theNeuron, double dos) // theNeuron in question, dos = delta output sum // FAST
  {
    if (theNeuron == NULL) { return; } // Empty

    size_t count = theNeuron->inputs.size();

    double dw;
    for (size_t i = 0; i < count; i++)
    {
      if (DIV_IN_TRAIN) { dw = dos / theNeuron->inputs[i]->get(); } else { dw = dos * theNeuron->inputs[i]->get(); }
      theNeuron->nw[i] = theNeuron->nw[i] + dw;
    }
  }

  protected: void addDeltaHiddenSums(ProcNeuronTrainee* theNeuron, double dos) // FAST
  {
    if (theNeuron == NULL) { return; } // Empty

    size_t count = theNeuron->inputs.size();

    double ds;
    for (size_t i = 0; i < count; i++)
    {
      auto input = theNeuron->inputGetTrainee(i); // dynamic_cast<ProcNeuronTrainee*>(theNeuron->inputs[i]);
      if (input == NULL)
      {
        // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
      }
      else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
      {
        if (DIV_IN_TRAIN) { ds = dos / theNeuron->w[i] * input->SD(input->getSum()); } else { ds = dos * theNeuron->w[i] * input->SD(input->getSum()); }
        input->addDeltaOutputSum(ds);
      }
    }
  }

  public: virtual std::vector<double> trainBySample(Network &NET, const std::vector<double> &DATA, const std::vector<double> &TARG, double speed) override
  {
    // NET=network, DATA=input, TARG=expeted
    // CALC=calculated output (will be calculated)
    // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

    if (std::isnan(speed) ||  (speed <= 0.0)) { speed = 0.1; } // 1.0 is max

    auto CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

    for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
    {
      auto iicount = NET.layers[i]->neurons.size();
      for (size_t ii = 0; ii < iicount; ii++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee *>(NET.layers[i]->neurons[ii]);
        if (neuron != NULL)
        {
          neuron->initTrainStep(); // prepare
        }
      }
    }

    // proc output layer (special handling)

    if (NET.layers.size() > 0)
    {
      auto &LOUT = NET.layers[NET.layers.size()-1]->neurons;

      for (size_t i = 0; i < LOUT.size(); i++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee *>(LOUT[i]);
        if (neuron != NULL)
        {
          double osme = (TARG[i] - CALC[i]) * speed;
          addDeltaOutputSum(neuron, osme);
          addDeltaWeights(neuron, neuron->getDeltaOutputSum());
          addDeltaHiddenSums(neuron, neuron->getDeltaOutputSum());
        }
      }
    }

    // proc hidden layers, skip input layer

    for (int li = static_cast<int>(NET.layers.size() - 2); li > 0; li--)
    {
      auto& LOUT = NET.layers[li]->neurons;

      for (size_t i = 0; i < LOUT.size(); i++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee*>(LOUT[i]);
        if (neuron != NULL)
        {
          addDeltaWeights(neuron, neuron->getDeltaOutputSum());
          addDeltaHiddenSums(neuron, neuron->getDeltaOutputSum());
        }
      }
    }

    for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
    {
      auto iicount = NET.layers[i]->neurons.size();
      for (size_t ii = 0; ii < iicount; ii++)
      {
        auto neuron = dynamic_cast<ProcNeuronTrainee*>(NET.layers[i]->neurons[ii]);
        if (neuron != NULL)
        {
          neuron->applyNewWeights(); // adjust
        }
      }
    }

    return CALC;
  }
};

inline NetworkTrainer* getDefTrainer() { static NetworkTrainerBackPropFast defTrainer;  return &defTrainer; }

// Class checker for training is done (by results differ from groud truth no more than eps) implementation

class TrainingDoneCheckerEps : public TrainingDoneChecker
{
  const double DEFAULT_EPS = NetworkStat::DEFAULT_EPS;

  protected: double eps = DEFAULT_EPS;

  // Constructor

  public: TrainingDoneCheckerEps()
  {
    this->eps = DEFAULT_EPS;
  }

  public: TrainingDoneCheckerEps(double eps)
  {
    if (std::isnan(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5
    this->eps = eps;
  }

  public: virtual bool trainSampleIsValid(Network& NET, const std::vector<double>& TARG, const std::vector<double>& CALC)
  override
  {
    return NetworkStat::isResultSampleMatchEps(TARG, CALC, eps);
  }
};

// Main training function

/// Train the neural network
inline bool doTrain(Network &NET, const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, TrainingParams *trainingParams, TrainingProgressReporter *trainingProgressReporter = NULL, TrainingDoneChecker *trainingDoneChecker = NULL, NetworkTrainer *trainer = NULL)
{
  TrainingParams trainingParamsDefault;
  if (trainingParams == NULL) { trainingParams = &trainingParamsDefault; }

  TrainingProgressReporterVoid trainingProgressReporterDefault;
  if (trainingProgressReporter == NULL) { trainingProgressReporter = &trainingProgressReporterDefault; }

  TrainingDoneCheckerEps trainingDoneCheckerDefault;
  if (trainingDoneChecker == NULL) { trainingDoneChecker = &trainingDoneCheckerDefault; }

  if (trainer == NULL) { trainer = getDefTrainer(); }

  TrainingDatasetInfo trainingDatasetInfo;
  trainingDatasetInfo.trainDatasetSize = DATAS.size();

  trainer->trainStart(NET, *trainingParams, trainingDatasetInfo);
  trainingDoneChecker->trainStart(NET, *trainingParams, trainingDatasetInfo);
  trainingProgressReporter->trainStart(NET, *trainingParams, trainingDatasetInfo);

  auto MaxN  = trainingParams->maxEpochCount;
  auto speed = trainingParams->speed;

  // steps
  bool isDone = false;
  bool isAbort = false;
  for (int n = 0; (n < MaxN) && (!isDone) && (!isAbort); n++)
  {
    trainer->trainEposhStart(NET, n);
    trainingDoneChecker->trainEposhStart(NET, n);
    trainingProgressReporter->trainEposhStart(NET, n);

    if (!trainingParams->fastVerify)
    {
      // strict verify
      isDone = true;
      for (size_t s = 0; (s < DATAS.size()) && (!isAbort); s++)
      {
        auto CALC = doProc(NET, DATAS[s]);

        if (!trainingDoneChecker->trainSampleIsValid(NET, TARGS[s], CALC))
        {
          isDone = false;
        }

        if (!trainingProgressReporter->trainSampleReportAndCheckContinue(NET, DATAS[s], TARGS[s], CALC, n, s))
        {
          isDone = false;
          isAbort = true;
        }
      }

      if ((!isDone) && (!isAbort))
      {
        for (size_t s = 0; s < DATAS.size(); s++)
        {
          trainer->trainBySample(NET, DATAS[s], TARGS[s], speed);
        }
      }
    }
    else
    {
      // fast verify
      isDone = true;
      for (size_t s = 0; (s < DATAS.size()) && (!isAbort); s++)
      {
        auto CALC_BEFORE_TRAIN = trainer->trainBySample(NET, DATAS[s], TARGS[s], speed);

        // we use calc before train as input
        // strictly speaking this is incorrect, 
        // but we assume that single training step will not affect stats very much and will only improve results

        const auto& CALC = CALC_BEFORE_TRAIN;

        if (!trainingDoneChecker->trainSampleIsValid(NET, TARGS[s], CALC))
        {
          isDone = false;
        }

        if (!trainingProgressReporter->trainSampleReportAndCheckContinue(NET, DATAS[s], TARGS[s], CALC, n, s))
        {
          isDone = false;
          isAbort = true;
        }
      }
    }

    trainer->trainEposhEnd(NET, n);
    trainingDoneChecker->trainEposhEnd(NET, n);
    trainingProgressReporter->trainEposhEnd(NET, n);
 }

  trainer->trainEnd(NET, isDone);
  trainingDoneChecker->trainEnd(NET, isDone);
  trainingProgressReporter->trainEnd(NET, isDone);

  return(isDone);
}

// doTrain with alternative params format

inline bool doTrain(Network* NET, const std::vector<std::vector<double>>& DATAS, const std::vector<std::vector<double>>& TARGS, TrainingParams* trainingParams, TrainingProgressReporter* trainingProgressReporter = NULL, TrainingDoneChecker* trainingDoneChecker = NULL)
{
  return doTrain(*NET, DATAS, TARGS, trainingParams, trainingProgressReporter, trainingDoneChecker);
}

inline bool doTrain(Network* NET, const std::vector<std::vector<double>>* DATAS, const std::vector<std::vector<double>>* TARGS, TrainingParams* trainingParams, TrainingProgressReporter* trainingProgressReporter = NULL, TrainingDoneChecker* trainingDoneChecker = NULL)
{
  return doTrain(*NET, *DATAS, *TARGS, trainingParams, trainingProgressReporter, trainingDoneChecker);
}

/*
// Exports

// Some internals (Similar API to JS version)

NN.Internal = {};
NN.Internal.getPRNG = getPRNG;
NN.Internal.getRandom = getRandom;
NN.Internal.getRandomInt = getRandomInt;

// Activation

NN.ActFuncSigmoid = ActFuncSigmoid;
NN.ActFuncSigmoidTrainee = ActFuncSigmoidTrainee;
NN.ActFuncRELU = ActFuncRELU;
NN.ActFuncRELUTrainee = ActFuncRELUTrainee;
NN.ActFuncLRELU = ActFuncLRELU;
NN.ActFuncLRELUTrainee = ActFuncLRELUTrainee;
NN.ActFuncTanh = ActFuncTanh;
NN.ActFuncTanhTrainee = ActFuncTanhTrainee;

// Core

NN.BaseNeuron = BaseNeuron;
NN.InputNeuron = InputNeuron; NN.dynamicCastInputNeuron = dynamicCastInputNeuron;
NN.ProcNeuron = ProcNeuron; NN.dynamicCastProcNeuron = dynamicCastProcNeuron;
NN.ProcNeuronTrainee = ProcNeuronTrainee; NN.dynamicCastProcNeuronTrainee = dynamicCastProcNeuronTrainee;
NN.BiasNeuron = BiasNeuron;

NN.Layer       = Layer;
NN.Network     = Network;
NN.doProc      = doProc;

NN.TheNeuronFactory = TheNeuronFactory;
NN.ExtNeuronFactory = ExtNeuronFactory;

// Math

NN.NetworkStat = NetworkStat;

// Training

NN.TrainingParams = TrainingParams;
NN.TrainingDatasetInfo = TrainingDatasetInfo;
NN.TrainingProccessor = TrainingProccessor;
NN.TrainingDoneChecker = TrainingDoneChecker;
NN.TrainingProgressReporter = TrainingProgressReporter;
NN.TrainingProgressReporterVoid = TrainingProgressReporterVoid;
NN.NetworkTrainer = NetworkTrainer;
NN.NetworkTrainerBackProp = NetworkTrainerBackProp;
NN.NetworkTrainerBackPropFast = NetworkTrainerBackPropFast;
NN.getDefTrainer = getDefTrainer;
NN.TrainingDoneCheckerEps = TrainingDoneCheckerEps;
NN.doTrain = doTrain;
*/

} // NN

#endif
