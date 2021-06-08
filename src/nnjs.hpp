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

// Use "/" instead of * during train. Used in unit tests only, should be false on production
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

class NonAssignable // derive from this to prevent copy of move
{
  private: NonAssignable(NonAssignable const&) { }
  private: NonAssignable& operator=(NonAssignable const&) { }
  public:  NonAssignable() {}
};

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

// Neuron types

// Neuron have to have following functions:
// // Base                                                        Input             Normal             Bias  
// .get()                        - to provide its current result  value of .set     result of .proc    1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links  [ignored]
// // Input
// .set(inp)                     - to assing input value          assing inp value  [N/A]              [N/A]
// // Proc
// .inputs[] (used for train)    - current input Neurons links    [N/A]             [input link count] [] // empty
// .w[]      (used for train)    - current input Neurons weights  [N/A]             [input link count] [] // empty
// // Construction
// .addInput(Neuron,weight)      - add input link to Neuron       [N/A]             add input link     [ignored]
// .addInputAll(Neurons,weights) - add input link to Neurons      [N/A]             add input links    [ignored]
// // Train
// .getSum()                     - raw sum of all inputs before S [N/A]             sum of .proc       1.0
// .nw[]                         - new input Neurons weights      [N/A]             [input link count] [] // empty
// .initNewWeights()             - init new  weights (.nw) array  [N/A]             copy .w to .nw     [ignored]
// .addNewWeightsDelta(DW)       - adds DW to new  weights (.nw)  [N/A]             add dw to .nw      [ignored]
// .applyNewWeights()            - adds DW to new  weights (.nw)  [N/A]             copy .nw to .w     [ignored]

class BaseNeuron : protected NonAssignable
{
  // Returns "state" value that neuron currently "holds" (state updated by proc() function)
  public: virtual double get() = 0; // abstract

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

// Proc Neuron
// Neuron that proccess its input inside proc method

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

  // Core proccsing
  // Computes output based on input

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

  // for train

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

  // Replace current weights with new weights

  public: void applyNewWeights()
  {
    //this->w.assign(this->nw.begin(), this->nw.end());
    this->w = this->nw;
  }
};

// BiasNeuron
// Always return 1.0 as its output

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

/// Layer
/// Represent a layer of network
/// This class composes neuron network layer and acts as container for neurons
/// Layer Container "owns" Neuron(s)

class Layer : protected NonAssignable
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
  if (theNeuron == NULL) { return std::vector<double>(); } // Empty

  size_t count = theNeuron->inputs.size();
  std::vector<double> DWS(count); // reserve capacity, so we do not need push_back

  double dw;
  for (size_t i = 0; i < count; i++)
  {
    if (DIV_IN_TRAIN) { dw = DOS / theNeuron->inputs[i]->get(); } else { dw = DOS * theNeuron->inputs[i]->get(); }
    DWS[i] = dw; // DWS.push_back(dw);
  }

  return(DWS);
}

inline std::vector<double> getDeltaHiddenSums(ProcNeuronTrainee *theNeuron, double DOS) // theNeuron in question, DOS = delta output sum
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
      ds = NAN; // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
    }
    else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
    {
      if (DIV_IN_TRAIN) { ds = DOS / theNeuron->w[i] * SD(input->getSum()); } else { ds = DOS * theNeuron->w[i] * SD(input->getSum()); }
    }

    DHS[i] = ds; // DHS.push_back(ds);
  }

  return(DHS);
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
      if (R[i] > R[currMaxIndex]) { prevMaxIndex = currMaxIndex; currMaxIndex = i; }
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
      if (R[i] > R[currMaxIndex]) { currMaxIndex = i; }
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

  // Aggregated error

  const double AGG_ERROR_DIVIDED_BY = 2.0; // to be used as error function, should be mutiplied by 1/2 so derivative will not have 2x in front

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

  inline double getResultSetAggErrorSum(const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS)  // private
  {
    auto count = TARGS.size();

    assert(count == CALCS.size());

    double result = 0;

    for (size_t s = 0; s < count; s++)
    {
      result += getResultSampleAggErrorSum(TARGS[s], CALCS[s]);
    }

    return(result);
  }

  inline double getResultSetAggError(const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS)
  {
    auto result = getResultSetAggErrorSum(TARGS, CALCS);
    auto count = TARGS.size();
    if (count > 0) { count *= TARGS[0].size(); }
    if (count <= 0) { return NAN; }
    return(result / count / AGG_ERROR_DIVIDED_BY);
  }

  inline double getResultSetAggErrorByAggErrorSum(double sum, size_t sampleSize, size_t samplesCount = 1)
  {
    auto count = samplesCount;
    if (count > 0) { count *= sampleSize; }
    if (count <= 0) { return NAN; }
    return(sum / count / AGG_ERROR_DIVIDED_BY);
  }

  // Misc

  inline std::vector<double> getR1Array(int index, int total, double SET = 1, double NOTSET = 0)
  {
    // Retuns array with only one index of total item set to SET(=1) and all other as NOTSET(=0): 0=[1, 0, 0 ...], 1=[0, 1, 0, ...], 2=[0, 0, 1, ...]

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

inline void doTrainStep(Network &NET, const std::vector<double> &DATA, const std::vector<double> &TARG, double SPEED)
{
  // NET=network, DATA=input, TARG=expeted
  // CALC=calculated output (will be calculated)
  // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

  if (std::isnan(SPEED) ||  (SPEED <= 0.0)) { SPEED = 0.1; } // 1.0 is max

  auto CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

  for (size_t i = 1; i < NET.layers.size(); i++) // skip input layer
  {
    auto iicount = NET.layers[i]->neurons.size();
    for (size_t ii = 0; ii < iicount; ii++)
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

  // proc output layer

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
  // will apply training back recursively
  // recursion controlled by laterIndex

  doTrainStepProcPrevLayer(LOUT, DOS, NET.layers.size()-1);

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
}

/// Class checker for training is done

class TrainingDoneChecker
{
  /// Function checks if training is done
  /// DATAS is a list of source data sets
  /// TARGS is a list of target data sets
  /// CALCS is a list of result data sets
  public: virtual bool isTrainingDone(const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS) = 0;
};

const double DEFAULT_EPS = 0.1;

class TrainingDoneCheckerEps : public TrainingDoneChecker
{
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

  // simple single vectors match

  public: static bool isTrainingDoneSimple(const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS, double eps)
  {
    assert(TARGS.size() == CALCS.size());

    for (size_t s = 0; s < TARGS.size(); s++)
    {
      if (!NetworkStat::isResultSampleMatchEps(TARGS[s], CALCS[s], eps))
      {
        return(false);
      }
    }

    return(true);
  }

  public: virtual bool isTrainingDone(const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, const std::vector<std::vector<double>> &CALCS)
  override
  {
    assert(DATAS.size() == TARGS.size());
    assert(TARGS.size() == CALCS.size());
    return(isTrainingDoneSimple(TARGS, CALCS, eps));
  }
};

/// Training progress reporter
/// Will be called during traing to report progress
/// May rise traing abort event

class TrainingProgressReporter
{
  // TrainingArgs parameter

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

  // TrainingStep parameter
  // for onTrainingStep

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

  // Report methods

  public: virtual void onTrainingBegin(TrainingArgs *args) { }
  public: virtual bool onTrainingStep (TrainingArgs *args, TrainingStep *step) { return true; } // return false to abort training
  public: virtual void onTrainingEnd  (TrainingArgs *args, bool isOk) { }
};

// Main training function

const int    DEFAULT_TRAIN_COUNT    = 50000;
const double DEFAULT_TRAINING_SPEED = 0.125;

/// Train the neural network
inline bool doTrain(Network &NET, const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, double SPEED = -1, int MAX_N = -1, TrainingProgressReporter *progressReporter = NULL, TrainingDoneChecker *isTrainingDoneChecker = NULL)
{
  if (MAX_N < 0)       { MAX_N = DEFAULT_TRAIN_COUNT; }
  if (SPEED < 0)       { SPEED = DEFAULT_TRAINING_SPEED; }

  TrainingDoneCheckerEps isTrainingDoneCheckerDefault;

  if (isTrainingDoneChecker == NULL) { isTrainingDoneChecker = &isTrainingDoneCheckerDefault; }

  TrainingProgressReporter::TrainingArgs trainArgs(NET, DATAS, TARGS, SPEED, MAX_N);

  if (progressReporter != NULL) { progressReporter->onTrainingBegin(&trainArgs); }

  // steps
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
      TrainingProgressReporter::TrainingStep step(CALCS, n);

      if (!progressReporter->onTrainingStep(&trainArgs, &step))
      {
        // Abort training
        progressReporter->onTrainingEnd(&trainArgs, false);
        return(false);
      }
    }

    isDone = isTrainingDoneChecker->isTrainingDone(DATAS, TARGS, CALCS);

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

/*
// Exports

// Some internals

NN.Internal = {};
NN.Internal.PRNG = PRNG;
NN.Internal.getRandom = getRandom;
NN.Internal.getRandomInt = getRandomInt;
NN.Internal.getDeltaOutputSum  = getDeltaOutputSum;
NN.Internal.getDeltaWeights    = getDeltaWeights;
NN.Internal.getDeltaHiddenSums = getDeltaHiddenSums;

// Core

NN.ProcNeuron  = ProcNeuron;
NN.InputNeuron = InputNeuron;
NN.BiasNeuron  = BiasNeuron;
NN.Layer       = Layer;
NN.doProc      = doProc;

// Math

NN.NetworkStat = NetworkStat;

// Training

NN.TrainingDoneChecker = TrainingDoneChecker;
NN.TrainingDoneCheckerEps = TrainingDoneCheckerEps;
NN.TrainingProgressReporter = TrainingProgressReporter;
NN.doTrain = doTrain;
*/

} // NN

#endif
