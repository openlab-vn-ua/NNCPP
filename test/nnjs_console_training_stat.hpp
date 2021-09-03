#ifndef NNJS_CONSOLE_TRAINING_STAT_HPP
#define NNJS_CONSOLE_TRAINING_STAT_HPP

// Simple Neural Network toolkit
// Open Source Software under MIT License
// Console training stat dumper

#include <nnjs.hpp>
#include "nnjs_console.hpp"
#include "nnjs_time_metter.hpp"
#include <string>

// Console reporter

namespace NN {

// Simple console write trainging reporter

class TrainingProgressReporterConsole : public TrainingProgressReporter
{
  public: enum _ { DEFAULT_REPORT_INTERVAL = 100 };

  // Constructor

  protected: int    reportInterval = 0;
  protected: bool   reportSamples = false;

  protected: int    maxEpochCount = 0;
  protected: size_t samplesDone   = 0;

  protected: int    lastSeenIndex = 0;

  protected: double aggErrorSum   = NaN;
  protected: size_t aggValCount   = 0;

  protected: TimeMetter beginTimeMetter;

  public: TrainingProgressReporterConsole(int reportIntervalIn = DEFAULT_REPORT_INTERVAL, bool reportSamplesIn = false) 
          : reportInterval(reportIntervalIn), reportSamples(reportSamplesIn)
  {
    if (reportInterval < 0) { reportInterval = 0; }
  }

  public: template<typename T1> static std::string STR(const T1& x) { return console::asString(x); };

  // methods/callbacks

  public: virtual void trainStart(Network& NET, TrainingParams& trainingParams, TrainingDatasetInfo& datasetInfo)
  override
  { 
    maxEpochCount = trainingParams.maxEpochCount;
    samplesDone = 0;
    console::log("TRAINING Started", "speed:"+STR(trainingParams.speed), "fastVerify:"+STR(trainingParams.fastVerify));
    beginTimeMetter.start();
  }

  public: virtual void trainEposhStart(Network& NET, int epochIndex)
  override
  {
    lastSeenIndex = epochIndex;
    aggErrorSum = NaN;
    aggValCount = 0;
  }

  public: virtual bool trainSampleReportAndCheckContinue(Network& NET, const std::vector<double>& DATA, const std::vector<double>& TARG, const std::vector<double>& CALC, int epochIndex, size_t sampleIndex)
  override
  {
    samplesDone++;

    auto n = epochIndex + 1;
    auto s = sampleIndex;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      if (std::isnan(aggErrorSum)) { aggErrorSum = 0.0; }

      aggErrorSum += NN::NetworkStat::getResultSampleAggErrorSum(TARG, CALC);
      aggValCount += TARG.size();

      if (reportSamples)
      {
        console::log("TRAINING Result.N[n,s]", maxEpochCount, n, s, DATA, TARG, CALC);
      }
    }

    return true;
  }

  public: virtual void trainEposhEnd(Network& NET, int epochIndex)
  override
  {
    auto n = epochIndex + 1;
    auto MAX_N = maxEpochCount;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      auto variance = NN::NetworkStat::getResultAggErrorByAggErrorSum(aggErrorSum, aggValCount);
      console::log("TRAINING AggError[n,s]", MAX_N, n, variance);
    }
  }

  public: virtual void trainEnd(Network& NET, bool isOk)
  override
  {
    beginTimeMetter.stop();

    auto n = lastSeenIndex + 1;

    auto spentTime = beginTimeMetter.millisPassed(); // ms
    if (spentTime <= 0) { spentTime = 1; }

    auto steps = samplesDone;
    auto scale = NN::NetworkStat::getNetWeightsCount(NET) * steps;
    auto speed = round((1.0 * scale / spentTime));

    auto stepTime = round(((1.0 * spentTime) / steps) * 1000.0);

    if (isOk)
    {
      console::log("TRAINING OK", "iterations:" + STR(n), "time:" + STR(spentTime) + " ms", "speed:" + STR(speed) + "K w*s/s", "step:" + STR(stepTime) + " us", NET.layers);
    }
    else
    {
      console::log("TRAINING FAILED", "timeout:" + STR(n), NET.layers);
    }
  }
};

}

#endif