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

  protected: int  reportInterval = 0;
  protected: bool reportSamples = false;

  protected: int  lastSeenIndex = 0;

  protected: TimeMetter beginTimeMetter;

  public: TrainingProgressReporterConsole(int reportIntervalIn = DEFAULT_REPORT_INTERVAL, bool reportSamplesIn = false) 
          : reportInterval(reportIntervalIn), reportSamples(reportSamplesIn)
  {
    if (reportInterval < 0) { reportInterval = 0; }
  }

  public: template<typename T1> static std::string STR(const T1& x) { return console::toString(x); };

  // methods/callbacks

  public: virtual void onTrainingBegin(TrainingArgs* args)
  { 
    console::log("TRAINING Started", args->SPEED); 
    beginTimeMetter.start();
  }

  public: virtual bool onTrainingStep(TrainingArgs* args, TrainingStep* step)
  {
    lastSeenIndex = step->stepIndex;

    auto  n = step->stepIndex + 1;
    auto  MAX_N = args->maxStepsCount;
    auto& DATAS = args->DATAS;
    auto& TARGS = args->TARGS;
    auto& CALCS = step->CALCS;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      auto variance = NN::NetworkStat::getResultSetAggError(TARGS, CALCS);
      console::log("TRAINING AggError[n,s]", MAX_N, n, variance);
      if (reportSamples)
      {
        for (size_t s = 0; s < DATAS.size(); s++)
        {
          console::log("TRAINING Result.N[n,s]", MAX_N, n, s, DATAS[s], TARGS[s], CALCS[s]);
        }
      }
    }

    return true;
  }

  public: virtual void onTrainingEnd(TrainingArgs* args, bool isOk)
  {
    beginTimeMetter.stop();

    auto n = lastSeenIndex + 1;
    auto &NET = args->NET;

    auto spentTime = beginTimeMetter.millisPassed(); // ms
    if (spentTime <= 0) { spentTime = 1; }

    auto steps = args->DATAS.size() * n;
    auto scale = NN::NetworkStat::getNetWeightsCount(NET) * args->DATAS.size() * n;
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