#pragma once
#ifndef NNJS_TIME_METTER_HPP
#define NNJS_TIME_METTER_HPP

// Simple operation timer checker
// ----------------------------------------------------

#include <chrono>

namespace NN {

class TimeMetter
{
  public: typedef long tvalue_t; // long long

  public: static tvalue_t millisGlobal()
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  protected: enum _:tvalue_t { VOID_TIME = -1 };

  protected: tvalue_t timeStart = VOID_TIME;
  protected: tvalue_t timeStop  = VOID_TIME;

  // Methods

  public: void start()
  {
    timeStart = millisGlobal();
    timeStop  = VOID_TIME;
  }

  public: tvalue_t stop()
  {
    if (timeStart == VOID_TIME) { return -1; }
    timeStop = millisGlobal();
    return timeStop - timeStart;
  }

  public: bool isStarted()
  {
    return (timeStart != VOID_TIME);
  }

  public: bool isStoped()
  {
    return (timeStop != VOID_TIME);
  }

  public: tvalue_t millisPassed()
  {
    if (timeStart == VOID_TIME) { return -1; }
    if (timeStop == VOID_TIME) { return millisGlobal() - timeStart; }
    return timeStop - timeStart;
  }

  // Constructor

  public: TimeMetter()
  {
    //start();
  }
};

}

#endif

