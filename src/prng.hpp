#ifndef PRNG_HPP
#define PRNG_HPP

#include <stdint.h>

// Simple Random value generator
// Uses an optimized version of the Park-Miller PRNG.
// Inspired by 
// http://www.firstpr.com.au/dsp/rand31/
// https://gist.github.com/blixt/f17b47c62508be59987b
// Open Source Software under MIT License

class Random
{
  protected: int64_t seed;

  public: void srand(int32_t seed); // C like API
  public: void setSeed(int32_t seed) { srand(seed); } // Synonym
  public: void SetSeed(int32_t seed) { srand(seed); } // Synonym

  public: Random(int32_t seed) { srand(seed); }

  public: Random() : Random(42) { }

  public: enum _:int32_t
  {
    /// Min output value for next() == 1
    NEXT_MIN = 1,

    /// Max output value for next() == 2^31-2
    NEXT_MAX = 2147483646,
  };

  /// Returns a pseudo-random value between NEXT_MIN (1) and NEXT_MAX (2^32 - 2) [NEXT_MIN .. NEXT_MAX] (inclusive)
  public: int32_t next();
  public: int32_t Next() { return next(); } // Synonym

  /// Returns a pseudo-random floating point number in range [0.0 .. 1.0) (upper bound exclsive)
  public: double nextFloat();
  public: double NextFloat() { return nextFloat(); } // Synonym

  // Like C Random

  public: enum __:int32_t
  {
    /// Maximum output value for rand() == [0..RAND_MAX_VALUE]
    RAND_MAX_VALUE = (_::NEXT_MAX - _::NEXT_MIN),
    #if !defined(RAND_MAX)
    RAND_MAX = RAND_MAX_VALUE, // Unfortunately RAND_MAX is a macro in most cases, so we cannot define this identifier in that case
    #endif
  };

  /// Returns next random in range [0..RAND_MAX_VALUE] (inclusive)
  public: int32_t rand();

  /// randFloat()        Returns random in range [0.0 .. 1.0] (inclusive)
  public: double randFloat();

  /// randFloat(max)     Returns random in range [0.0 .. max] (inclusive)
  public: double randFloat(double max);

  /// randFloat(min,max) Returns random in range [min .. max] (inclusive)
  public: double randFloat(double min, double max);

  public: template<typename T> static int32_t getRandomSeed(T timeVal)
  {
    return static_cast<int32_t>(timeVal) % 0x7FFF0000 + 1;
  }
};

#endif
