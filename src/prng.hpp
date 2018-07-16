#ifndef PRNG_HPP
#define PRNG_HPP

#include <stdint.h>

/**
 * Creates a pseudo-random value generator. The seed must be an integer.
 * Uses an optimized version of the Park-Miller PRNG.
 * http://www.firstpr.com.au/dsp/rand31/
 * TODO: Returns only positive values. Rename me to PosRandom?
 */

class Random
{
  protected: int64_t seed;

  public: Random(int32_t seed);

  public: Random() : Random(42) { }

  /// Returns a pseudo-random value between 1 and 2^32 - 2.
  public: int32_t Next();
  public: int32_t next() { return Next(); } // synonim

  /// Returns a pseudo-random floating point number in range [0, 1).
  public: double NextFloat();
  public: double nextFloat() { return NextFloat(); } // synonim
};

#endif
