#include "prng.hpp"

int32_t Random::Next()
{
    return this->seed = this->seed * 16807 % 2147483647;
}

Random::Random(int32_t seed)
{
    this->seed = seed % 2147483647; // 0x7FFFFFF
    if (this->seed <= 0) this->seed += 2147483646;
}

double Random::NextFloat()
{
    // We know that result of next() will be 1 to 2147483646 (inclusive).
    return (this->Next() - 1) / 2147483646.0;
}
