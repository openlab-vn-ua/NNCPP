// Simple Random value generator
// Open Source Software under MIT License

#include "prng.hpp"

void Random::srand(int32_t seed)
{
  if (seed < 0) { seed = -seed; }
  if (seed == 0) { seed = 1; }
  this->seed = seed % 2147483647;
}

int32_t Random::next()
{
  this->seed = this->seed * 16807 % 2147483647;
  return static_cast<int32_t>(this->seed);
}

double Random::nextFloat()
{
  // We know that result of next() will be 1 to 2147483646 (inclusive).

  //single-line call, but may emit warnings
  //return static_cast<double>(this->next() - this->NEXT_MIN) / (this->NEXT_MAX - this->NEXT_MIN + 1);

  //muti step call with explict operations to prevent warning
  int32_t next = this->next(); next -= this->NEXT_MIN; // 0-based value index
  int32_t ncnt = this->NEXT_MAX; ncnt -= this->NEXT_MIN; ncnt += 1; // number of distinct prng values
  return static_cast<double>(next) / ncnt;
}

int32_t Random::rand()
{
  return this->next() - this->NEXT_MIN;
}

double Random::randFloat()
{
  return static_cast<double>(this->rand()) / this->RAND_MAX_VALUE;
}

double Random::randFloat(double max)
{
  return randFloat() * max;
}

double Random::randFloat(double min, double max)
{
  double diff = max - min;
  return randFloat() * diff + min;
}
