// Simple Neural Network toolkit
// Open Source Software under MIT License
// [OCR demo network]

#include <ctime>
#include <string>
#include <vector>

#include <nnjs.hpp>
#include "nnjs_console.hpp"
#include "nnjs_console_training_stat.hpp"

namespace NN { namespace Demo {

// Utils
// ----------------------------------------------------

template<typename T1> std::string STR(const T1& x) { return console::toString(x); }

// Ocr sample
// ----------------------------------------------------

const int SAMPLE_OCR_SX = 9;
const int SAMPLE_OCR_SY = 8;

inline std::string emptyStr() { return ""; }

std::vector<std::vector<std::vector<double>>> sampleOcrGetSamples()
{
  // Return array of array of input samples of each letter [ [IA0, IA1, IA2 ...], [IB0, ...], [IC0, ...] ]

  auto IA0 = emptyStr()
           +"         "
           +"    *    "
           +"   * *   "
           +"  *   *  "
           +"  *****  "
           +"  *   *  "
           +"  *   *  "
           +"         "
           +"";

  auto IA1 = emptyStr()
           +"         "
           +"    *    "
           +"  ** **  "
           +" **   ** "
           +" ******* "
           +" **   ** "
           +" **   ** "
           +"         "
           +"";

  auto IA2 = emptyStr()
           +"         "
           +"   ***   "
           +"  *   *  "
           +"  *   *  "
           +"  *****  "
           +"  *   *  "
           +" *** *** "
           +"         "
           +"";

  auto IB0 = emptyStr()
           +"         "
           +"  ****   "
           +"  *   *  "
           +"  ****   "
           +"  *   *  "
           +"  *   *  "
           +"  ****   "
           +"         "
           +"";

  auto IB1 = emptyStr()
           +"         "
           +" *****   "
           +" **  **  "
           +" *****   "
           +" **   ** "
           +" **   ** "
           +" *****   "
           +"         "
           +"";

  auto IC0 = emptyStr()
           +"         "
           +"   ***   "
           +"  *   *  "
           +"  *      "
           +"  *      "
           +"  *   *  "
           +"   ***   "
           +"         "
           +"";

  auto IC1 = emptyStr()
           +"         "
           +"   ***   "
           +" **   ** "
           +" **      "
           +" **      "
           +" **   ** "
           +"   ***   "
           +"         "
           +"";

  auto ID0 = emptyStr()
           +"         "
           +"  ****   "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  ****   "
           +"         "
           +"";

  auto ID1 = emptyStr()
           +"         "
           +" *****   "
           +" **   ** "
           +" **   ** "
           +" **   ** "
           +" **   ** "
           +" *****   "
           +"         "
           +"";

  auto ID2 = emptyStr()
           +"         "
           +" *****   "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +" *****   "
           +"         "
           +"";

  auto getLNArray = [](const std::string &L) -> std::vector<double>
  {
    // Convert letter from fancy text to plain array of 1 and 0
    std::vector<double> R;
    for (auto y = 0; y < SAMPLE_OCR_SY; y++)
    {
      for (auto x = 0; x < SAMPLE_OCR_SX; x++)
      {
        R.push_back(L[y*SAMPLE_OCR_SX+x] == '*' ? 1 : 0);
      }
    }

    return(R);
  };

  // letters to recognize, each SX * SY size in many samples
  std::vector<std::vector<std::vector<double>>> I0 = {
             { getLNArray(IA0), getLNArray(IA1), getLNArray(IA2), },
             { getLNArray(IB0), getLNArray(IB1) },
             { getLNArray(IC0), getLNArray(IC1) },
             { getLNArray(ID0), getLNArray(ID1), getLNArray(ID2) }
           }; 

  return(I0);
}

// Train input preparation

auto NOISE_TYPE_PIXEL_FLIP = 0;
auto NOISE_TYPE_PIXEL_DARKER_LIGHTER = 1;
auto NOISE_TYPE_PIXEL_RANDOM = 2;

std::vector<double> getNoisedInput(const std::vector<double> &L, int noiseCount = 0, int noiseType = NOISE_TYPE_PIXEL_FLIP)
{
  // type: 0=flip pixel, 1=drarker/lighter
  //if (noiseType == null) { noiseType = NOISE_TYPE_PIXEL_FLIP; }

  auto makeNoise = [noiseType](double value) -> double
  {
    if (noiseType == NOISE_TYPE_PIXEL_DARKER_LIGHTER)
    {
      if (value <= 0) { return(NN::Internal::getRandom(0.0 , 0.49)); }
      if (value >= 1) { return(NN::Internal::getRandom(0.51, 1.0 )); }
      return(value);
    }

    if (noiseType == NOISE_TYPE_PIXEL_RANDOM)
    {
      return(NN::Internal::getRandom(0.0,1.0));
    }

    return(1-value); // flip pixel
  };

  //if (noiseCount == null) { noiseCount = 0; }

  auto R = L; // L.slice(); // copy

  for (auto i = 0; i < noiseCount; i++)
  {
    auto noiseIndex = NN::Internal::getRandomInt(R.size());
    R[noiseIndex] = makeNoise(R[noiseIndex]);
  }

  return(R);
}

std::vector<double> getShiftedImg(const std::vector<double> &L, int sx = 0, int sy = 0)
{
  //if (sx == null) { sx = 0; }
  //if (sy == null) { sy = 0; }

  std::vector<double> R;

  for (auto y = 0; y < SAMPLE_OCR_SY; y++)
  {
    for (auto x = 0; x < SAMPLE_OCR_SX; x++)
    {
      auto ox = (x + -sx); ox = (ox < 0) ? SAMPLE_OCR_SX+ox : ox; ox %= SAMPLE_OCR_SX;
      auto oy = (y + -sy); oy = (oy < 0) ? SAMPLE_OCR_SY+oy : oy; oy %= SAMPLE_OCR_SY;
      R.push_back(L[oy*SAMPLE_OCR_SX+ox]);
    }
  }

  return(R);
}

std::vector<std::string> sampleAddLetTexts(const std::vector<double> &L, std::vector<std::string> &inT, bool addTopSep = true, bool addLeftSep = true, bool addBottomSep = true, bool addRightSep = true)
{
  #define USE_ASCII 1

  auto inText = [&inT](int i) -> std::string
  {
    if (inT.empty()) { return(""); } else { return(inT[i]); }
  };

  // T will be SAMPLE_OCR_SY+1+1 height

  std::vector<std::string> T;

  // !00000000000! top Sep
  // !<text Y[0]>
  // !<text Y[SAMPLE_OCR_SY-1]>
  // !00000000000! bottom Sep

  int ty;

  ty = 0;
  if (addTopSep) { T.push_back(inText(ty++)); }
  for (auto y = 0; y < SAMPLE_OCR_SY; y++)
  {
    T.push_back(inText(ty++));
  }
  if (addBottomSep) { T.push_back(inText(ty++)); }

  auto t = emptyStr(); 
  if (addLeftSep) { t += "!"; }
  for (auto x = 0; x < SAMPLE_OCR_SX; x++)
  {
    t += "-";
  }
  if (addRightSep) { t += "!"; }

  auto SEP = t; // sep line

  ty = 0;
  if (addTopSep) { T[ty++] += SEP; }
  for (auto y = 0; y < SAMPLE_OCR_SY; y++)
  {
    auto t = emptyStr();
    if (addLeftSep) { t += "!"; }
    for (auto x = 0; x < SAMPLE_OCR_SX; x++)
    {
      auto v = L[y*SAMPLE_OCR_SX+x];
      auto c = "";

      if (v <= 0)
      {
        c = " ";
      }
      else if (v >= 1)
      {
        #if USE_ASCII
          c = "*"; // "*";
        #else
          c = "\u2588"; // "█";
        #endif
      }
      else
      {
        // v = Math.floor(v * 10); c = v.toString()[0];
        v = floor(v * 10);
        #if USE_ASCII
        const char *(F[])={"0",      "1",      "2",      "3",      "4",      "5",      "6",      "7",      "8",      "9"      }; // "0123456789";
        #else
        const char *(F[])={"\u2591", "\u2591", "\u2591", "\u2592", "\u2592", "\u2592", "\u2593", "\u2593", "\u2593", "\u2593" }; // "░░░▒▒▒▒▓▓▓";
        #endif
        c = F[(int)v];
      }

      t += c;
    }
    if (addRightSep) { t += "!"; }
    T[ty++] += t;
  }
  if (addBottomSep) { T[ty++] += SEP; }

  #undef  USE_ASCII
  return(T);
}

bool sampleOcrNetwork()
{
  // The Dataset

  if (true)
  {
    auto seed = time(NULL) % 0x7FFF0000 + 1;
    NN::Internal::PRNG.setSeed(seed);
    console::log("sampleOcrNetwork", "(samples)", "seed=", seed);
  }

  auto SAMPLES = sampleOcrGetSamples(); // [letter][sample] = data[]
  std::vector<std::vector<double>> RESULTS; // [letter] = R1 array

  for (size_t dataIndex = 0; dataIndex < SAMPLES.size(); dataIndex++)
  {
    auto RESULT = NN::NetworkStat::getR1Array(dataIndex,SAMPLES.size()); // target result for this letter
    RESULTS.push_back(RESULT);
  }

  // The Net

  auto LAYERS = 3;
  NN::Network NET;

  if (true)
  {
    auto seed = time(NULL) % 0x7FFF0000 + 1;
    NN::Internal::PRNG.setSeed(seed);
    console::log("sampleOcrNetwork", "(net)", "seed=", seed, "layers=", LAYERS);
  }

  if (LAYERS == 3)
  {
    auto IN  = new NN::Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN::TheNeuronFactory<NN::InputNeuron>{}); IN->addNeuron(NN::TheNeuronFactory<NN::BiasNeuron>{});
    auto L1  = new NN::Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L1->addNeuron(NN::TheNeuronFactory<NN::BiasNeuron>{}); L1->addInputAll(IN);
    auto OUT = new NN::Layer(SAMPLES.size(), NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); OUT->addInputAll(L1); // Outputs: 0=A, 1=B, 2=C, ...
    NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);
  }
  else
  {
    auto IN  = new NN::Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN::TheNeuronFactory<NN::InputNeuron>{}); IN->addNeuron(NN::TheNeuronFactory<NN::BiasNeuron>{});
    auto L1  = new NN::Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L1->addNeuron(NN::TheNeuronFactory<NN::BiasNeuron>{}); L1->addInputAll(IN);
    auto L2  = new NN::Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); L2->addNeuron(NN::TheNeuronFactory<NN::BiasNeuron>{}); L2->addInputAll(L1);
    auto OUT = new NN::Layer(SAMPLES.size(), NN::TheNeuronFactory<NN::ProcNeuronTrainee>{}); OUT->addInputAll(L2); // Outputs: 0=A, 1=B, 2=C, ...
    NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(L2); NET.addLayer(OUT);
  }

  // Dataset prepration

  // SAMPLES // 2D array [letter][sample] = data[]
  // RESULTS // 1D array [letter] = result[] expected

  // Prepare DATAS and TARGS as source and expected results to train

  //auto DATASE = [ getLArray(LA0), getLArray(LB0), getLArray(LC0), getLArray(LD0) ];
  //auto TARGSE = [ getR1Out(0,4),  getR1Out(1,4),  getR1Out(2,4),  getR1Out(3,4)  ];

  std::vector<std::vector<double>> DATASE; // data source etalon (no noise) : source samples as plain array (inputs)
  std::vector<std::vector<double>> TARGSE; // data target etalon (no noise) : results expected as plain array (outputs)

  for (size_t dataIndex = 0; dataIndex < SAMPLES.size(); dataIndex++)
  {
    for (size_t ii = 0; ii < SAMPLES[dataIndex].size(); ii++)
    {
      DATASE.push_back(SAMPLES[dataIndex][ii]);
      TARGSE.push_back(RESULTS[dataIndex]); // for all samples of same input result should be the same
    }
  }

  // Augmented data (original + shifted)

  std::vector<std::vector<double>> DATAS; // work samples (may be noised)
  std::vector<std::vector<double>> TARGS; // work targets (for in noised)

  for (size_t dataIndex = 0; dataIndex < DATASE.size(); dataIndex++)
  {
    DATAS.push_back(DATASE[dataIndex]);
    TARGS.push_back(TARGSE[dataIndex]);
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],0,1));
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],1,0));
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],1,1));
    TARGS.push_back(TARGSE[dataIndex]);
    TARGS.push_back(TARGSE[dataIndex]);
    TARGS.push_back(TARGSE[dataIndex]);
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],0,-1));
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],-1,0));
    DATAS.push_back(getShiftedImg(DATASE[dataIndex],-1,-1));
    TARGS.push_back(TARGSE[dataIndex]);
    TARGS.push_back(TARGSE[dataIndex]);
    TARGS.push_back(TARGSE[dataIndex]);
  }

  //auto DATA_AUGMENTATION_MUTIPLIER = DATAS.length / DATASE.length; // there will be 7 DATAS images per 1 source DATASE image

  // Dump dataset before train

  bool DUMP_DATASET = false;

  auto dumpSamples = [](const std::vector<std::vector<double>> &DATAS, int imagesPerSample)
  {
    int sampleCount = DATAS.size() / imagesPerSample; // sumber of samples

    for (auto sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
      auto T = std::vector<std::string>();
      for (auto imageIndex = 0; imageIndex < imagesPerSample; imageIndex++)
      {
        T = sampleAddLetTexts(DATAS[sampleIndex*imagesPerSample+imageIndex], T);
      }

      for (size_t lineIndex = 0; lineIndex < T.size(); lineIndex++)
      {
        console::log(T[lineIndex], lineIndex, sampleIndex);
      }
    }
  };

  if (DUMP_DATASET) { dumpSamples(DATAS, DATAS.size() / DATASE.size()); }

  // Training

  console::log("Training, please wait ...");
  if (!NN::doTrain(NET, DATAS, TARGS, -1, -1, &NN::TrainingProgressReporterConsole(10)))
  {
    console::log("Training failed!", NET.layers);
    return(false);
  }

  console::log("Training complete", NET.layers);

  // Verification

  auto verifyProc= [](NN::Network &NET, const std::vector<std::vector<double>> &DATAS, const std::vector<std::vector<double>> &TARGS, const char *stepName, int imagesPerSample) -> bool
  {
    bool DUMP_FAILED_IMAGES = false;

    auto CHKRS = std::vector<std::vector<double>>();
    for (size_t dataIndex = 0; dataIndex < DATAS.size(); dataIndex++)
    {
      CHKRS.push_back(NN::doProc(NET, DATAS[dataIndex]));
    }

    auto vdif = 0.15; // max diff for smart verification
    auto veps = 0.4; // epsilon for strict verification

    int statGood = 0;
    int statFail = 0;
    int statWarn = 0;

    auto isOK = true;
    for (size_t dataIndex = 0; dataIndex < DATAS.size(); dataIndex++)
    {
      auto imageIndex = dataIndex % imagesPerSample;
      auto sampleIndex = (dataIndex-imageIndex) / imagesPerSample;
      bool isSimpleMatchOK = NN::NetworkStat::isResultSampleMatchEps(TARGS[dataIndex], CHKRS[dataIndex], veps);

      std::string status = "";

      if (isSimpleMatchOK)
      {
        //status = "OK.OK.OK.OK.OK.OK.OK.OK"; // uncomment to dump all
        statGood++;
      }
      else
      {
        auto smartMatchSampleIndex = NN::NetworkStat::getMaximumIndexEps(CHKRS[dataIndex], vdif);
        if (smartMatchSampleIndex < 0)
        {
          status = "FAIL*";
          statFail++;
          isOK = false;
        }
        else
        {
          auto smartMatchExpectIndex = NN::NetworkStat::getMaximumIndex(TARGS[dataIndex]);
          if (smartMatchSampleIndex != smartMatchExpectIndex)
          {
            status = "FAIL";
            statFail++;
            isOK = false;
          }
          else // match, but not simple match
          {
            status = "WARN";
            statWarn++;
          }
        }
      }

      if (!status.empty()) // && (status != ""))
      {
        console::log("Verification step " + STR(stepName) + "[" + STR(dataIndex) + "]" + ":" + STR(status) + ""); // , [DATAS[dataIndex], TARGS[dataIndex], CHKRS[dataIndex]], smartMatchSampleIndex, [veps, vdif]);
        if (DUMP_FAILED_IMAGES)
        {
          auto T = sampleAddLetTexts(DATAS[dataIndex], std::vector<std::string>(), true, true, true, true);
          for (size_t lineIndex = 0; lineIndex < T.size(); lineIndex++)
          {
            console::log(T[lineIndex], lineIndex, sampleIndex, imageIndex);
          }
        }
      }
    }

    if (isOK)
    {
      console::log("Verification step " + STR(stepName) + ":OK [100%]");
    }
    else
    {
      auto statFull = 0.0 + statGood + statFail + statWarn;
      auto showPerc = [](double val) -> std::string { return STR("") + STR(round(val * 1000.0) / 10.0); };
      console::log("Verification step " + STR(stepName) + ":Done:" + (" GOOD=" + showPerc(statGood / statFull)) + (" WARN=" + showPerc(statWarn / statFull)) + (" FAIL=" + showPerc(statFail / statFull)));
    }

    return(isOK);
  };

  // Verify on source dataset (dry run)

  verifyProc(NET, DATAS, TARGS, "Source", DATAS.size() / DATASE.size());

  // Create noised data

  decltype(DATAS) DATASN;

  DATASN.clear();
  for (size_t dataIndex = 0; dataIndex < DATAS.size(); dataIndex++)
  {
    DATASN.push_back(getNoisedInput(DATAS[dataIndex],1));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.F1", DATASN.size() / DATASE.size());

  DATASN.clear();
  for (size_t dataIndex = 0; dataIndex < DATAS.size(); dataIndex++)
  {
    DATASN.push_back(getNoisedInput(DATAS[dataIndex],30,NOISE_TYPE_PIXEL_DARKER_LIGHTER));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.DL30", DATASN.size() / DATASE.size());

  DATASN.clear();
  for (size_t dataIndex = 0; dataIndex < DATAS.size(); dataIndex++)
  {
    DATASN.push_back(getNoisedInput(DATAS[dataIndex],10,NOISE_TYPE_PIXEL_RANDOM));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.R10", DATASN.size() / DATASE.size());
  return(true);
}

} } // NN::Demo
