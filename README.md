# NNCPP
Simple Self-Contained Neural Network toolkit written on C++ (usefull as sample or tutorial)

## Example

### Create network 
```
// Create NN.Layers
auto IN  = new NN::Layer(28*28, NN::TheNeuronFactory<NN::InputNeuron>()); IN->addNeurons(1,NN::TheNeuronFactory<NN::BiasNeuron>());
auto L1  = new NN::Layer(28*28, NN::TheNeuronFactory<NN::ProcNeuronTrainee>()); L1->addNeurons(1,NN::TheNeuronFactory<NN::BiasNeuron>());
auto OUT = new NN::Layer(10, NN::TheNeuronFactory<NN::ProcNeuronTrainee>()); // Outputs: 0="0", 1="1", 2="2", ...
// Connect layers
L1->addInputAll(IN);
OUT->addInputAll(L1);
// Create NN.Network by NN.Layers
NN::Network NET;
NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);
```

### Train
```
// NN::doTrain(NET, DATAS, TARGS) // run training session
//std::vector<std::vector<double>> DATAS = [ [...], ... ]; // source data (each sample holds values for inputs)
//std::vector<std::vector<double>> TARGS = [ [...], ... ]; // expected results (each result holds expected outputs)
cout << "Training, please wait ...";
if (!NN::doTrain(NET, DATAS, TARGS))
{
  cout << "Training failed!";
}
cout << "Training complete";
```

### Run inference
```
// NN::doProc(NET, DATA) // run single infrence calculation
// std::vector<double> DATA Input (must have same count as number of inputs)
// Returns result std::vector<double> (will have same count as number of outputs)
auto CALC = NN::doProc(NET, DATA);
```

## Cousin JavaScript project: NNJS
There is a NN engine implemenation on JavaScript with (almost) the same API:

https://github.com/openlab-vn-ua/NNJS

## Useful links (and thanks to!)
* Steven Miller, "Mind: How to Build a Neural Network (Part One)"<br/>
http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
* Matt Mazur, "A Step by Step Backpropagation Example"<br/>
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
* Arnis71, "Introduction to neural networks" (on Russian)<br/>
https://habrahabr.ru/post/312450/ <br/>
https://habrahabr.ru/post/313216/ <br/>
https://habrahabr.ru/post/307004/
