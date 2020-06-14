#pragma once
#include <vector>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

/**
* A class representing a single neuron in our Neural Network.
* Apart from basic calculations, contains the weight of all the output connections.
*/
class Neuron
{
public:
	Neuron(size_t numOutputs, size_t myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);
private:
	static double eta; // Overall net training rate
	static double alpha; // Multiplier of last weight change (a.k.a. momentum)

	static double transferFunction(double x); // A sigmoid
	static double transferFunctionDerivative(double x); // The derivative of the sigmoid
	
	static double randomWeight(void) { return 2 * (rand() / double(RAND_MAX)) - 1; }

	double sumDOW(Layer const& nextLayer) const;

	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	size_t m_myIndex;
	double m_gradient;
};
