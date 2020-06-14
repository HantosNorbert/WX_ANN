#pragma once
#include <vector>
#include "Neuron.h"

/**
* The Artificial Neural Network: a fully connected, dense feedforward network.
* Uses sigmoid activation, biases, and RME error backpropagation.
*/
class Net
{
public:
	Net(std::vector<size_t> const& topology);
	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	static double m_recentAverageSmoothingFactor; // Number of training samples to average over

	std::vector<Layer> m_layers; //As m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
};
