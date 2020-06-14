#include "Net.h"
#include <cassert>
#include <iostream>

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(std::vector<double>& resultVals) const
{
	resultVals.clear();
	for (size_t n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double>& targetVals)
{
	// Calculate overal net error (Root Mean Square Error of output neuron errors)
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;
	// bias excluded
	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // Get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement
	if (m_recentAverageError < 0.0)
		m_recentAverageError = m_error;
	else
		m_recentAverageError =
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	// Calculate gradients on hidden layers
	for (size_t layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for (size_t n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (size_t layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (size_t n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const std::vector<double>& inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for (size_t i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate: all the non-bias neurons, feeded from the previous later
	for (size_t layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer& prevLayer = m_layers[layerNum - 1];
		for (size_t n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
Net::Net(const std::vector<size_t>& topology)
	: m_error(0.0)
	, m_recentAverageError(-1.0)
{
	size_t numLayers = topology.size();
	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer
		for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}
