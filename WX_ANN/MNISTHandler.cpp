#include "MNISTHandler.h"
#include <fstream>

uint32_t MNISTHandler::_swapEndian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

MNISTHandler::State MNISTHandler::loadTrainingDB(wxStatusBar* statusBar, unsigned long sampleSize)
{
	m_trainingImages.clear();
	m_trainingLabels.clear();
	return _loadDB(m_dbPath + "train-images.idx3-ubyte", m_dbPath + "train-labels.idx1-ubyte", statusBar, true, sampleSize);
}

MNISTHandler::State MNISTHandler::loadTestingDB(wxStatusBar* statusBar, unsigned long sampleSize)
{
	m_testingImages.clear();
	m_testingLabels.clear();
	return _loadDB(m_dbPath + "t10k-images.idx3-ubyte", m_dbPath + "t10k-labels.idx1-ubyte", statusBar, false, sampleSize);
}

MNISTHandler::State MNISTHandler::_loadDB(std::string const& imageFilename, std::string const& labelFilename,
	wxStatusBar* statusBar, bool isTraining, unsigned long sampleSize)
{
	// Open files
	std::basic_ifstream<uint8_t> imageFileStream(imageFilename, std::ios::in | std::ios::binary);
	std::ifstream labelFileStream(labelFilename, std::ios::in | std::ios::binary);

	if (!imageFileStream.is_open() || !labelFileStream.is_open())
	{
		return MNISTHandler::State::FILE_OPEN_ERROR;
	}

	// Read the magic and the meta data
	uint32_t magic;
	uint32_t numItems;
	uint32_t numLabels;
	uint32_t rows;
	uint32_t cols;

	// Magic numbers as they are documented on MNIST's site
	imageFileStream.read(reinterpret_cast<uint8_t*>(&magic), 4);
	magic = _swapEndian(magic);
	if (magic != 2051)
	{
		return MNISTHandler::State::MAGIC_NUM_ERROR;
	}

	labelFileStream.read(reinterpret_cast<char*>(&magic), 4);
	magic = _swapEndian(magic);
	if (magic != 2049)
	{
		return MNISTHandler::State::MAGIC_NUM_ERROR;
	}

	imageFileStream.read(reinterpret_cast<uint8_t*>(&numItems), 4);
	numItems = _swapEndian(numItems);
	labelFileStream.read(reinterpret_cast<char*>(&numLabels), 4);
	numLabels = _swapEndian(numLabels);
	if (numItems != numLabels)
	{
		return MNISTHandler::State::MISMATCHING_IMAGE_AND_LABEL_SIZE;
	}

	imageFileStream.read(reinterpret_cast<uint8_t*>(&rows), 4);
	rows = _swapEndian(rows);
	imageFileStream.read(reinterpret_cast<uint8_t*>(&cols), 4);
	cols = _swapEndian(cols);

	char label;
	uint8_t* pixels = new uint8_t[rows * cols];

	// Read at most sampleSize number of images / labels
	if (sampleSize < numItems)
	{
		numItems = sampleSize;
	}

	for (size_t item_id = 0; item_id < numItems; ++item_id)
	{

		if (isTraining)
			statusBar->SetLabel(wxString::Format(wxT("Loading training data... (%i / %i)"),
				item_id, numItems));
		else
			statusBar->SetLabel(wxString::Format(wxT("Loading testing data... (%i / %i)"),
				item_id, numItems));

		// Read image pixel
		imageFileStream.read(pixels, rows * cols);
		// Read label
		labelFileStream.read(&label, 1);

		std::vector<uint8_t> imageVec(pixels, pixels + rows * cols * sizeof(uint8_t));

		if (isTraining)
		{
			m_trainingImages.push_back(imageVec);
			m_trainingLabels.push_back(int8_t(label));
		}
		else
		{
			m_testingImages.push_back(imageVec);
			m_testingLabels.push_back(int8_t(label));
		}
	}

	imageFileStream.close();
	labelFileStream.close();
	delete[] pixels;

	return MNISTHandler::State::OK;
}
