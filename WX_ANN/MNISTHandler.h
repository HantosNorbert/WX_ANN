#pragma once

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include<vector>
#include<string>

/**
* Class for loading and holding MNIST data: training images/labels and testing images/labels.
*/
class MNISTHandler
{
public:
	enum class State { OK, FILE_OPEN_ERROR, MAGIC_NUM_ERROR, MISMATCHING_IMAGE_AND_LABEL_SIZE };
public:
	void setPath(wxString dbPath) { m_dbPath = std::string(dbPath.mb_str()) + "\\"; }
	State loadTrainingDB(wxStatusBar* statusBar, unsigned long sampleSize);
	State loadTestingDB(wxStatusBar* statusBar, unsigned long sampleSize);

	std::vector<std::vector<uint8_t>> const& getTrainingImages() { return m_trainingImages; }
	std::vector<std::vector<uint8_t>> const& getTestingImages() { return m_testingImages; }
	std::vector<int8_t> const& getTrainingLabels() { return m_trainingLabels; }
	std::vector<int8_t> const& getTestingLabels() { return m_testingLabels; }
private:
	uint32_t _swapEndian(uint32_t val);
	State _loadDB(std::string const &imageFilename, std::string const &labelFilename,
		wxStatusBar* statusBar, bool isTraining, unsigned long sampleSize);

	std::string m_dbPath;
	std::vector<std::vector<uint8_t>> m_trainingImages;
	std::vector<std::vector<uint8_t>> m_testingImages;
	std::vector<int8_t> m_trainingLabels;
	std::vector<int8_t> m_testingLabels;
};
