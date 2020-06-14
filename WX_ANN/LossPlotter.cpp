#include "LossPlotter.h"

LossPlotter::LossPlotter(size_t imW, size_t imH)
	: m_imW(imW)
	, m_imH(imH)
	, m_maxLoss(0.0)
{
	m_rawRGBData.resize(imW * imH * 3, 0);
}

void LossPlotter::addLoss(double loss)
{
	if (loss > m_maxLoss)
		m_maxLoss = loss;

	m_losses.push_back(loss);

	if (m_losses.size() > m_imW)
		m_losses.pop_front();
}

std::vector<unsigned char> const &LossPlotter::getRawRGBData()
{
	_refreshRawRGBData();
	return m_rawRGBData;
}

void LossPlotter::clear()
{
	m_maxLoss = 0.0;
	m_losses.clear();
}

void LossPlotter::_refreshRawRGBData()
{
	// rescale the losses so that the overall maximum loss's height matches the image's height
	double scaleFactor = m_maxLoss == 0.0 ? 1.0 : m_imH / m_maxLoss;
	
	for (size_t x = 0; x < m_imW; ++x)
	{
		for (size_t y = 0; y < m_imH; ++y)
		{
			size_t pixelCoordR = 3 * (y * m_imW + x) + 0;
			size_t pixelCoordG = 3 * (y * m_imW + x) + 1;
			size_t pixelCoordB = 3 * (y * m_imW + x) + 2;

			// draw a bar at position x which has the height of m_losses[x]
			if (x < m_losses.size() and scaleFactor * m_losses[x] > m_imH - y)
			{
				m_rawRGBData[pixelCoordR] = 0;
				m_rawRGBData[pixelCoordG] = 0;
				m_rawRGBData[pixelCoordB] = 255;
			}
			// background
			else
			{
				m_rawRGBData[pixelCoordR] = 180;
				m_rawRGBData[pixelCoordG] = 180;
				m_rawRGBData[pixelCoordB] = 255;
			}
		}
	}
}
