#pragma once
#include<vector>
#include<deque>

/**
* A very naive way to plot some error bars.
* An error bar is one pixel wide, and only the last imageWidth number or error bars
* are shown. The height of the error bars are adjusted: the (so far) biggest error
* has exactly imageHeight height in pixels, and all the errors are adjusted accordingly.
*/
class LossPlotter
{
public:
	LossPlotter(size_t imW, size_t imH);
	void addLoss(double loss);
	std::vector<unsigned char> const &getRawRGBData();

	double getMaxLoss() const { return m_maxLoss; }
	size_t getW() const { return m_imW; }
	size_t getH() const { return m_imH; }
	void clear();
private:
	void _refreshRawRGBData();

	size_t m_imW;
	size_t m_imH;
	double m_maxLoss;
	std::deque<double> m_losses;
	std::vector<unsigned char> m_rawRGBData;
};
