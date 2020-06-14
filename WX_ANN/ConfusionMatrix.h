#pragma once

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include <vector>

/**
* The confusion matrix (error matrix) to sum up the different type of errors.
* The number at position [i][j] represents how many times the network have a test
* case where the ground truth label was i, and the network predicted label j.
*/
class ConfusionMatrix
{
public:
	ConfusionMatrix();
	ConfusionMatrix(size_t m, size_t n);
	wxString toStringWithHeaders() const; // Represent the matrix in a formatted way
	void add(size_t i, size_t j) { m_matrix.at(i).at(j) += 1; }
	int16_t getTestResultPercentage() const; // The ratio of the diagonal sum (correct answers) and
	// the sum of all elements

private:
	size_t m_m;
	size_t m_n;
	std::vector<std::vector<size_t>> m_matrix;
};
