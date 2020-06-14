#include "ConfusionMatrix.h"

ConfusionMatrix::ConfusionMatrix()
	: ConfusionMatrix(10, 10)
{ }

ConfusionMatrix::ConfusionMatrix(size_t m, size_t n)
	: m_m(m)
	, m_n(n)
{
	for (size_t i = 0; i < m; ++i)
	{
		std::vector<size_t> row(n, 0);
		m_matrix.push_back(row);
	}
}

int16_t ConfusionMatrix::getTestResultPercentage() const
{
	size_t correct = 0;
	size_t sum = 0;
	// Calculate the ratio of the diagonal sum (correct answers) and the sum of all elements
	for (size_t i = 0; i < m_m; ++i)
	{
		for (size_t j = 0; j < m_n; ++j)
		{
			if (i == j)
				correct += m_matrix[i][j];
			sum += m_matrix[i][j];
		}
	}
	if (sum == 0)
		return 0;
	return int16_t(round(100.0 * correct / sum));
}

wxString ConfusionMatrix::toStringWithHeaders() const
{
	wxString result;

	// Create column header
	result << "  ";
	for (size_t j = 0; j < m_n; ++j)
	{
		wxString colHeader = wxString::Format(wxT("%i"), j);
		result << colHeader.Pad(4 - colHeader.size(), ' ', false);
	}
	result << "\n";

	// Create horizontal line
	wxString line = "  ";
	line.Pad(4 * m_n, '-', true);
	result << line << "\n";

	// Fill the table (with row header and vertical line)
	for (size_t i = 0; i < m_m; ++i)
	{
		wxString rowHeader = wxString::Format(wxT("%i|"), i);
		result << rowHeader;
		for (size_t j = 0; j < m_n; ++j)
		{
			wxString num = wxString::Format(wxT("%i"), m_matrix[i][j]);
			result << num.Pad(4 - num.size(), ' ', false);
		}
		result << "\n";
	}
	return result;
}
