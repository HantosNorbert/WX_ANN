#pragma once

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include "ConfusionMatrix.h"

class PredictionEvent;
wxDECLARE_EVENT(PREDICTION_EVENT, PredictionEvent);

// An event that holds the index of the current training image and its prediction label
class PredictionEvent : public wxCommandEvent
{
public:
    PredictionEvent(wxEventType commandType, int id)
        : wxCommandEvent(commandType, id)
        , m_trainingImageIdx(0)
        , m_prediction(-1)
    { }

    // *Must* copy here the data to be transported
    PredictionEvent(const PredictionEvent& event)
        : wxCommandEvent(event)
    {
        this->setTrainingImageIdx(event.getTrainingImageIdx());
        this->setPrediction(event.getPrediction());
    }

    // Required for sending
    wxEvent* Clone() const { return new PredictionEvent(*this); }

    size_t getTrainingImageIdx() const { return m_trainingImageIdx; }
    size_t getPrediction() const { return m_prediction; }
    void setTrainingImageIdx(size_t idx) { m_trainingImageIdx = idx; }
    void setPrediction(size_t pred) { m_prediction = pred; }

private:
    size_t m_trainingImageIdx;
    size_t m_prediction;
};

class TestResultEvent;
wxDECLARE_EVENT(TEST_RESULT_EVENT, TestResultEvent);

// An event that holds the result of a test (a confusion matrix)
class TestResultEvent : public wxCommandEvent
{
public:
    TestResultEvent(wxEventType commandType, int id)
        : wxCommandEvent(commandType, id)
    { }

    // *Must* copy here the data to be transported
    TestResultEvent(const TestResultEvent& event)
        : wxCommandEvent(event)
    {
        this->setConfusionMatrix(event.getConfusionMatrix());
    }

    // Required for sending
    wxEvent* Clone() const { return new TestResultEvent(*this); }

    ConfusionMatrix getConfusionMatrix() const { return m_confusionMatrix; }
    void setConfusionMatrix(const ConfusionMatrix& m) { m_confusionMatrix = m; }

private:
    ConfusionMatrix m_confusionMatrix;
};

class LossEvent;
wxDECLARE_EVENT(LOSS_EVENT, LossEvent);

// An event that holds the loss value
class LossEvent : public wxCommandEvent
{
public:
    LossEvent(wxEventType commandType, int id)
        : wxCommandEvent(commandType, id)
        , m_loss(-1.0)
    { }

    // *Must* copy here the data to be transported
    LossEvent(const LossEvent& event)
        : wxCommandEvent(event)
    {
        this->setLoss(event.getLoss());
    }

    // Required for sending
    wxEvent* Clone() const { return new LossEvent(*this); }

    double getLoss() const { return m_loss; }
    void setLoss(double l) { m_loss = l; }

private:
    double m_loss;
};
