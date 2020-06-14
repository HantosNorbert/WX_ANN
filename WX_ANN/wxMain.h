#pragma once
#include "MNISTHandler.h"
#include "CustomEvents.h"
#include "LossPlotter.h"
#include "Net.h"

#include <wx/msgqueue.h>

/**
* MainApp is the class for our application, thecontainer of the main window.
*/
class MainApp : public wxApp
{
public:
    virtual bool OnInit();

    // For thread handling
    wxCriticalSection m_critsect;
    wxThread* m_trainingThread = NULL;
    wxSemaphore m_semAllDone;
    bool m_shuttingDown = false;
};

class TrainingThread;
class TestingThread;

/**
* The main window class. Responsible for displaying buttons, images, and event handling.
*/
class MainFrame : public wxFrame
{
public:
    static const size_t MNIST_IMAGE_WIDTH = 28;
    static const size_t MNIST_IMAGE_HEIGHT = 28;
    static const size_t MNIST_LABEL_NUM = 10;
    static const size_t TESTING_IMAGE_SAMPLE_SIZE = 1000;
    static const size_t LOSS_IMAGE_WIDTH = 200;
    static const size_t LOSS_IMAGE_HEIGHT = 100;
    static const long WX_QUEUE_TIMEOUT_MS = 10;
    static const size_t PREDICTION_ELAPSE_TIME_MS = 1000;
public:
    enum class TestSignal { IDLE, DO_TEST };
public:
    MainFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
    ~MainFrame();

    void OnAbout(wxCommandEvent& event);

    // The callback functions of the buttons
    void onLoadDB(wxCommandEvent& event);
    void onStartTraining(wxCommandEvent& event);
    void onStopTraining(wxCommandEvent& event);
    void onPauseTraining(wxCommandEvent& event);
    void onResumeTraining(wxCommandEvent& event);
    void onStartTesting(wxCommandEvent& event);

    // The callback funcions for handling events coming from the training thread
    void onPrediction(PredictionEvent& event);
    void onLossUpdate(LossEvent& event);
    void onTestResult(TestResultEvent& event);
    void onMessageUpdate(wxCommandEvent& event);

    // Thread-save queue for sending message to the thread (for user-required testing)
    wxMessageQueueError receiveTimeout(TestSignal &msg);

    std::vector<std::vector<uint8_t>> const& getTrainingImages() { return m_MNISTHandler.getTrainingImages(); }
    std::vector<std::vector<uint8_t>> const& getTestingImages() { return m_MNISTHandler.getTestingImages(); }
    std::vector<int8_t> const& getTrainingLabels() { return m_MNISTHandler.getTrainingLabels(); }
    std::vector<int8_t> const& getTestingLabels() { return m_MNISTHandler.getTestingLabels(); }

private:
    TrainingThread* _createTrainingThread();

    wxStaticText* m_trainingSampleSizeText;
    wxTextCtrl* m_trainingSampleSizeTextCtrl;

    wxButton* m_loadDBButton;

    wxButton* m_startTrainingButton;
    wxButton* m_stopTrainingButton;
    wxButton* m_pauseTrainingButton;
    wxButton* m_resumeTrainingButton;

    wxButton* m_startTestingButton;

    wxStaticText* m_predictionText;
    wxStaticText* m_maxLossText;
    wxStaticText* m_currentLossText;
    wxStaticText* m_confusionMatrixText;

    std::vector<unsigned char> m_trainImageRGBData;
    wxBitmap m_trainImageBitmap;
    wxImage m_trainImage;
    wxStaticBitmap* m_trainImageStaticBitmap;

    LossPlotter m_lossPlotter;
    std::vector<unsigned char> m_lossImageRGBData;
    wxBitmap m_lossImageBitmap;
    wxImage m_lossImage;
    wxStaticBitmap* m_lossImageStaticBitmap;

    MNISTHandler m_MNISTHandler;
    wxMessageQueue<TestSignal> m_queue;

    DECLARE_EVENT_TABLE()
};

enum
{
    LOAD_DB_BUTTON = wxID_HIGHEST + 1, // declares an id which will be used to call our button
    START_TRAINING_BUTTON,
    STOP_TRAINING_BUTTON,
    PAUSE_TRAINING_BUTTON,
    RESUME_TRAINING_BUTTON,
    START_TESTING_BUTTON,
    MENU_ABOUT,
    TRAINING_SAMPLE_SIZE_TEXT,
    TRAINING_SAMPLE_SIZE_TEXT_CTRL,
    MAX_LOSS_TEXT,
    PREDICTION_TEXT,
    CURRENT_LOSS_TEXT,
    CONFUSION_MATRIX_TEXT,
    LOSS_UPDATE, // event
    PREDICTION, // event
    TEST_RESULT, // event
    MESSAGE_UPDATE
};

///////////////////////////////////////////////////////////////////////////

class TrainingThread : public wxThread
{
public:
    static const std::vector<size_t> NEURAL_NET_TOPOLOGY;
public:
    TrainingThread(MainFrame* mainFrame);
    virtual ~TrainingThread();
    virtual void* Entry();
private:
    // Helper functions for converting data to and from the neural network
    static void _normalizeImage(std::vector<double> &normImage, std::vector<uint8_t> const& image);
    static void _labelToVec(std::vector<double> &labelVec, int label);
    static void _vecToLabel(int &label, std::vector<double> const &labelVec);

    MainFrame* m_mainFrame;
    Net m_neuralNet;
};
