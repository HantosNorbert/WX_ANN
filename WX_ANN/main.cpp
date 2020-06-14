#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include "wxMain.h"
#include "MNISTHandler.h"
#include "CustomEvents.h"

#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

IMPLEMENT_APP(MainApp) // Initializes the MainApp class and tells our program to run it
bool MainApp::OnInit()
{
    if (!wxApp::OnInit())
        return false;

    // Create an instance of our frame, or window
    MainFrame* mainFrame = new MainFrame(_T("Neural Network for MNIST"), wxPoint(0, 0), wxSize(500, 600));

    mainFrame->Show(TRUE);
    mainFrame->Centre(wxBOTH);
    SetTopWindow(mainFrame);
    return TRUE;
}

// Build up the event table
typedef void (wxEvtHandler::* PredictionEventFunction)(PredictionEvent&);
#define PredictionEventHandler(func) wxEVENT_HANDLER_CAST(PredictionEventFunction, func)  
wxDEFINE_EVENT(PREDICTION_EVENT, PredictionEvent);

typedef void (wxEvtHandler::* TestResultEventFunction)(TestResultEvent&);
#define TestResultEventHandler(func) wxEVENT_HANDLER_CAST(TestResultEventFunction, func)  
wxDEFINE_EVENT(TEST_RESULT_EVENT, TestResultEvent);

typedef void (wxEvtHandler::* LossEventFunction)(LossEvent&);
#define LossEventHandler(func) wxEVENT_HANDLER_CAST(LossEventFunction, func)  
wxDEFINE_EVENT(LOSS_EVENT, LossEvent);

BEGIN_EVENT_TABLE(MainFrame, wxFrame)
EVT_MENU(wxID_ABOUT, MainFrame::OnAbout)
EVT_BUTTON(LOAD_DB_BUTTON, MainFrame::onLoadDB)
EVT_BUTTON(START_TRAINING_BUTTON, MainFrame::onStartTraining)
EVT_BUTTON(STOP_TRAINING_BUTTON, MainFrame::onStopTraining)
EVT_BUTTON(PAUSE_TRAINING_BUTTON, MainFrame::onPauseTraining)
EVT_BUTTON(RESUME_TRAINING_BUTTON, MainFrame::onResumeTraining)
EVT_BUTTON(START_TESTING_BUTTON, MainFrame::onStartTesting)
EVT_COMMAND(MESSAGE_UPDATE, wxEVT_COMMAND_TEXT_UPDATED, MainFrame::onMessageUpdate)
END_EVENT_TABLE()

////////////////////////////////////////////////////////////////////////
// MainFrame
////////////////////////////////////////////////////////////////////////

MainFrame::MainFrame(const wxString& title, const wxPoint& pos, const wxSize
    & size)
    : wxFrame((wxFrame*)NULL, -1, title, pos, size)
    , m_lossPlotter(LOSS_IMAGE_WIDTH, LOSS_IMAGE_HEIGHT)
{
    wxMenu* menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);
    wxMenuBar* menuBar = new wxMenuBar;
    menuBar->Append(menuHelp, "&Help");
    SetMenuBar(menuBar);

    m_trainingSampleSizeText = new wxStaticText(this, TRAINING_SAMPLE_SIZE_TEXT, _T("MNIST sample size:"),
        wxPoint(5, 5), wxDefaultSize, 0);

    m_trainingSampleSizeTextCtrl = new wxTextCtrl(this, TRAINING_SAMPLE_SIZE_TEXT_CTRL, "5000",
        wxPoint(5, 25), wxSize(120, 20), wxTE_LEFT, wxDefaultValidator, wxTextCtrlNameStr);

    m_loadDBButton = new wxButton(this, LOAD_DB_BUTTON, _T("Load MNIST"),
        wxPoint(5, 55), wxSize(120, 25), 0);

    m_startTrainingButton = new wxButton(this, START_TRAINING_BUTTON, _T("Start network"),
        wxPoint(5, 115), wxSize(120, 25), 0);

    m_stopTrainingButton = new wxButton(this, STOP_TRAINING_BUTTON, _T("Stop network"),
        wxPoint(5, 145), wxSize(120, 25), 0);

    m_pauseTrainingButton = new wxButton(this, PAUSE_TRAINING_BUTTON, _T("Pause network"),
        wxPoint(5, 185), wxSize(120, 25), 0);

    m_resumeTrainingButton = new wxButton(this, RESUME_TRAINING_BUTTON, _T("Resume network"),
        wxPoint(5, 215), wxSize(120, 25), 0);

    m_startTestingButton = new wxButton(this, START_TESTING_BUTTON, _T("Start testing"),
        wxPoint(305, 215), wxSize(120, 25), 0);

    m_predictionText = new wxStaticText(this, PREDICTION_TEXT, _T("PRED.: NONE"),
        wxPoint(255, 155), wxDefaultSize, 0);

    m_maxLossText = new wxStaticText(this, MAX_LOSS_TEXT, _T("MAX: NONE"),
        wxPoint(365, 15), wxDefaultSize, 0);

    m_currentLossText = new wxStaticText(this, CURRENT_LOSS_TEXT, _T("LOSS: NONE"),
        wxPoint(365, 85), wxDefaultSize, 0);

    ConfusionMatrix emptyMatrix = ConfusionMatrix(MNIST_LABEL_NUM, MNIST_LABEL_NUM);
    m_confusionMatrixText = new wxStaticText(this, CONFUSION_MATRIX_TEXT,
        "Confusion matrix (last accuracy: 0%)\n" + emptyMatrix.toStringWithHeaders(),
        wxPoint(5, 265), wxDefaultSize, 0);
    wxFont font;
    font.SetPointSize(12);
    font.SetFamily(wxFONTFAMILY_TELETYPE);
    m_confusionMatrixText->SetFont(font);

    // Load a default, gray image
    m_trainImageRGBData.resize(MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT * 3);
    for (int i = 0; i < MNIST_IMAGE_WIDTH; ++i)
    {
        for (int j = 0; j < MNIST_IMAGE_HEIGHT; ++j)
        {
            m_trainImageRGBData[(i * MNIST_IMAGE_WIDTH + j) * 3 + 0] = 128;
            m_trainImageRGBData[(i * MNIST_IMAGE_WIDTH + j) * 3 + 1] = 128;
            m_trainImageRGBData[(i * MNIST_IMAGE_WIDTH + j) * 3 + 2] = 128;
        }
    }
    m_trainImage = wxImage(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, &m_trainImageRGBData[0], true);
    m_trainImageBitmap = wxBitmap(m_trainImage, 24);
    m_trainImageStaticBitmap = new wxStaticBitmap(this, wxID_ANY, m_trainImageBitmap, wxPoint(205, 145),
        wxSize(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT));


    m_lossImageRGBData = m_lossPlotter.getRawRGBData();
    m_lossImage = wxImage(m_lossPlotter.getW(), m_lossPlotter.getH(), &m_lossImageRGBData[0], true);
    m_lossImageBitmap = wxBitmap(m_lossImage, 24);
    m_lossImageStaticBitmap = new wxStaticBitmap(this, wxID_ANY, m_lossImageBitmap, wxPoint(155, 5),
        wxSize(m_lossPlotter.getW(), m_lossPlotter.getH()));

    Bind(PREDICTION_EVENT, &MainFrame::onPrediction, this, PREDICTION);
    Bind(TEST_RESULT_EVENT, &MainFrame::onTestResult, this, TEST_RESULT);
    Bind(LOSS_EVENT, &MainFrame::onLossUpdate, this, LOSS_UPDATE);

    CreateStatusBar();
    SetStatusText("Ready.");
}

void MainFrame::OnAbout(wxCommandEvent& event)
{
    wxString msg;
    msg << "A simple application for training a fully connected feedforward neural network\n";
    msg << "on the MNIST database.\n\n";
    msg << "Created by Norbert Hantos\n2020. 06. 14.";
    wxMessageBox(msg, "About this application", wxOK | wxICON_INFORMATION);
}

// If a prediction event comes from the training thread, refresh the training sample image and its prediction
void MainFrame::onPrediction(PredictionEvent& event)
{
    size_t imageIdx = event.getTrainingImageIdx();
    size_t prediction = event.getPrediction();

    auto const trainImage = m_MNISTHandler.getTrainingImages().at(imageIdx);

    m_predictionText->SetLabel(wxString::Format(wxT("PRED.: %i"), prediction));

    for (size_t i = 0; i < trainImage.size(); ++i)
    {
        m_trainImageRGBData[3 * i + 0] = trainImage[i];
        m_trainImageRGBData[3 * i + 1] = trainImage[i];
        m_trainImageRGBData[3 * i + 2] = trainImage[i];
    }
    m_trainImageBitmap = wxBitmap(m_trainImage, 24);
    m_trainImageStaticBitmap->SetBitmap(m_trainImageBitmap);
    m_trainImageStaticBitmap->Refresh();
}

// If a loss event comes from the training thread, update the loss plotter and the loss text
void MainFrame::onLossUpdate(LossEvent& event)
{
    m_lossPlotter.addLoss(event.getLoss());

    m_currentLossText->SetLabel(wxString::Format(wxT("LOSS: %f"), event.getLoss()));

    // draw loss graph
    m_lossImageRGBData = m_lossPlotter.getRawRGBData();
    m_lossImageBitmap = wxBitmap(m_lossImage, 24);
    m_lossImageStaticBitmap->SetBitmap(m_lossImageBitmap);
    m_lossImageStaticBitmap->Refresh();

    m_maxLossText->SetLabel(wxString::Format(wxT("MAX: %f"), m_lossPlotter.getMaxLoss()));
}

// If a test result comes from the training thread, update the confusion matrix
// and the training error percentage
void MainFrame::onTestResult(TestResultEvent& event)
{
    wxString message;
    message << "Confusion matrix (last accuracy: " << event.getConfusionMatrix().getTestResultPercentage() << "%)\n";
    message << event.getConfusionMatrix().toStringWithHeaders();
    m_confusionMatrixText->SetLabel(message);
}

MainFrame::~MainFrame()
{
    {
        wxCriticalSectionLocker locker(wxGetApp().m_critsect);
        // check if we have any threads running first
        if (wxGetApp().m_trainingThread)
        {
            if (wxGetApp().m_trainingThread->IsPaused())
            {
                wxGetApp().m_trainingThread->Resume();
                // set the flag indicating that all threads should exit
            }
            wxGetApp().m_shuttingDown = true;
        }
    }
    // now wait for them to really terminate
    if (wxGetApp().m_trainingThread)
        wxGetApp().m_semAllDone.Wait();
}

void MainFrame::onLoadDB(wxCommandEvent& event)
{
    if (m_MNISTHandler.getTrainingImages().size() > 0 || m_MNISTHandler.getTestingImages().size() > 0)
    {
        wxLogError(wxT("MNIST database is already loaded into memory."));
        return;
    }

    unsigned long sampleSize;
    wxString sampleSizeStr = m_trainingSampleSizeTextCtrl->GetLineText(0);
    if (!sampleSizeStr.ToULong(&sampleSize))
    {
        wxLogError(wxT("Cannot convert the sample size to integer."));
        return;
    }

    if (sampleSize < 1000)
    {
        wxLogError(wxT("Please set the sample size to at least 1000."));
        return;
    }

    wxDirDialog dirDialog(this, "Select the MNIST root folder", "");
    if (dirDialog.ShowModal() != wxID_OK)
    {
        wxLogError(wxT("Cannot load MNIST database."));
        return;
    }

    m_loadDBButton->Disable();
    m_MNISTHandler.setPath(dirDialog.GetPath());
    MNISTHandler::State trainingState = m_MNISTHandler.loadTrainingDB(this->GetStatusBar(), sampleSize);
    MNISTHandler::State testingState = m_MNISTHandler.loadTestingDB(this->GetStatusBar(), sampleSize);
    m_loadDBButton->Enable();

    switch (trainingState)
    {
    case MNISTHandler::State::OK:
        break;
    case MNISTHandler::State::FILE_OPEN_ERROR:
        wxLogError(wxT("Cannot open MNIST training files."));
        break;
    case MNISTHandler::State::MAGIC_NUM_ERROR:
        wxLogError(wxT("Magic number is incorrect in MNIST training files."));
        break;
    case MNISTHandler::State::MISMATCHING_IMAGE_AND_LABEL_SIZE:
        wxLogError(wxT("The number of training images and labels are different."));
        break;
    default:
        wxLogError(wxT("Unknown error in training files."));
    }

    switch (testingState)
    {
    case MNISTHandler::State::OK:
        break;
    case MNISTHandler::State::FILE_OPEN_ERROR:
        wxLogError(wxT("Cannot open MNIST testing files."));
        break;
    case MNISTHandler::State::MAGIC_NUM_ERROR:
        wxLogError(wxT("Magic number is incorrect in MNIST testing files."));
        break;
    case MNISTHandler::State::MISMATCHING_IMAGE_AND_LABEL_SIZE:
        wxLogError(wxT("The number of testing images and labels are different."));
        break;
    default:
        wxLogError(wxT("Unknown error in testing files."));
    }

    if (m_MNISTHandler.getTrainingImages().empty() || m_MNISTHandler.getTestingImages().empty())
    {
        wxLogError(wxT("MNIST database is not properly loaded."));
        return;
    }

    SetStatusText("MNIST database loaded. Ready for training.");
}

// The training thread wants to poll from the message queue (did the user requested a testing?)
wxMessageQueueError MainFrame::receiveTimeout(TestSignal &msg)
{
    return m_queue.ReceiveTimeout(WX_QUEUE_TIMEOUT_MS, msg);
}

// To help the training thread updating the status bar
void MainFrame::onMessageUpdate(wxCommandEvent& evt)
{
    SetStatusText(evt.GetString());
}

void MainFrame::onStartTraining(wxCommandEvent& WXUNUSED(event))
{
    if (m_MNISTHandler.getTrainingImages().empty() || m_MNISTHandler.getTestingImages().empty())
    {
        wxLogError(wxT("MNIST database is not loaded yet."));
        return;
    }

    wxCriticalSectionLocker enter(wxGetApp().m_critsect);
    if (wxGetApp().m_trainingThread)
    {
        wxLogError(wxT("A training thread already exists."));
        return;
    }

    m_lossPlotter.clear();
    TrainingThread* thread = _createTrainingThread();

    if (thread->Run() != wxTHREAD_NO_ERROR)
    {
        wxLogError(wxT("Can't start train thread."));
    }
}

void MainFrame::onStartTesting(wxCommandEvent& WXUNUSED(event))
{
    m_queue.Post(TestSignal::DO_TEST);
}

TrainingThread* MainFrame::_createTrainingThread()
{
    TrainingThread* thread = new TrainingThread(this);

    if (thread->Create() != wxTHREAD_NO_ERROR)
    {
        wxLogError(wxT("Can't create train thread."));
    }

    wxCriticalSectionLocker enter(wxGetApp().m_critsect);
    wxGetApp().m_trainingThread = thread;

    return thread;
}

void MainFrame::onStopTraining(wxCommandEvent& WXUNUSED(event))
{
    wxThread* toDelete = NULL;
    {
        wxCriticalSectionLocker enter(wxGetApp().m_critsect);
        // stop the last thread
        if (!wxGetApp().m_trainingThread)
        {
            wxLogError(wxT("No train thread to stop."));
        }
        else
        {
            toDelete = wxGetApp().m_trainingThread;
        }
    }

    if (toDelete)
    {
        // This can still crash if the thread gets to delete itself
        // in the mean time.
        toDelete->Delete();
    }
}

void MainFrame::onPauseTraining(wxCommandEvent& WXUNUSED(event))
{
    wxCriticalSectionLocker enter(wxGetApp().m_critsect);
    if (!wxGetApp().m_trainingThread || !wxGetApp().m_trainingThread->IsRunning())
    {
        wxLogError(wxT("No train thread to pause."));
    }
    else
    {
        wxGetApp().m_trainingThread->Pause();
    }
}

void MainFrame::onResumeTraining(wxCommandEvent& WXUNUSED(event))
{
    wxCriticalSectionLocker enter(wxGetApp().m_critsect);
    if (!wxGetApp().m_trainingThread || !wxGetApp().m_trainingThread->IsPaused())
    {
        wxLogError(wxT("No train thread to resume."));
    }
    else
    {
        wxGetApp().m_trainingThread->Resume();
    }
}

////////////////////////////////////////////////////////////////////////
// TrainingThread
////////////////////////////////////////////////////////////////////////

const std::vector<size_t> TrainingThread::NEURAL_NET_TOPOLOGY = {
    MainFrame::MNIST_IMAGE_WIDTH * MainFrame::MNIST_IMAGE_WIDTH,
    200, 80, MainFrame::MNIST_LABEL_NUM
};

TrainingThread::TrainingThread(MainFrame* mainFrame)
    : wxThread()
    , m_neuralNet(Net(NEURAL_NET_TOPOLOGY))
{
    m_mainFrame = mainFrame;
}

TrainingThread::~TrainingThread()
{
    wxCriticalSectionLocker locker(wxGetApp().m_critsect);
    wxGetApp().m_trainingThread = NULL;
    // signal the main thread that there are no more threads left if it is
    // waiting for us
    if (wxGetApp().m_shuttingDown)
    {
        wxGetApp().m_shuttingDown = false;
        wxGetApp().m_semAllDone.Post();
    }
}

wxThread::ExitCode TrainingThread::Entry()
{
    std::vector<std::vector<uint8_t>> const& trainingImages = m_mainFrame->getTrainingImages();
    std::vector<std::vector<uint8_t>> const& testingImages = m_mainFrame->getTestingImages();
    std::vector<int8_t> const& trainingLabels = m_mainFrame->getTrainingLabels();
    std::vector<int8_t> const& testingLabels = m_mainFrame->getTestingLabels();

    // We have to convert the stored data into what the neural networks wants
    std::vector<double> normalizedImage;
    std::vector<double> predictedLabelVec;
    int predictedLabel;
    std::vector<double> trueLabelVec;

    // For random testing shuffling
    std::random_device rd;
    std::mt19937 g(rd());

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    for (size_t idx = 0; idx < trainingImages.size(); ++idx)
    {
        // Ask the main frame if the user requested a testing
        MainFrame::TestSignal testSignal = MainFrame::TestSignal::IDLE;
        bool doTest = false;
        // Do testing if user requested...
        if (m_mainFrame->receiveTimeout(testSignal) != wxMSGQUEUE_TIMEOUT)
            if (testSignal == MainFrame::TestSignal::DO_TEST)
                doTest = true;
        // ...Or if the last training is reached
        if (idx == trainingImages.size() - 1)
            doTest = true;

        if (doTest)
        {
            ConfusionMatrix confMatrix(MainFrame::MNIST_LABEL_NUM, MainFrame::MNIST_LABEL_NUM);
            
            // Select some test indices randomly
            std::vector<size_t> testIndices(testingImages.size());
            std::iota(testIndices.begin(), testIndices.end(), 0);
            std::shuffle(testIndices.begin(), testIndices.end(), g);

            // Do the actual testing cycle
            for (size_t testIdx = 0; testIdx < MainFrame::TESTING_IMAGE_SAMPLE_SIZE; ++testIdx)
            {
                // Check if the application is shutting down: in this case all threads should stop ASAP
                {
                    wxCriticalSectionLocker locker(wxGetApp().m_critsect);
                    if (wxGetApp().m_shuttingDown)
                        return NULL;
                }

                // Check if just this thread was asked to exit
                if (TestDestroy())
                    return NULL;

                // Feedforward the data
                _normalizeImage(normalizedImage, testingImages[testIndices[testIdx]]);
                m_neuralNet.feedForward(normalizedImage);
                m_neuralNet.getResults(predictedLabelVec);
                _vecToLabel(predictedLabel, predictedLabelVec);
                confMatrix.add(testingLabels[testIndices[testIdx]], predictedLabel);

                {
                    // Tell main frame where are we in the testing
                    wxCommandEvent messageEvent(wxEVT_COMMAND_TEXT_UPDATED, MESSAGE_UPDATE);
                    wxString message;
                    message << "Testing: " << testIdx << " / " << MainFrame::TESTING_IMAGE_SAMPLE_SIZE;
                    messageEvent.SetString(message);
                    m_mainFrame->GetEventHandler()->AddPendingEvent(messageEvent);
                }
            }
            // Tell the main frame the result of the test
            TestResultEvent testResultEvent(TEST_RESULT_EVENT, TEST_RESULT);
            testResultEvent.setConfusionMatrix(confMatrix);
            m_mainFrame->GetEventHandler()->AddPendingEvent(testResultEvent);
        }

        // Check if the application is shutting down: in this case all threads should stop ASAP
        {
            wxCriticalSectionLocker locker(wxGetApp().m_critsect);
            if (wxGetApp().m_shuttingDown)
                return NULL;
        }

        // Check if just this thread was asked to exit
        if (TestDestroy())
            return NULL;

        // Feedforward a training example
        _normalizeImage(normalizedImage, trainingImages[idx]);
        m_neuralNet.feedForward(normalizedImage);
        m_neuralNet.getResults(predictedLabelVec);
        _vecToLabel(predictedLabel, predictedLabelVec);

        // Send the loss to main frame
        LossEvent lossEvent(LOSS_EVENT, LOSS_UPDATE);
        lossEvent.setLoss(m_neuralNet.getRecentAverageError());
        m_mainFrame->GetEventHandler()->AddPendingEvent(lossEvent);

        // Back propagation
        _labelToVec(trueLabelVec, trainingLabels[idx]);
        m_neuralNet.backProp(trueLabelVec);

        // If enough time elapsed, send the mainframe the current training image's index and its prediction
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
        if (elapsedTime > MainFrame::PREDICTION_ELAPSE_TIME_MS)
        {
            PredictionEvent predictionEvent(PREDICTION_EVENT, PREDICTION);
            predictionEvent.setTrainingImageIdx(idx);
            predictionEvent.setPrediction(predictedLabel);
            m_mainFrame->GetEventHandler()->AddPendingEvent(predictionEvent);

            startTime = currentTime;
        }

        {
            // Tell main frame where are we in the training
            wxCommandEvent messageEvent(wxEVT_COMMAND_TEXT_UPDATED, MESSAGE_UPDATE);
            wxString message;
            message << "Training: " << idx << " / " << trainingImages.size();
            messageEvent.SetString(message);  // pass some data along the event, a number in this case
            m_mainFrame->GetEventHandler()->AddPendingEvent(messageEvent);
        }
    }

    {
        // Tell main frame where are done
        wxCommandEvent messageEvent(wxEVT_COMMAND_TEXT_UPDATED, MESSAGE_UPDATE);
        messageEvent.SetString("Done.");
        m_mainFrame->GetEventHandler()->AddPendingEvent(messageEvent);
    }
    return NULL;
}

void TrainingThread::_normalizeImage(std::vector<double>& normImage, std::vector<uint8_t> const& image)
{
    normImage = std::vector<double>(image.begin(), image.end());
    std::transform(normImage.begin(), normImage.end(), normImage.begin(),
        [](double v) -> double { return v / 255.0; }
    );
}

void TrainingThread::_labelToVec(std::vector<double>& labelVec, int label)
{
    labelVec.clear();
    labelVec.resize(10, 0.0);
    labelVec.at(label) = 1.0;
}

void TrainingThread::_vecToLabel(int& label, std::vector<double> const& labelVec)
{
    label = std::distance(labelVec.begin(), std::max_element(labelVec.begin(), labelVec.end()));
}
