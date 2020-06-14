# The WX-ANN-MNIST Project

This is a simple c++ application to train an artificial neural network on the MNIST database. The training is managable by a GUI, using wxWidgets.

## MNIST

MNIST is database containing handwritten digits, each of them labelled manually so it can be used for supervised learning. Each digit is normalized to fit into a 28x28 pixel bounding box, where the pixels are antialiased, and between 0 (background) and 255 (foreround) in value. The database contains 60,000 samples for training and 10,000 samples for testing.
The MNIST database can be [downloaded from here](http://yann.lecun.com/exdb/mnist/).

## Artificial Neural Network

In this project a simple, feedforward, fully connected neural network can be trained on the MNIST database to recognise handwritten digits. Neural networks are great tools of machine learning, made up of artificial neurons structured into layers. See Wikipedia's page on [Feedforward networks](https://en.wikipedia.org/wiki/Feedforward_neural_network) for more.

## wxWidgets

wxWidgets is a free and open source widget toolkit and tools library for creating graphical user interfaces for cross-platform applications. Learn more on the [official site]([https://www.wxwidgets.org/](https://www.wxwidgets.org/)).

## Features of the Application

The application has the following features:

- A text box to select the amount of data to be loaded from MNIST. This can be as low as 1000 for a quick test, or 60,000 for reading the entire database. Keep in mind that loading many data might take a while.
- A button to load the selected amount of data from the MNIST database. The button opens up a window where the user can select the root folder of the MNIST files; that is, where the folder containg the following files: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`.
- A button to start a new neural network training (if the database is loaded and there is no running or paused training). The training cycle runs through the training data once. If it reaches the end, a final test cycle will automatically runs.
- A button to stop the neural network (if it's running). Once the neural network stopped, starting it again will start an entire new training.
- A button to pause the neural network (if it's running).
- A button to resume the nerual network (if it's paused).
- A button to start a testing cycle of the neural network. In this case, 1000 samples from the testing database will be randomly selected, and the application calculates the accuracy, as well as creates a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). During testing the training is halted. Once the testing cycle is finished, the training resumes.
- Displays the current (smoothened) loss, as well as visualizes the past 200 losses.
- Displays the last test result. This includes the accuracy of 1000 random test samples, and the confusion matrix.
- Every second, the application displays what the actual training image is, and what the network predicted for it.

![Screenshot of theapplication](https://github.com/HantosNorbert/WX_ANN/blob/master/screenshot.png?raw=true)

*Screenshot of the running application. The first 5000 samples were selected from MNIST, for training, and the neural network already trained on 1139 images. The current (smoothed) loss is 0.177090. The last test resulted a 70% accuracy. Currently, the network got an image of the digit 3, which it correctly predicted to be 3.*

## Implementation Details

The application is written in c++ language, in Visual Studio 2019, on Microsoft Windows 10.

### wxWidgets

For setting up wxWidgets, I used the [following tutorial](https://www.youtube.com/watch?v=sRhoZcNpMb4). The overall look of the GUI is hard-coded, using wxButtons, wxStaticTexts, and wxTextCtrl. For displaying images, wxStaticBitmaps are used. Some actions refreshes the StatusBar, sending meaningful messages to the user (such as how many images the network already used for training, or what is the progress on the testing cycle). The application notifies the user with an error window if an error occurs, for example, stopping a non-existent training, or the application couldn't find the MNIST files in the given folder.

### Neural Network

For implementing the neural network, I followed [David Miller's tutorial](https://vimeo.com/19569529). In the final implementation, the neural network uses 28x28 neurons on the input layer, 10 neurons on the output layers, and two hidden layers with 200 neurons and 80 neurons, respectively. The neural networrk also contains bias neurons. The weights are initialized uniform randomly between -1 and +1. The learning rate is 0.01, and the momentum is 0.9. The activation function in each layer is sigmoid, and the error function is the root mean square error (RMSE). The input images are normalized to have values between 0 and 1 (although this happens out of the neural net's scope). Each training sample's error is backpropagated immediately (no mini-batches or stochastic gradient descent). The loss, which is displayed on the GUI, is smoothed across the previous losses.
This network can achieve 80% precision after about 4000 training samples.

### Threads and Inter-Thread Communication

The thread handling was written by following the wxWidget's sample code on threads, as well as the wiki's [inter-thread and inter-process communication](https://wiki.wxwidgets.org/Inter-Thread_and_Inter-Process_communication) tutorial. For sending signal from the main thread toward the neural network's thread (e.g., 'start a test cycle'), the thread-safe `wxMessageQueue` is used, which the network's thread is constantly monitoring. For sending data from the network's thread to the main thread (e.g., 'show the user the test result'), the network's thread assigns custom events for the main thread, with the custom event containg the required data.

### Displaying the Loss

Instead of using a 3rd party library, I implemented a naive, extremely simple way of plotting the loss. The loss is an image created pixel by pixel, as follows: the image contains a bar chart of the last 200 losses. Each bar column is 1 pixel wide, and the height is adjusted: the largest (so-far) loss has a height that matches the height of the loss image exactly (100 pixels), and every other loss is scaled accordingly.

## Possible Improvements

This project is a simple GUI application for training a neural network on MNIST, and such as, can be improved in a lot of ways. A few ideas, including but not limited to:

- Scramble the training data. Currently the application reads the MNIST training samples in order (although for the testing, random indices are selected). 
- Display the elapsed time.
- Many parameters are defined as static consts in various classes. These should either be
adjustable parameters on the GUI, or derived from source (e.g., the size of MNIST images).
- The MNIST database loading should be on a separate thread. Currently it is slow, and the GUI is not responsive during the loading process. Also, the database is loaded into the stack, which might cause issues if there is not enough memory.
- The current parameters of the neural network are selected arbitrary. An optimization might be nice.
- Plotting the loss can be nicer, as well as faster. Use a 3rd party library.
- Training and testing is slow; the code is not optimized for speed. Also, implementing the network on GPU would mean a significant boost.
- Write unit tests.
- Upgrade the neural network: use convolutional network instead (much better for image related machine learning tasks), use softmax in the last layer with cross entropy, use mini-batches and stochastic gradient descent, dropout, early-stop, etc.
- Upgrade the application to train on other datasets, not just MNIST.
- Option for saving and loading the weights of the network.

## Known Issues

- If the user stops the network, testing is not possible. This is because the entire thread is stopped, and the thread is responsible for the testing too. Recommended fix: testing should have its own thread.
- Although during the MNIST database loading the load button is disabled, it still stores the click event which the GUI wants to handle after the loading process ended. Recommended fix: different type of event handling for buttons.
- Similarly, the test button is spammable - clicking it on multiple times will result multiple testing cycles, one after the other. Recommended fix: ignore test requests during testing.
- The MNIST database can be loaded once. Once it is loaded, even if the user wants a different sample size, the application does not allow another loading. Recommended fix: allow re-loading if the sample size changes.
- The GUI is not responsive during the MNIST database loading. This is because it happens on the same thread. Recommended fix: the loading should have its own thread.

Norbert Hantos  
2020.06.14.  
Budapest
