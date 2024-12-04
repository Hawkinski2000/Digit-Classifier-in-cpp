# Digit-Classifier-in-cpp
A neural network implemented in C++ for classifying handwritten digits. Digits can be drawn on the canvas and predictions are displayed on the graph in real time.

![2](https://github.com/user-attachments/assets/3986637e-5296-47a4-978e-4457a70ce227)

![3](https://github.com/user-attachments/assets/f06c85c6-02a5-4d3e-bcc2-75664a6c8827)

![6](https://github.com/user-attachments/assets/7e204c9e-696a-4f92-b640-7103c23f7dd4)

## Compiling

To run this project, you will need to compile the .dll file with the following command so the Python GUI can interact with the C++ file:

Windows: `g++ -Ofast -march=native -ffast-math -fopenmp -shared -o libclassifier.dll mnist_classifier.cpp`

Linux/macOS: `g++ -Ofast -march=native -ffast-math -fopenmp -shared -o libclassifier.so mnist_classifier.cpp`
