// This project uses some code inspired by "NeuralNetworkFromScratch" (https://github.com/Bot-Academy/NeuralNetworkFromScratch) by Bot Academy.
#include <cctype>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <omp.h>

using namespace std;

// Run this command to compile: g++ -Ofast -march=native -ffast-math -fopenmp -shared -o libclassifier.dll mnist_classifier.cpp
extern "C" {

    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const float learn_rate = 0.001;
    const int epochs = 10;

    vector<vector<float>> input_hidden;
    vector<vector<float>> hidden_output;
    vector<vector<float>> hidden_bias;
    vector<vector<float>> output_bias;

    // Training data
    vector<vector<float>> images;
    vector<vector<int>> labels;

    // Random number generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.5, 0.5);

    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    float relu(float x) {
        return max(0.0f, x);
    }

    // Matrix multiplication
    vector<vector<float>> matmul(
        const vector<vector<float>>& A, 
        const vector<vector<float>>& B) {
        
        vector<vector<float>> result(
            A.size(), vector<float>(B[0].size(), 0.0f));

        #pragma omp parallel for collapse(3)
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < B[0].size(); j++) {
                for (size_t k = 0; k < B.size(); k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // Matrix addition
    vector<vector<float>> matrix_add(
        const vector<vector<float>>& A, 
        const vector<vector<float>>& B) {
        
        vector<vector<float>> result = A;
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < A[0].size(); j++) {
                result[i][j] += B[i][j];
            }
        }
        return result;
    }

    // Transpose matrix
    vector<vector<float>> transpose(
        const vector<vector<float>>& matrix) {
        
        vector<vector<float>> result(
            matrix[0].size(), vector<float>(matrix.size()));

        for (size_t i = 0; i < matrix.size(); i++) {
            for (size_t j = 0; j < matrix[0].size(); j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    // Load MNIST data
    void load_data() {
        ifstream imagesFile("images.bin", ios::binary);
        ifstream labelsFile("labels.bin", ios::binary);

        float pixel;
        int label;
        while (true) {
            vector<float> image;
            for (int i = 0; i < 784; ++i) {
                if (!imagesFile.read(reinterpret_cast<char*>(&pixel), sizeof(float))) {
                    break;
                }
                image.push_back(pixel);
            }

            if (image.size() < 784) break;
            images.push_back(image);

            if (!labelsFile.read(reinterpret_cast<char*>(&label), sizeof(int))) {
                break;
            }

            vector<int> encoded_label(10, 0);
            encoded_label[label] = 1;
            labels.push_back(encoded_label);
        }

        imagesFile.close();
        labelsFile.close();
    }

    // Initialize weights and biases
    void init_weights() {
        input_hidden.resize(hidden_size, vector<float>(input_size));
        hidden_output.resize(output_size, vector<float>(hidden_size));
        
        // Random weight initialization
        for (auto& row : input_hidden) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
        
        for (auto& row : hidden_output) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }

        // Initialize biases to zero
        hidden_bias.resize(hidden_size, vector<float>(1, 0.0f));
        output_bias.resize(output_size, vector<float>(1, 0.0f));

        // Load training data
        load_data();
    }

    void train() {
        init_weights();
        cout << "Training..." << endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            int num_correct = 0;

            #pragma omp parallel for reduction(+:num_correct)
            for (size_t i = 0; i < images.size(); ++i) {
                // Prepare image as column vector
                vector<vector<float>> image(input_size, vector<float>(1));
                for (int j = 0; j < input_size; ++j) {
                    image[j][0] = images[i][j];
                }

                // Prepare label as column vector
                vector<vector<float>> label(output_size, vector<float>(1));
                for (int j = 0; j < output_size; ++j) {
                    label[j][0] = labels[i][j];
                }

                // Forward propagation input -> hidden
                auto hidden_pre = matrix_add(hidden_bias, matmul(input_hidden, image));
                vector<vector<float>> hidden(hidden_pre.size(), vector<float>(hidden_pre[0].size()));
                #pragma omp parallel for collapse(2)
                for (size_t r = 0; r < hidden_pre.size(); ++r) {
                    for (size_t c = 0; c < hidden_pre[0].size(); ++c) {
                        hidden[r][c] = sigmoid(hidden_pre[r][c]);
                    }
                }

                // Forward propagation hidden -> output
                auto output_pre = matrix_add(output_bias, matmul(hidden_output, hidden));
                vector<vector<float>> output(output_pre.size(), vector<float>(output_pre[0].size()));
                #pragma omp parallel for collapse(2)
                for (size_t r = 0; r < output_pre.size(); ++r) {
                    for (size_t c = 0; c < output_pre[0].size(); ++c) {
                        output[r][c] = sigmoid(output_pre[r][c]);
                    }
                }
                
                // Check prediction accuracy
                int pred_label = 0;
                int true_label = 0;
                float max_pred = -1;
                float max_true = -1;
                for (int j = 0; j < output_size; ++j) {
                    if (output[j][0] > max_pred) {
                        max_pred = output[j][0];
                        pred_label = j;
                    }
                    if (label[j][0] > max_true) {
                        max_true = label[j][0];
                        true_label = j;
                    }
                }
                
                if (pred_label == true_label) {
                    num_correct++;
                }

                // Backpropagation output -> hidden
                vector<vector<float>> delta_o(output_size, vector<float>(1));
                for (int j = 0; j < output_size; ++j) {
                    delta_o[j][0] = output[j][0] - label[j][0];
                }

                // Update hidden to output weights
                auto delta_h_o = matmul(delta_o, transpose(hidden));
                for (size_t r = 0; r < hidden_output.size(); ++r) {
                    for (size_t c = 0; c < hidden_output[0].size(); ++c) {
                        hidden_output[r][c] -= learn_rate * delta_h_o[r][c];
                    }
                }

                // Update hidden to output bias
                for (size_t r = 0; r < output_bias.size(); ++r) {
                    output_bias[r][0] -= learn_rate * delta_o[r][0];
                }

                // Backpropagation hidden -> input
                vector<vector<float>> delta_h(hidden_size, vector<float>(1));
                auto t_hidden_output = transpose(hidden_output);
                
                for (size_t r = 0; r < hidden_size; ++r) {
                    float sum = 0.0f;
                    for (size_t c = 0; c < output_size; ++c) {
                        sum += t_hidden_output[r][c] * delta_o[c][0];
                    }
                    delta_h[r][0] = sum * hidden[r][0] * (1 - hidden[r][0]);
                }

                // Update input to hidden weights
                auto delta_i_h = matmul(delta_h, transpose(image));
                for (size_t r = 0; r < input_hidden.size(); ++r) {
                    for (size_t c = 0; c < input_hidden[0].size(); ++c) {
                        input_hidden[r][c] -= learn_rate * delta_i_h[r][c];
                    }
                }

                // Update input to hidden bias
                for (size_t r = 0; r < hidden_bias.size(); ++r) {
                    hidden_bias[r][0] -= learn_rate * delta_h[r][0];
                }
            }

            // Print epoch accuracy
            float accuracy = (num_correct / static_cast<float>(images.size())) * 100.0f;
            cout << "Epoch " << epoch + 1 << " Accuracy: " 
                    << fixed << setprecision(2) << accuracy << "%" << endl;
        }
    }

    float* classify(float* flat_image) {
        // Prepare image as column vector
        vector<vector<float>> image(input_size, vector<float>(1));
        for (int j = 0; j < input_size; ++j) {
            image[j][0] = flat_image[j];
        }

        // Forward propagation input -> hidden
        auto hidden_pre = matrix_add(hidden_bias, matmul(input_hidden, image));
        vector<vector<float>> hidden(hidden_pre.size(), vector<float>(hidden_pre[0].size()));
        #pragma omp parallel for collapse(2)
        for (size_t r = 0; r < hidden_pre.size(); ++r) {
            for (size_t c = 0; c < hidden_pre[0].size(); ++c) {
                hidden[r][c] = sigmoid(hidden_pre[r][c]);
            }
        }

        // Forward propagation hidden -> output
        auto output_pre = matrix_add(output_bias, matmul(hidden_output, hidden));
        vector<vector<float>> output(output_pre.size(), vector<float>(output_pre[0].size()));
        #pragma omp parallel for collapse(2)
        for (size_t r = 0; r < output_pre.size(); ++r) {
            for (size_t c = 0; c < output_pre[0].size(); ++c) {
                output[r][c] = sigmoid(output_pre[r][c]);
            }
        }

        // Flatten output matrix to 1D vector
        float* flat_output = new float[output_size];
        for (int i = 0; i < output_size; ++i) {
            flat_output[i] = output[i][0];
        }
        
        return flat_output;
    }
}