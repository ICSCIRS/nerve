// nerve.cpp : This file contains the 'main' function. Program execution begins and ends there.
// I'm just writing this code. I have no an idea what shall be done as result :))

#include <iostream>

enum class Activation { SIGMOID, SOFTMAX, RELU };

template <typename T>
class Neuron {
public:
    unsigned numOfWeights;
    Activation transferFunction;

    Neuron() : numOfWeights(0), condition(0), delta(0), weights(nullptr), inputs(nullptr), transferFunction(Activation::SIGMOID) {}
    ~Neuron() {
        if (weights != nullptr) {
            delete[] weights;
        }
        if (inputs != nullptr) {
            delete[] inputs;
        }
    }

    void animate(unsigned numOfWeights, Activation transferFunction) {
        this->numOfWeights = numOfWeights;
        this->transferFunction = transferFunction;

        try {
            weights = new T[numOfWeights]();
        }
        catch (std::exception & ex) {
            std::cout << "Something is happened with allocating heap for the weights: " << ex.what() << std::endl;
            return;
        }

        try {
            inputs = new T * [numOfWeights]();
            for (int i = 0; i < numOfWeights; ++i) {
                inputs[i] = nullptr;
            }
        }
        catch (std::exception & ex) {
            std::cout << "Something is happened with allocating heap for the inputs: " << ex.what() << std::endl;
        }
    }

    void ForwardPropagation() {
        condition = 0;

        if (numOfWeights != 0) {

            for (unsigned i = 0; i < numOfWeights; ++i) {
                condition += weights[i] * *inputs[i];
            }

            switch (transferFunction) {
            case Activation::SIGMOID:
                condition = Sigmoid(condition);
                break;
            case Activation::RELU:
                condition = Relu(condition);
            case Activation::SOFTMAX:
                break;
            default:
                break;
            }
        }
    }

    const T Sigmoid(const T& x)
    {
        return static_cast<T>(1) / (1 + exp(-x));
    }

    template<typename T>
    const T Relu(const T& x)
    {
        return (x < 0) ? static_cast<T>(0) : x;
    }

private:
    T condition;
    T delta;
    T* weights;
    T** inputs;
};

template<typename T>
class NeuralClaster {
public:
    unsigned numOfNeurons;

    explicit NeuralClaster(unsigned numOfNeurons) : numOfNeurons(numOfNeurons), neurons(nullptr) {
        try {
            neurons = new Neuron<T>[numOfNeurons]();
        }
        catch (std::exception & ex) {
            std::cout << "A something wrong is happened with allocating heap for the neural claster: " << ex.what() << std::endl;
        }

    }
    ~NeuralClaster() {
        if (neurons != nullptr) {
            delete[] neurons;
        }
    }

private:
    Neuron<T>* neurons;
};

template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork(unsigned numOfInputs) : numOfInputs(numOfInputs) {}
    ~NeuralNetwork() {}

    class Domain {
    public:

    };

private:
    unsigned numOfInputs;
};


int main(int argc, char* argv[])
{
    Neuron<double> neurons; //Just will create a dead neuron;
    //NeuralClaster<double> nc1(9);

    std::cout << "Hello World!\n";

    return 0;
}
