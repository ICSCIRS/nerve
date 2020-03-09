// Nerve - a simple library for building the deep neural networks. 
// I'm just writing this code. I have no an idea what shall be done as result :))
// Best regards from https://github.com/Runsolar/easydnn
// 09.03.2020

#include <iostream>

enum class Activation { SIGMOID, SOFTMAX, RELU };

//Just a mere neuron
template <typename T>
class Neuron {
public:
    unsigned numOfWeights;
    Activation transferFunction;

    // This default constructor just make a dead neuron xD
    Neuron() : numOfWeights(0), condition(0), delta(0), weights(nullptr), inputs(nullptr), transferFunction(Activation::SIGMOID) {}
    ~Neuron() {
        if (weights != nullptr) {
            delete[] weights;
        }
        if (inputs != nullptr) {
            delete[] inputs;
        }
    }

    // This method can enliven a dead neuron ;)
    void animate(unsigned numOfWeights, Activation transferFunction) {
        this->numOfWeights = numOfWeights;
        this->transferFunction = transferFunction;

        try {
            weights = new T[numOfWeights]();
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something is happened with allocating heap for the weights: " << ex.what() << std::endl;
            return;
        }

        try {
            inputs = new T * [numOfWeights]();
            for (int i = 0; i < numOfWeights; ++i) {
                inputs[i] = nullptr;
            }
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something is happened with allocating heap for the inputs: " << ex.what() << std::endl;
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


// In conventional sense it's calling a layer, but I like to name it a neural cluster.
template<typename T>
class NeuralClaster {
public:
    unsigned numOfNeurons;

    explicit NeuralCluster(unsigned numOfNeurons) : numOfNeurons(numOfNeurons), neurons(nullptr) {
        try {
            neurons = new Neuron<T>[numOfNeurons]();
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something wrong is happened with allocating heap for the neural cluster: " << ex.what() << std::endl;
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
        Domain* pNextDomain;
        Domain* pPreviousDomain;
    };

private:
    unsigned numOfInputs;
};


int main(int argc, char* argv[])
{
    Neuron<double> neurons; //Just will create a dead neuron;
    //NeuralCluster<double> nc1(9);

    std::cout << "Hello World!\n";

    return 0;
}
