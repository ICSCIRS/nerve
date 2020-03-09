// Nerve - a simple library for building the deep neural networks. 
// I'm just writing this code. I have no an idea what shall be done as result :))
// Best regards from https://github.com/Runsolar/easydnn
// 09.03.2020

#include <iostream>

enum class Activation { SIGMOID, SOFTMAX, RELU };
template <typename T> struct DataSet;
template<typename T> class Neuron;
template<typename T> class NeuralClaster;
template<typename T> class NeuralNetwork;

//Just a mere neuron
template <typename T>
class Neuron {
public:
    unsigned numOfWeights;
    Activation transferFunction;

    // This default constructor just make a dead neuron xD
    Neuron() : numOfWeights(0), condition(0), delta(0), weights(nullptr), inputs(nullptr), transferFunction(Activation::SIGMOID) {};
    ~Neuron();

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
            for (unsigned i = 0; i < numOfWeights; ++i) {
                //inputs[i] = nullptr;
            }
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something is happened with allocating heap for the inputs: " << ex.what() << std::endl;
            return;
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
Neuron<T>::~Neuron() {
    if (weights != nullptr) {
        delete[] weights;
        weights = nullptr;
    }
    if (inputs != nullptr) {
        delete[] inputs;
        inputs = nullptr;
    }
}



// In conventional sense it's calling a layer, but I like to name it a neural cluster.
template<typename T>
class NeuralCluster {
    friend NeuralNetwork<T>;

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
    ~NeuralCluster() {
        if (neurons != nullptr) {
            delete[] neurons;
            neurons = nullptr;
        }
    }

private:
    Neuron<T>* neurons;
};



// Neural network
template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork(unsigned numOfInputs) : numOfInputs(numOfInputs), head(nullptr), tail(nullptr) {}
    ~NeuralNetwork() {  
        if (head != nullptr) {
            while (head != nullptr) {
                Domain<T>* current = head;
                head = current->pNextDomain;
                delete current;
                std::cout << "A Domain has been deleted... " << this << std::endl;
            }
            head = tail = nullptr;
            std::cout << "The NeuralNetwork has been deleted... " << this << std::endl;
        }
    }

    void pushCluster(NeuralCluster<T>& neuralcluster);

private:
    template<typename T>
    class Domain {
    public:
        NeuralCluster<T>& neuralcluster;
        Domain<T>* pNextDomain;
        Domain<T>* pPreviousDomain;
        
        Domain(NeuralCluster<T>& neuralcluster = NeuralCluster<T>(), Domain<T>* pPreviousDomain = nullptr, Domain<T>* pNextDomain = nullptr):
            neuralcluster(neuralcluster),
            pNextDomain(pNextDomain),
            pPreviousDomain(pPreviousDomain) {}
    };

    unsigned numOfInputs;
    unsigned numOfNeuronsOfCurentCluster = 0;
    Domain<T>* head;
    Domain<T>* tail;
};

template<typename T>
void NeuralNetwork<T>::pushCluster(NeuralCluster<T>& neuralcluster) {
    if (head == nullptr) {
        head = new Domain<T>(neuralcluster);
        for (unsigned i = 0; i < (head->neuralcluster).numOfNeurons; ++i) {
            (head->neuralcluster).neurons[i].animate(numOfInputs, Activation::SIGMOID);
        }
        numOfNeuronsOfCurentCluster = (head->neuralcluster).numOfNeurons;
    }
    else {
        Domain<T>* current = head;
        while (current->pNextDomain != nullptr) {
            current = current->pNextDomain;
        }
        current->pNextDomain = new Domain<T>(neuralcluster, current);
        
        for (unsigned i = 0; i < (current->pNextDomain->neuralcluster).numOfNeurons; ++i) {
            (current->pNextDomain->neuralcluster).neurons[i].animate(numOfNeuronsOfCurentCluster, Activation::SIGMOID);
        }
        numOfNeuronsOfCurentCluster = (current->pNextDomain->neuralcluster).numOfNeurons;
    }
}

int main(int argc, char* argv[])
{
    //Neuron<double> neurons; //Just will create a dead neuron.

    NeuralCluster<double> nc1(3);
    NeuralCluster<double> nc2(9);
    NeuralCluster<double> nc3(9);
    NeuralCluster<double> nc4(1);

    NeuralNetwork<double> nn(3);

    nn.pushCluster(nc1);
    nn.pushCluster(nc2);
    nn.pushCluster(nc3);
    nn.pushCluster(nc4);

    std::cout << "Hello World!\n";

    return 0;


}
