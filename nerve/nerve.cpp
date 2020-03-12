// Nerve - a simple library for building the deep neural networks. 
// I'm just writing this code. I have no an idea what shall be done as result :))
// Best regards from https://github.com/Runsolar/easydnn
// 09.03.2020

#include <iostream>

enum class Activation { SIGMOID, SOFTMAX, RELU };
template <typename T> struct DataSet;
template<typename T> class Neuron;
template<typename T> class NeuralCluster;
template<typename T> class NeuralNetwork;


template<typename T>
class Array {
public:
    int len;
    explicit Array(const int& len) : len(len), data(nullptr) {
        try {
            data = new T[len]();
            //std::cout << "A new Vector hase been created... " << this << std::endl;
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }
    }
    ~Array() {
        delete[] data;
    }

protected:
    T* data;
};

template<typename T>
class Vector : public Array<T> {
    friend NeuralNetwork<T>;
public:
    int len;

    Vector() = delete;
    //Vector() : len(0), data(nullptr) {}
    explicit Vector(const int& len) : Array<T>::Array(len), len(len) {
        try {
            pdata = new T * [len]();
        }
        catch (std::exception & ex) {
            std::cout << "En exception is happened... " << ex.what() << std::endl;
            return;
        }

        for (int i = 0; i < len; ++i) {
            pdata[i] = &(Array<T>::data[i]);
        }
    }

    Vector(const Vector<T>& vec) : Vector(vec.len) {
        if (this == &vec) return;
        for (int i = 0; i < len; ++i) {
            *pdata[i] = *vec.pdata[i];
        }
    }

    ~Vector() {
        delete[] pdata;
        //std::cout << "A Vector hase been deleted... " << this << std::endl;
    }

    T& operator[](const int index) {
        return *pdata[index];
    }

    const T& operator[](const int index) const {
        return *pdata[index];
    }

protected:
    T** pdata;
};



template<typename T>
class Matrix {

public:
    int cols;
    int rows;

    Matrix() = delete;
    //Matrix(): rows(0), cols(0), matrix(nullptr) {}

    explicit Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
        matrix = new Vector<T> * [cols];
        for (int i = 0; i < cols; ++i) {
            matrix[i] = new Vector<T>(rows);
        }
        //std::cout << "A new matrix has been created... " << this << std::endl;
    }

    Vector<T>& operator[](const int index) const {
        return *matrix[index];
    }

    /*
    const Matrix<T>& operator=(const Matrix<T>& matrixObj) const {
        if (cols == matrixObj.cols && rows == matrixObj.rows) {
            for (int i = 0, j; i < cols; ++i) {
                T& col = *matrix[i];
                for (j = 0; j < rows; j++) {
                    col[j] = matrixObj[i][j];
                }
            }
        }
        return *this;
    }
*/

    ~Matrix() {
        for (int i = 0; i < cols; ++i) {
            delete matrix[i];
        }
        delete[] matrix;
        //std::cout << "A matrix has been deleted... " << this << std::endl;
    }

protected:

    Vector<T>** matrix;
};


//Just a mere neuron
template <typename T>
class Neuron {
    friend NeuralNetwork<T>;

public:
    int numOfWeights;
    Activation transferFunction;

    // This default constructor just make a dead neuron xD
    Neuron() : numOfWeights(0), condition(0), delta(0), weights(nullptr), inputs(nullptr), transferFunction(Activation::SIGMOID) {};
    ~Neuron();

    // This method can enliven a dead neuron ;)
    void animate(int numOfWeights, Activation transferFunction) {
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
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something is happened with allocating heap for the inputs: " << ex.what() << std::endl;
            return;
        }

        for (int i = 0; i < numOfWeights; ++i) {
            weights[i] = static_cast<int>(rand() % 2) ? static_cast<T>(rand()) / RAND_MAX : static_cast<T>(rand()) / -RAND_MAX;
            inputs[i] = nullptr;
        }
    }


    void forwardPropagation() {
        condition = 0;

        if (numOfWeights != 0) {

            for (int i = 0; i < numOfWeights; ++i) {
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
    const int numOfNeurons;

    explicit NeuralCluster(const int numOfNeurons) : numOfNeurons(numOfNeurons), neurons(nullptr) {
        try {
            neurons = new Neuron<T>[numOfNeurons]();
        }
        catch (std::exception & ex) {
            std::cout << "Ups O_O! Something wrong is happened with allocating heap for the neural cluster: " << ex.what() << std::endl;
        }

    }

    void forwardProp() {
        for (int i = 0; i < numOfNeurons; ++i) {
            neurons[i].forwardPropagation();
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
    mutable T m_error = 0;

    NeuralNetwork(const int numOfInputs) : numOfInputs(numOfInputs), input(numOfInputs), head(nullptr), tail(nullptr) {}
    ~NeuralNetwork() {  
        if (head != nullptr) {
            while (head != nullptr) {
                Domain* current = head;
                head = current->pNextDomain;
                delete current;
                std::cout << "A Domain has been deleted... " << this << std::endl;
            }
            head = tail = nullptr;
            std::cout << "The NeuralNetwork has been deleted... " << this << std::endl;
        }
    }

    void pushCluster(NeuralCluster<T>& neuralcluster);
    void mountDataSet(const DataSet<T>& dataset);
    void trainNerve(const int &epochs) const;
    void feedForward(const Vector<T>& input) const;

private:

    class Domain {
    public:
        NeuralCluster<T>& neuralcluster;
        Domain* pNextDomain;
        Domain* pPreviousDomain;
        
        Domain(NeuralCluster<T>& neuralcluster = NeuralCluster<T>(), Domain* pPreviousDomain = nullptr,  Domain* pNextDomain = nullptr):
            neuralcluster(neuralcluster),
            pNextDomain(pNextDomain),
            pPreviousDomain(pPreviousDomain) {}
    };

    const int numOfInputs;
    int numOfNeuronsOfPreviousCluster = 0;
    Domain* head;
    Domain* tail;
    const DataSet<T>* pDataSet = nullptr;
    Vector<T> input;
};

template<typename T>
void NeuralNetwork<T>::pushCluster(NeuralCluster<T>& neuralcluster) {
    if (head == nullptr) {
        head = new Domain(neuralcluster);
        for (int i = 0; i < (head->neuralcluster).numOfNeurons; ++i) {
            (head->neuralcluster).neurons[i].animate(numOfInputs, Activation::SIGMOID);
        }
        numOfNeuronsOfPreviousCluster = (head->neuralcluster).numOfNeurons;
    }
    else {
        Domain* current = head;
        while (current->pNextDomain != nullptr) {
            current = current->pNextDomain;
        }
        current->pNextDomain = new Domain(neuralcluster, current);
        
        for (int i = 0; i < (current->pNextDomain->neuralcluster).numOfNeurons; ++i) {
            (current->pNextDomain->neuralcluster).neurons[i].animate(numOfNeuronsOfPreviousCluster, Activation::SIGMOID);

            for (int j = 0; j < numOfNeuronsOfPreviousCluster; ++j) {
                (current->pNextDomain->neuralcluster).neurons[i].inputs[j] = &((current->pNextDomain->pPreviousDomain->neuralcluster).neurons[j].condition);
            }
        }
        numOfNeuronsOfPreviousCluster = (current->pNextDomain->neuralcluster).numOfNeurons;
    }
}

template<typename T>
void NeuralNetwork<T>::mountDataSet(const DataSet<T>& dataset) {
    pDataSet = &dataset;
};


template<typename T>
void NeuralNetwork<T>::trainNerve(const int& epochs) const {
    Domain* current = tail;
    //NeuralCluster<T>* pNeuralCluster = &current->layer;
    //T r;
    //T delta_err;

    //Vector<T> input(pDataSet->inputs.cols);
    Vector<T> label(pDataSet->labels.rows);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        //std::cout << "!!!!!!!!!!!!!!!!!epoch is: " << epoch << std::endl;

        for (int j = 0; j < pDataSet->inputs.rows; ++j)
        {
            for (int i = 0; i < pDataSet->inputs.cols; ++i) {
                *input.pdata[i] = pDataSet->inputs[i][j];
            }

            for (int i = 0; i < pDataSet->labels.rows; ++i) {
                label[i] = pDataSet->labels[j][i];
            }

            feedForward(input);

            m_error = 0;
/*
            for (int i = 0; i < (pLayer->outputs).len; ++i) {
                delta_err = label[i] - pLayer->outputs[i];
                m_error += delta_err * delta_err;
            }

            r = ((pLayer->outputs).len > 1) ? (pLayer->outputs).len - 1 : 1;
            m_error = m_error / r;
            m_error = static_cast<T>(sqrt(m_error));
*/
            //printf("error = %f\r\n", m_error);

            //BackPropagation(input, label);
        }
    }
}


template<typename T>
void NeuralNetwork<T>::feedForward(const Vector<T>& input) const {
    Domain* current = head;

    while (current != nullptr) {
        if (current->pPreviousDomain == nullptr) {
            for (int i = 0; i < (current->neuralcluster).numOfNeurons; ++i) {
                for (int j = 0; j < input.len; ++j) {
                    (current->neuralcluster).neurons[i].inputs[j] = input.pdata[j];
                }
            }
        }

        (current->neuralcluster).forwardProp();
        current = current->pNextDomain;
    }
}

template <typename T>
struct DataSet {
    DataSet(const Matrix<T>& inputs, const Matrix<T>& labels) : inputs(inputs), labels(labels) {};
    const Matrix<T>& inputs;
    const Matrix<T>& labels;
};

int main(int argc, char* argv[])
{
    const Matrix<double> inputs(8, 3);
    inputs[0][0] = 0; inputs[1][0] = 0; inputs[2][0] = 0;
    inputs[0][1] = 0; inputs[1][1] = 0; inputs[2][1] = 1;
    inputs[0][2] = 0; inputs[1][2] = 1; inputs[2][2] = 0;
    inputs[0][3] = 0; inputs[1][3] = 1; inputs[2][3] = 1;
    inputs[0][4] = 1; inputs[1][4] = 0; inputs[2][4] = 0;
    inputs[0][5] = 1; inputs[1][5] = 0; inputs[2][5] = 1;
    inputs[0][6] = 1; inputs[1][6] = 1; inputs[2][6] = 0;
    inputs[0][7] = 1; inputs[1][7] = 1; inputs[2][7] = 1;


    const Matrix<double> expectedLabels(1, 8);
    expectedLabels[0][0] = 0; expectedLabels[1][0] = 1; expectedLabels[2][0] = 1;  expectedLabels[3][0] = 0;
    expectedLabels[4][0] = 1; expectedLabels[5][0] = 0; expectedLabels[6][0] = 0;  expectedLabels[7][0] = 1;

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

    DataSet<double> dataset(inputs, expectedLabels);
    nn.mountDataSet(dataset);

    nn.trainNerve(8000);

    std::cout << "Hello World!\n";

    return 0;


}
