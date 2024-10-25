#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Add this with your other activation functions
double tanh(double x) {
    return std::tanh(x);
}

// matrix class for handling matrix operations
class Matrix {
   public:
    std::vector<std::vector<double>> data;  // 2D vector to store matrix data
    size_t rows, cols;                      // dimensions of the matrix

    /**
     * @brief Constructs a Matrix object with the specified dimensions.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows,
                    std::vector<double>(cols, 0.0));  // resise the data vector to have 'rows' elements with a vector as the value
    }

    /**
     * @brief Initializes the matrix with random values between -1 and 1.
     */
    void uniform_initialise() {
        std::random_device rd;                            // obtain a random number from hardware
        std::mt19937 gen(rd());                           // seed the generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // define the range
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix with zeros.
     */
    void zero_initialise() {
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = 0;  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix using Xavier initialization method.
     * Suitable for sigmoid activation functions.
     */
    void xavier_initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = sqrt(6.0 / (rows + cols));
        std::uniform_real_distribution<> dis(-limit, limit);
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Initializes the matrix using He initialization method.
     * Suitable for ReLU activation functions.
     */
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev = sqrt(2.0 / cols);
        std::normal_distribution<> dis(0, std_dev);
        for (auto &row : data) {
            for (auto &elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Overloads the multiplication operator for matrix multiplication.
     * @param other The matrix to multiply with.
     * @return The resulting matrix after multiplication.
     */
    Matrix operator*(const Matrix &other) const {
        if (cols != other.rows) {
            std::cerr << "Attempted to multiply matrices of incompatible dimensions: "
                      << "(" << rows << "x" << cols << ") * (" << other.rows << "x" << other.cols << ")" << std::endl;
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        Matrix result(rows, other.cols);
        // weird loop order (k before j) makes more cache friendly
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < cols; k++) {
                for (size_t j = 0; j < other.cols; j++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    /**
     * @brief Overloads the addition operator for element-wise matrix addition.
     * @param other The matrix to add.
     * @return The resulting matrix after addition.
     */
    Matrix operator+(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the subtraction operator for element-wise matrix subtraction.
     * @param other The matrix to subtract.
     * @return The resulting matrix after subtraction.
     */
    Matrix operator-(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the multiplication operator for scalar multiplication.
     * @param scalar The scalar value to multiply with.
     * @return The resulting matrix after scalar multiplication.
     */
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * @brief Computes the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Computes the Hadamard product (element-wise multiplication) of two matrices.
     * @param other The matrix to perform Hadamard product with.
     * @return The resulting matrix after Hadamard product.
     */
    Matrix hadamard(const Matrix &other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @param func A function pointer to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @tparam Func The type of the callable object.
     * @param func A callable object to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    template <typename Func>
    Matrix apply(Func func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies the softmax function to the matrix.
     * @return The resulting matrix after applying softmax.
     */
    Matrix softmax() const {
        Matrix result(rows, cols);

        for (size_t j = 0; j < cols; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, data[i][j]);
            }

            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] = std::exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }

            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }
};

class GRUCell {
   private:
    // Gate weights and biases
    Matrix W_z;  // Update gate weights for input
    Matrix U_z;  // Update gate weights for hidden state
    Matrix b_z;  // Update gate bias

    Matrix W_r;  // Reset gate weights for input
    Matrix U_r;  // Reset gate weights for hidden state
    Matrix b_r;  // Reset gate bias

    Matrix W_h;  // Candidate hidden state weights for input
    Matrix U_h;  // Candidate hidden state weights for hidden state
    Matrix b_h;  // Candidate hidden state bias

    size_t input_size;
    size_t hidden_size;

   public:
    GRUCell(size_t input_size, size_t hidden_size)
        : input_size(input_size),
          hidden_size(hidden_size),
          W_z(hidden_size, input_size),
          U_z(hidden_size, hidden_size),
          b_z(hidden_size, 1),
          W_r(hidden_size, input_size),
          U_r(hidden_size, hidden_size),
          b_r(hidden_size, 1),
          W_h(hidden_size, input_size),
          U_h(hidden_size, hidden_size),
          b_h(hidden_size, 1) {
        // Initialize weights using Xavier initialization
        W_z.xavier_initialize();
        U_z.xavier_initialize();
        W_r.xavier_initialize();
        U_r.xavier_initialize();
        W_h.xavier_initialize();
        U_h.xavier_initialize();

        // Initialize biases with zeros
        b_z.zero_initialise();
        b_r.zero_initialise();
        b_h.zero_initialise();
    }

    Matrix forward(const Matrix &x, const Matrix &h_prev) {
        // Update gate: determines how much of the past information to keep
        Matrix z = (W_z * x + U_z * h_prev + b_z).apply(sigmoid);

        // Reset gate: determines how much of the past information to forget
        Matrix r = (W_r * x + U_r * h_prev + b_r).apply(sigmoid);

        // Candidate hidden state: combines current input with filtered past information
        Matrix h_candidate = (W_h * x + U_h * (r.hadamard(h_prev)) + b_h).apply(tanh);

        // Final hidden state: weighted sum of previous hidden state and candidate hidden state
        Matrix h = z.hadamard(h_prev) + (z.apply([](double x) { return 1.0 - x; }).hadamard(h_candidate));

        return h;
    }
};


