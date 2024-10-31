#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

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
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix with zeros.
     */
    void zero_initialise() {
        for (auto& row : data) {
            for (auto& elem : row) {
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
        for (auto& row : data) {
            for (auto& elem : row) {
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
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Overloads the multiplication operator for matrix multiplication.
     * @param other The matrix to multiply with.
     * @return The resulting matrix after multiplication.
     */
    Matrix operator*(const Matrix& other) const {
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
    Matrix operator+(const Matrix& other) const {
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
    Matrix operator-(const Matrix& other) const {
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
    Matrix hadamard(const Matrix& other) const {
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

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        os << "[";
        for (size_t i = 0; i < matrix.rows; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < matrix.cols; ++j) {
                os << matrix.data[i][j];
                if (j < matrix.cols - 1) os << ", ";
            }
            os << "]";
            if (i < matrix.rows - 1) os << "\n";
        }
        os << "]";
        return os;
    }
};



struct TrainingExample {
    std::vector<Matrix> sequence;
    Matrix target;

    TrainingExample() : target(1, 1) {}  // initialize target with size 1x1 as matrix does not have default constructor
};

// gradient storage struct for GRUCell -- defined globally to avoid circular dependency
struct GRUGradients {
    Matrix dW_z, dU_z, db_z;  // update gate gradients
    Matrix dW_r, dU_r, db_r;  // reset gate gradients
    Matrix dW_h, dU_h, db_h;  // hidden state gradients
    size_t input_size;
    size_t hidden_size;

    GRUGradients(size_t input_size, size_t hidden_size)
        : dW_z(hidden_size, input_size),
          dU_z(hidden_size, hidden_size),
          db_z(hidden_size, 1),
          dW_r(hidden_size, input_size),
          dU_r(hidden_size, hidden_size),
          db_r(hidden_size, 1),
          dW_h(hidden_size, input_size),
          dU_h(hidden_size, hidden_size),
          db_h(hidden_size, 1),
          input_size(input_size),
          hidden_size(hidden_size) {}
    
    // operator overloading for addition
    GRUGradients operator+(const GRUGradients& other) const {
        if (input_size != other.input_size or hidden_size != other.hidden_size) {
            throw std::invalid_argument("GRUGradients dimensions don't match for addition");
        }
        GRUGradients result(input_size, hidden_size);
        result.dW_z = dW_z + other.dW_z;
        result.dU_z = dU_z + other.dU_z;
        result.db_z = db_z + other.db_z;
        result.dW_r = dW_r + other.dW_r;
        result.dU_r = dU_r + other.dU_r;
        result.db_r = db_r + other.db_r;
        result.dW_h = dW_h + other.dW_h;
        result.dU_h = dU_h + other.dU_h;
        result.db_h = db_h + other.db_h;
        return result;
    }

    // operator overloading for scalar multiplication
    GRUGradients operator*(double scalar) const {
        GRUGradients result(input_size, hidden_size);
        result.dW_z = dW_z * scalar;
        result.dU_z = dU_z * scalar;
        result.db_z = db_z * scalar;
        result.dW_r = dW_r * scalar;
        result.dU_r = dU_r * scalar;
        result.db_r = db_r * scalar;
        result.dW_h = dW_h * scalar;
        result.dU_h = dU_h * scalar;
        result.db_h = db_h * scalar;
        return result;
    }
};

class GRUCell {
   private:
    // store sequence of states for BPTT
    struct TimeStep {
        Matrix z, r, h_candidate, h;
        Matrix h_prev;
        Matrix x;

        TimeStep(size_t hidden_size, size_t input_size)
            : z(hidden_size, 1),
              r(hidden_size, 1),
              h_candidate(hidden_size, 1),
              h(hidden_size, 1),
              h_prev(hidden_size, 1),
              x(input_size, 1) {}
    };
    std::vector<TimeStep> time_steps;

    // clear the stored states
    void clear_states() { time_steps.clear(); }

    // store the gradients and the gradient of the hidden state from the previous timestep
    struct BackwardResult {
        GRUGradients grads;
        Matrix dh_prev;

        BackwardResult(GRUGradients g, Matrix h) : grads(g), dh_prev(h) {}
    };

    // compute gradients for a single timestep
    BackwardResult backward(const Matrix& delta_h_t, size_t t) {
        if (t >= time_steps.size()) {
            throw std::runtime_error("Time step index out of bounds");
        }

        // get the stored states for this timestep
        const TimeStep& step = time_steps[t];

        // initialize gradients for this timestep
        GRUGradients timestep_grads(input_size, hidden_size);

        // 1. Hidden state gradients
        Matrix one_matrix(delta_h_t.rows, delta_h_t.cols);
        one_matrix.zero_initialise();
        for (size_t i = 0; i < one_matrix.rows; i++)
            for (size_t j = 0; j < one_matrix.cols; j++) one_matrix.data[i][j] = 1.0;

        Matrix dh_tilde = delta_h_t.hadamard(one_matrix - step.z);
        Matrix dz = delta_h_t.hadamard(step.h_prev - step.h_candidate);

        // 2. Candidate state gradients
        Matrix dg = dh_tilde.hadamard(step.h_candidate.apply([](double x) { return 1.0 - x * x; })  // tanh derivative
        );

        timestep_grads.dW_h = dg * step.x.transpose();
        timestep_grads.dU_h = dg * (step.r.hadamard(step.h_prev)).transpose();
        timestep_grads.db_h = dg;

        Matrix dx_t = timestep_grads.dW_h.transpose() * dg;
        Matrix dr = (timestep_grads.dU_h.transpose() * dg).hadamard(step.h_prev);
        Matrix dh_prev = (timestep_grads.dU_h.transpose() * dg).hadamard(step.r);

        // 3. Reset gate gradients
        Matrix dr_total = dr.hadamard(step.r.apply(sigmoid_derivative));

        timestep_grads.dW_r = dr_total * step.x.transpose();
        timestep_grads.dU_r = dr_total * step.h_prev.transpose();
        timestep_grads.db_r = dr_total;

        dx_t = dx_t + timestep_grads.dW_r.transpose() * dr_total;
        dh_prev = dh_prev + timestep_grads.dU_r.transpose() * dr_total;

        // 4. Update gate gradients
        Matrix dz_total = dz.hadamard(step.z.apply(sigmoid_derivative));

        timestep_grads.dW_z = dz_total * step.x.transpose();
        timestep_grads.dU_z = dz_total * step.h_prev.transpose();
        timestep_grads.db_z = dz_total;

        dx_t = dx_t + timestep_grads.dW_z.transpose() * dz_total;
        dh_prev = dh_prev + timestep_grads.dU_z.transpose() * dz_total;

        // 5. Final hidden state contribution
        dh_prev = dh_prev + delta_h_t.hadamard(step.z);

        // return both values
        return BackwardResult(timestep_grads, dh_prev);
    }

   public:
    // gate weights and biases
    Matrix W_z;  // update gate weights for input
    Matrix U_z;  // update gate weights for hidden state
    Matrix b_z;  // update gate bias

    Matrix W_r;  // reset gate weights for input
    Matrix U_r;  // reset gate weights for hidden state
    Matrix b_r;  // reset gate bias

    Matrix W_h;  // candidate hidden state weights for input
    Matrix U_h;  // candidate hidden state weights for hidden state
    Matrix b_h;  // candidate hidden state bias

    size_t input_size;
    size_t hidden_size;

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
        // initialize weights using Xavier initialization
        W_z.xavier_initialize();
        U_z.xavier_initialize();
        W_r.xavier_initialize();
        U_r.xavier_initialize();
        W_h.xavier_initialize();
        U_h.xavier_initialize();

        // biases are initialised to zero by default
    }

    // get final hidden state
    Matrix get_last_hidden_state() const {
        if (time_steps.empty()) {
            throw std::runtime_error("No hidden state available - run forward pass first");
        }
        return time_steps.back().h;
    }

    // resets gradients in GRUGradients struct to zero
    void reset_gradients(GRUGradients& grads) {
        grads.dW_z.zero_initialise();
        grads.dU_z.zero_initialise();
        grads.db_z.zero_initialise();
        grads.dW_r.zero_initialise();
        grads.dU_r.zero_initialise();
        grads.db_r.zero_initialise();
        grads.dW_h.zero_initialise();
        grads.dU_h.zero_initialise();
        grads.db_h.zero_initialise();
    }

    // forward pass that stores states
    Matrix forward(const Matrix& x, const Matrix& h_prev) {
        TimeStep step(hidden_size, input_size);
        step.x = x;
        step.h_prev = h_prev;

        // update gate
        step.z = (W_z * x + U_z * h_prev + b_z).apply(sigmoid);

        // reset gate
        step.r = (W_r * x + U_r * h_prev + b_r).apply(sigmoid);

        // candidate hidden state
        step.h_candidate = (W_h * x + U_h * (step.r.hadamard(h_prev)) + b_h).apply(std::tanh);

        // final hidden state
        step.h = step.z.hadamard(h_prev) + (step.z.apply([](double x) { return 1.0 - x; }).hadamard(step.h_candidate));

        time_steps.push_back(step);
        return step.h;
    }

    GRUGradients backpropagate(const Matrix& final_gradient) {
        Matrix dh_next = final_gradient;
        GRUGradients accumulated_grads(input_size, hidden_size);
        reset_gradients(accumulated_grads);

        // backpropagate through time
        for (int t = time_steps.size() - 1; t >= 0; --t) {
            BackwardResult result = backward(dh_next, t);
            dh_next = result.dh_prev;

            // accumulate gradients for update gate
            accumulated_grads.dW_z = accumulated_grads.dW_z + result.grads.dW_z;
            accumulated_grads.dU_z = accumulated_grads.dU_z + result.grads.dU_z;
            accumulated_grads.db_z = accumulated_grads.db_z + result.grads.db_z;

            // accumulate gradients for reset gate
            accumulated_grads.dW_r = accumulated_grads.dW_r + result.grads.dW_r;
            accumulated_grads.dU_r = accumulated_grads.dU_r + result.grads.dU_r;
            accumulated_grads.db_r = accumulated_grads.db_r + result.grads.db_r;

            // accumulate gradients for candidate hidden state
            accumulated_grads.dW_h = accumulated_grads.dW_h + result.grads.dW_h;
            accumulated_grads.dU_h = accumulated_grads.dU_h + result.grads.dU_h;
            accumulated_grads.db_h = accumulated_grads.db_h + result.grads.db_h;
        }

        clear_states();
        return accumulated_grads;
    }
};

class Optimiser {
   public:
    virtual void compute_and_apply_updates(GRUCell& gru, const GRUGradients& gradients) = 0;
    virtual ~Optimiser() = default;
};

class SGDOptimiser : public Optimiser {
   private:
    double learning_rate;

   public:
    SGDOptimiser(double lr = 0.1) : learning_rate(lr) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // update weights and biases using gradients
        gru.W_z = gru.W_z - grads.dW_z * learning_rate;
        gru.U_z = gru.U_z - grads.dU_z * learning_rate;
        gru.b_z = gru.b_z - grads.db_z * learning_rate;

        gru.W_r = gru.W_r - grads.dW_r * learning_rate;
        gru.U_r = gru.U_r - grads.dU_r * learning_rate;
        gru.b_r = gru.b_r - grads.db_r * learning_rate;

        gru.W_h = gru.W_h - grads.dW_h * learning_rate;
        gru.U_h = gru.U_h - grads.dU_h * learning_rate;
        gru.b_h = gru.b_h - grads.db_h * learning_rate;
    }
};

class SGDMomentumOptimiser : public Optimiser {
   private:
    double learning_rate;
    double momentum;
    GRUGradients velocity;

   public:
    SGDMomentumOptimiser(double lr = 0.1, double mom = 0.9)
        : learning_rate(lr), momentum(mom), velocity(0, 0) {}  // sizes will be set on first use

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // initialize velocity if needed
        if (velocity.dW_z.rows == 0) {
            velocity = GRUGradients(gru.input_size, gru.hidden_size);
        }

        // update velocities and apply updates for each parameter
        velocity.dW_z = velocity.dW_z * momentum - grads.dW_z * learning_rate;
        velocity.dU_z = velocity.dU_z * momentum - grads.dU_z * learning_rate;
        velocity.db_z = velocity.db_z * momentum - grads.db_z * learning_rate;

        velocity.dW_r = velocity.dW_r * momentum - grads.dW_r * learning_rate;
        velocity.dU_r = velocity.dU_r * momentum - grads.dU_r * learning_rate;
        velocity.db_r = velocity.db_r * momentum - grads.db_r * learning_rate;

        velocity.dW_h = velocity.dW_h * momentum - grads.dW_h * learning_rate;
        velocity.dU_h = velocity.dU_h * momentum - grads.dU_h * learning_rate;
        velocity.db_h = velocity.db_h * momentum - grads.db_h * learning_rate;

        // apply updates
        gru.W_z = gru.W_z + velocity.dW_z;
        gru.U_z = gru.U_z + velocity.dU_z;
        gru.b_z = gru.b_z + velocity.db_z;

        gru.W_r = gru.W_r + velocity.dW_r;
        gru.U_r = gru.U_r + velocity.dU_r;
        gru.b_r = gru.b_r + velocity.db_r;

        gru.W_h = gru.W_h + velocity.dW_h;
        gru.U_h = gru.U_h + velocity.dU_h;
        gru.b_h = gru.b_h + velocity.db_h;
    }
};

class AdamOptimiser : public Optimiser {
   private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Matrix& param, Matrix& m_param, Matrix& v_param, const Matrix& grad) {
        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0 - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0 - beta2);

        // Compute bias-corrected first moment estimate
        Matrix m_hat = m_param * (1.0 / (1.0 - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Matrix v_hat = v_param * (1.0 / (1.0 - std::pow(beta2, t)));

        // Update parameters
        Matrix denom = v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); });
        Matrix update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    AdamOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.rows == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        t++;

        // Update all parameters
        update_parameter(gru.W_z, m.dW_z, v.dW_z, grads.dW_z);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, grads.dU_z);
        update_parameter(gru.b_z, m.db_z, v.db_z, grads.db_z);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, grads.dW_r);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, grads.dU_r);
        update_parameter(gru.b_r, m.db_r, v.db_r, grads.db_r);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, grads.dW_h);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, grads.dU_h);
        update_parameter(gru.b_h, m.db_h, v.db_h, grads.db_h);
    }
};

class AdamWOptimiser : public Optimiser {
   private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Matrix& param, Matrix& m_param, Matrix& v_param, const Matrix& grad, bool apply_weight_decay = true) {
        // Weight decay should be applied to the parameter directly
        if (apply_weight_decay) {
            param = param * (1.0 - learning_rate * weight_decay);
        }

        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0 - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0 - beta2);

        // Compute bias-corrected first moment estimate
        Matrix m_hat = m_param * (1.0 / (1.0 - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Matrix v_hat = v_param * (1.0 / (1.0 - std::pow(beta2, t)));

        // Update parameters
        Matrix denom = v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); });
        Matrix update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    AdamWOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.01)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.rows == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        t++;

        // Update weights (with weight decay)
        update_parameter(gru.W_z, m.dW_z, v.dW_z, grads.dW_z, true);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, grads.dU_z, true);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, grads.dW_r, true);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, grads.dU_r, true);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, grads.dW_h, true);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, grads.dU_h, true);

        // Update biases (without weight decay)
        update_parameter(gru.b_z, m.db_z, v.db_z, grads.db_z, false);
        update_parameter(gru.b_r, m.db_r, v.db_r, grads.db_r, false);
        update_parameter(gru.b_h, m.db_h, v.db_h, grads.db_h, false);
    }
};

class Predictor {
   private:
    GRUCell gru;
    Matrix W_out;  // output layer weights
    Matrix b_out;  // output layer bias
    double learning_rate;

    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    std::unique_ptr<Optimiser> optimiser;

   public:
    Predictor(size_t input_size, size_t hidden_size, size_t output_size, double learning_rate = 0.01)
        : gru(input_size, hidden_size),
          input_size(input_size),
          hidden_size(hidden_size),
          output_size(output_size),
          learning_rate(learning_rate),
          W_out(output_size, hidden_size),
          b_out(output_size, 1) {
        W_out.xavier_initialize();
    }

    // set optimiser - call before training
    void set_optimiser(std::unique_ptr<Optimiser> new_optimiser) { optimiser = std::move(new_optimiser); }

    // update GRU parameters using optimiser (can't be defined in GRUCell to avoid circular dependency)
    void update_parameters(const GRUGradients& grads) {
        if (!optimiser) {
            throw std::runtime_error("no optimiser set");
        }
        optimiser->compute_and_apply_updates(gru, grads);
    }

    // process sequence and return prediction
    Matrix predict(const std::vector<Matrix>& input_sequence) {
        Matrix h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
        }

        // final linear layer
        return W_out * h_t + b_out;
    }

    // gets the gradients for a single training example
    std::pair<GRUGradients, std::pair<Matrix, Matrix>> compute_gradients(const std::vector<Matrix>& input_sequence, const Matrix& target) {
        // forward pass
        Matrix prediction = predict(input_sequence);
        Matrix last_hidden_state = gru.get_last_hidden_state();

        // compute prediction error
        Matrix error = prediction - target;

        // backpropagate through output layer
        Matrix output_gradient = error * (2.0 / (error.rows * error.cols));  // MSE derivative
        Matrix hidden_gradient = W_out.transpose() * output_gradient;

        // compute complete gradients for output layer
        Matrix dW_out = output_gradient * last_hidden_state.transpose();
        Matrix db_out = output_gradient;

        // backpropagate through GRU
        auto gru_gradients = gru.backpropagate(hidden_gradient);
        return std::pair<GRUGradients, std::pair<Matrix, Matrix>>(gru_gradients, {dW_out, db_out});
    }

    void train(const std::vector<TrainingExample>& training_data, const std::vector<TrainingExample>& test_data, int epochs, int batch_size = 1) {
        // training loop
        for (int epoch = 0; epoch < epochs; epoch++) {

            int no_examples = training_data.size();

            for (int i=0; i<no_examples; i+=batch_size) {
                // create batch
                size_t batch_start = i;
                size_t batch_end = std::min(i + batch_size, no_examples);
                std::vector<TrainingExample> batch(training_data.begin() + batch_start, training_data.begin() + batch_end);

                std::vector<GRUGradients> accumulated_gru_gradients;
                std::vector<Matrix> accumulated_output_weights_gradients;
                std::vector<Matrix> accumulated_output_bias_gradients;
                accumulated_gru_gradients.reserve(batch_end - batch_start);
                accumulated_output_weights_gradients.reserve(batch_end - batch_start);
                accumulated_output_bias_gradients.reserve(batch_end - batch_start);

                for (const auto& example : batch) {
                    auto [gru_gradients, output_gradients] = compute_gradients(example.sequence, example.target);
                    auto [dW_out, db_out] = output_gradients;
                    accumulated_gru_gradients.push_back(gru_gradients);
                    accumulated_output_weights_gradients.push_back(dW_out);
                    accumulated_output_bias_gradients.push_back(db_out);
                }

                // average gradients
                GRUGradients averaged_gru_gradients(input_size, hidden_size);
                Matrix averaged_output_weights_gradients(output_size, hidden_size);
                Matrix averaged_output_bias_gradients(1, output_size);
                for (int i=0; i<batch_end - batch_start; i++) {
                    averaged_gru_gradients = averaged_gru_gradients + accumulated_gru_gradients[i];
                    averaged_output_weights_gradients = averaged_output_weights_gradients + accumulated_output_weights_gradients[i];
                    averaged_output_bias_gradients = averaged_output_bias_gradients + accumulated_output_bias_gradients[i];
                }
                averaged_gru_gradients = averaged_gru_gradients * (1.0 / (batch_end - batch_start));
                averaged_output_weights_gradients = averaged_output_weights_gradients * (1.0 / (batch_end - batch_start));
                averaged_output_bias_gradients = averaged_output_bias_gradients * (1.0 / (batch_end - batch_start));


                // update GRU parameters using optimiser once batch gradients are averaged
                update_parameters(averaged_gru_gradients);

                // update output layer weights
                // TODO: REPLACE WITH OPTIMISER CLASSES
                W_out = W_out - averaged_output_weights_gradients * learning_rate;
                b_out = b_out - averaged_output_bias_gradients * learning_rate;

                std::cout << "\rBatch " << i/batch_size << "/" << no_examples/batch_size << " complete" << std::flush;
            }
            std::cout << "\rEpoch " << epoch << "/" << epochs << " complete" << std::endl;
            auto test_metrics = evaluate(test_data);
            std::cout << test_metrics << std::endl;
        }
    }

    // add this to your Predictor class
    struct EvaluationMetrics {
        double mse;
        double mae;
        double rmse;

        friend std::ostream& operator<<(std::ostream& os, const EvaluationMetrics& metrics) {
            os << "----------------\n"
               << "MSE: " << metrics.mse << "\n"
               << "MAE: " << metrics.mae << "\n"
               << "RMSE: " << metrics.rmse << "\n"
               << "----------------";
            return os;
        }
    };

    EvaluationMetrics evaluate(const std::vector<TrainingExample>& test_data) {
        double total_loss = 0.0;
        double total_squared_error = 0.0;
        double total_absolute_error = 0.0;
        size_t total_examples = test_data.size();

        for (const auto& example : test_data) {
            // get prediction
            Matrix prediction = predict(example.sequence);
            
            // compute errors
            double error = prediction.data[0][0] - example.target.data[0][0];
            total_squared_error += error * error;
            total_absolute_error += std::abs(error);
            total_loss += error * error;  // MSE loss
        }

        // compute average metrics
        double mse = total_squared_error / total_examples;
        double mae = total_absolute_error / total_examples;
        double rmse = std::sqrt(mse);

        return {mse, mae, rmse};
    }
};


std::vector<TrainingExample> generate_sine_training_data(int num_samples, int sequence_length, double sampling_frequency = 0.1) {
    std::vector<TrainingExample> training_data;

    // generate a longer sine wave
    std::vector<double> sine_wave;
    for (int i = 0; i < num_samples + sequence_length; i++) {
        double x = i * sampling_frequency;
        sine_wave.push_back(std::sin(x));
    }

    // create sliding window examples
    for (int i = 0; i < num_samples; i++) {
        TrainingExample example;

        // create input sequence
        for (int j = 0; j < sequence_length; j++) {
            Matrix input(1, 1);  // single feature (sine value)
            input.data[0][0] = sine_wave[i + j];
            example.sequence.push_back(input);
        }

        // create target (next value in sequence)
        Matrix target(1, 1);
        target.data[0][0] = sine_wave[i + sequence_length];
        example.target = target;

        training_data.push_back(example);
    }

    return training_data;
}

std::vector<TrainingExample> load_stock_data(const std::string& filename, size_t sequence_length = 20) {
    std::vector<TrainingExample> training_data;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    
    // skip header if exists
    std::getline(file, line);
    
    // read first line to infer number of features
    std::getline(file, line);
    std::stringstream test_ss(line);
    std::string value;
    size_t n_features = 0;
    
    // skip date
    std::getline(test_ss, value, ',');
    
    // count remaining columns
    while (std::getline(test_ss, value, ',')) {
        n_features++;
    }
    
    // reset file position to start (after header)
    file.clear();
    file.seekg(0);
    std::getline(file, line);  // skip header again
    
    // temporary storage for building sequences
    std::vector<std::vector<double>> temp_sequence;
    
    // read each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        
        // skip date
        std::getline(ss, value, ',');
        
        // read features
        std::vector<double> features;
        while (std::getline(ss, value, ',') && features.size() < n_features) {
            try {
                features.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cerr << "warning: invalid number format in line: " << line << std::endl;
                continue;
            }
        }
        
        // ensure we got all features
        if (features.size() != n_features) {
            std::cerr << "warning: incorrect number of features in line: " << line << std::endl;
            continue;
        }
        
        temp_sequence.push_back(features);
        
        // once we have enough data for a sequence
        if (temp_sequence.size() >= sequence_length + 1) {  // +1 for target
            TrainingExample example;
            
            // create sequence from all but last entry
            for (size_t i = 0; i < sequence_length; i++) {
                Matrix input(n_features, 1);
                for (size_t f = 0; f < n_features; f++) {
                    input.data[f][0] = temp_sequence[i][f];
                }
                example.sequence.push_back(input);
            }
            
            // use returns from last entry as target
            example.target = Matrix(1, 1);
            example.target.data[0][0] = temp_sequence[sequence_length][0];  // returns column
            
            training_data.push_back(example);
            
            // remove oldest entry to slide window
            temp_sequence.erase(temp_sequence.begin());
        }
    }
    
    if (training_data.empty()) {
        throw std::runtime_error("no valid training examples could be created from file");
    }
    
    return training_data;
}

int main() {
    // setup for sine wave prediction
    size_t input_features = 8;  // just the sine value
    size_t hidden_size = 32;
    size_t output_size = 1;  // predicted next value

    // learning rate is for the linear layer, not the optimiser.
    Predictor predictor(input_features, hidden_size, output_size, 0.01);
    predictor.set_optimiser(std::make_unique<AdamWOptimiser>());

    // generate training data
    auto stock_data = load_stock_data("stock_data/AAPL_data.csv", 30);
    // todo normalise data
    std::shuffle(stock_data.begin(), stock_data.end(), std::mt19937(std::random_device()()));

    // split data into training and test sets
    size_t split_point = static_cast<size_t>(stock_data.size() * 0.8);
    auto training_data = std::vector<TrainingExample>(stock_data.begin(), stock_data.begin() + split_point);
    auto test_data = std::vector<TrainingExample>(stock_data.begin() + split_point, stock_data.end());

    predictor.train(training_data, test_data, 50, 50);

    // print some example predictions
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), test_data.size()); i++) {
        Matrix prediction = predictor.predict(test_data[i].sequence);
        std::cout << "Predicted: " << prediction.data[0][0] 
                  << " Actual: " << test_data[i].target.data[0][0] << std::endl;
    }
}
