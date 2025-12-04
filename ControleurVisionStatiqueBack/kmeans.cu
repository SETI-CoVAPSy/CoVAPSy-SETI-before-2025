#ifdef USE_CUDA
#include <cuda_runtime.h>
#define hipMemcpy(dst, src, size, kind) cudaMemcpy(dst, src, size, kind)
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemset(ptr, value, size) cudaMemset(ptr, value, size)
#define hipStreamSynchronize(str) cudaStreamSynchronize(str)
#define hipStreamCreate(ptr) cudaStreamCreate(ptr)
#define hipStreamDestroy(str) cudaStreamDestroy(str)
#define hipStream_t cudaStream_t
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMalloc(ptr, size) cudaMalloc(ptr, size)
#define hipFree(ptr) cudaFree(ptr)
#else
#include <hip/hip_runtime.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <fstream>

namespace py = pybind11;

static int constexpr BLOCK_DIM = 1024;
static constexpr int K = 5;
static constexpr int n_features = 3;
static constexpr float learning_rate = 1;

__device__ bool est_rouge_ou_vert(const float* __restrict__ d_X, int i) {
    float h = d_X[i * n_features];
    bool vert = h >= 20 && h < 80;
    bool rouge = (h < 20 || (h >= 160 && h < 180));
    return vert || rouge;
}

__global__ void predict_kernel(int* __restrict__ d_labels, const float* __restrict__ d_X, const float* __restrict__ d_centroids, const int n_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; /* Un i par observation de X */
    if (i >= n_samples) return;
    int best_k = 0;
    if (true && est_rouge_ou_vert(d_X, i)) best_k = -1;
    else {
        float best_dist = INFINITY;
        for (int k = 0; k < K; k++) {
            float dist = 0.0;
            for (int f = 0 ; f < n_features ; f++) {
                float diff = d_X[i * n_features + f] - d_centroids[k * n_features + f];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
    }
    d_labels[i] = best_k;
}

__global__ void count_sum_kernel(int* __restrict__ d_counts, float* __restrict__ d_sums, const int* __restrict__ d_labels,
                                 const float* __restrict__ d_X, const int n_samples, const int GRID_DIM,
                                 const int* shared_counts_in, const float* shared_sums_in) {
    extern __shared__ int shared_counts[];
    float* shared_sums = (float*) &shared_counts[GRID_DIM * K];
    if (shared_counts_in == NULL) {
        const int num_threads = BLOCK_DIM * GRID_DIM;
        const int elements_per_block = BLOCK_DIM * ((n_samples + num_threads - 1) / num_threads);
        for (int k = 0; k < K; k++) {
            shared_counts[threadIdx.x * K + k] = 0;
            for (int f = 0; f < n_features; f++) shared_sums[threadIdx.x * K * n_features + k * n_features + f] = 0;
        }
        const int range_a = elements_per_block * blockIdx.x;
        const int range_b = min(elements_per_block * (blockIdx.x + 1), n_samples);
        for (int idx = range_a + threadIdx.x ; idx < range_b ; idx += blockDim.x) {
            int best_k = d_labels[idx];
            if (best_k != -1) {
                shared_counts[threadIdx.x * K + best_k] += 1;
                for (int f = 0 ; f < n_features ; f++) shared_sums[threadIdx.x * n_features * K + best_k * n_features + f] += d_X[idx * n_features + f];
            }
        }
    } else {
        for (int k = 0; k < K; k++) {
            int idx_counts = threadIdx.x * K + k;
            shared_counts[idx_counts] = shared_counts_in[idx_counts];
            for (int f = 0; f < n_features; f++) {
                int idx_sums = threadIdx.x * K * n_features + k * n_features + f;
                shared_sums[idx_sums] = shared_sums_in[idx_sums];
            }
        }
    }
    __syncthreads();
    for (int fact = 1 ; fact <= BLOCK_DIM / 2 ; fact *= 2) {
        if (threadIdx.x % (fact * 2) == 0) {
            for (int k = 0 ; k < K ; k++) {
                int idx_counts = threadIdx.x * K + k;
                shared_counts[idx_counts] += shared_counts[idx_counts + fact * K];
                for (int f = 0; f < n_features; f++) {
                    int idx_sums = threadIdx.x * K * n_features + k * n_features + f;
                    shared_sums[idx_sums] = shared_sums[idx_sums + fact * K * n_features];
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0 ; k < K ; k++) {
            d_counts[blockIdx.x * K + k] = shared_counts[k];
            for (int f = 0 ; f < n_features ; f++) d_sums[blockIdx.x * K * n_features + k * n_features + f] = shared_sums[k];
        }
    }
}

__global__ void moyenne_centroides_kernel(float* __restrict__ d_centroids, const float* __restrict__ d_sums,
                                          const int* __restrict__ d_counts) {
    int k = threadIdx.x;
    if (d_counts[k] > 0) {
        for (int f = 0; f < n_features; f++) {
            d_centroids[k * n_features + f] = d_sums[k * n_features + f] / d_counts[k];
        }
    }
}

__global__ void apply_learning_rate_kernel(float* __restrict__ d_centroids, const float* __restrict__ d_old_centroids) {
    int k = threadIdx.x;
    float vec[n_features];
    float norm = 0;
    for (int f = 0; f < n_features; f++) {
        int index = k * n_features + f;
        vec[f] = d_centroids[index] - d_old_centroids[index];
        norm += vec[f] * vec[f];
    }
    norm = std::sqrt(norm);
    if (norm == 0) norm = 1;
    float facteur = learning_rate/norm;
    if (facteur > 1.0f) facteur = 1.0f;
    for (int f = 0; f < n_features; f++) {
        int index = k * n_features + f;
        d_centroids[index] = d_old_centroids[index] + vec[f] * facteur;
    }
}

class KMeans {
private:
    const int GRID_DIM;
    const int n_samples;
    float h_centroids[K][n_features];
    float h_old_centroids[K][n_features];
    float* d_centroids;
    float* d_old_centroids;
    float* d_X;
    int* d_labels;
    float * d_sums;
    int* d_counts;
    hipStream_t stream_predict;
    hipStream_t stream_adapt; /* peut se dérouler de manière asynchrone en dehors de l'appel des fonctions */
    int dominant_cluster;

    void _allocate_memory() {
        hipMalloc(&d_centroids, K * n_features * sizeof(float));
        hipMalloc(&d_old_centroids, K * n_features * sizeof(float));
        hipMalloc(&d_X, n_samples * n_features * sizeof(float));
        hipMalloc(&d_labels, n_samples * sizeof(int));
        hipMalloc(&d_sums, K * n_features * BLOCK_DIM * sizeof(float));
        hipMalloc(&d_counts, K * BLOCK_DIM * sizeof(int));
    }

    void _free_memory() {
        hipFree(d_centroids);
        hipFree(d_old_centroids);
        hipFree(d_X);
        hipFree(d_labels);
        hipFree(d_sums);
        hipFree(d_counts);
    }

    void _copy_to_device() {
        hipMemcpy(d_centroids, h_centroids, K * n_features * sizeof(float), hipMemcpyHostToDevice);
    }

    void _copy_to_host() {
        hipMemcpy(h_centroids, d_centroids, K * n_features * sizeof(float), hipMemcpyDeviceToHost);
    }

    void _init_kpp(const float* X) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> uni(0, n_samples - 1);

        int first_idx = uni(rng);
        for (int f = 0; f < n_features; f++)
            h_centroids[0][f] = X[first_idx * n_features + f];

        float distances[150000]; // assuming n_samples <= 150000

        for (int k = 1; k < K; k++) {
            for (int i = 0; i < n_samples; i++) {
                distances[i] = std::numeric_limits<float>::max();
                for (int j = 0; j < k; ++j) {
                    float dist = 0;
                    for (int f = 0; f < n_features; f++)
                        dist += (X[i*n_features+f] - h_centroids[j][f]) * (X[i*n_features+f] - h_centroids[j][f]);
                    distances[i] = std::min(distances[i], dist);
                }
            }
            std::discrete_distribution<int> distribution(distances, distances + n_samples);
            int choice = distribution(rng);
            for (int f = 0; f < n_features; f++)
                h_centroids[k][f] = X[choice * n_features + f];
        }
    }

    void _fit_iteration(const float* X) {
        hipMemcpy(d_old_centroids, d_centroids, K * n_features * sizeof(float), hipMemcpyDeviceToDevice);
        hipMemcpy(d_X, X, n_samples * n_features * sizeof(float), hipMemcpyHostToDevice);
        _predict();
        _count_and_sum();
        _moyenne_centroides();
    }

    void _moyenne_centroides() {
        moyenne_centroides_kernel<<<1, dim3(K), 0 , stream_adapt>>>(d_centroids, d_sums, d_counts);
    }

    void _apply_learning_rate() {
        apply_learning_rate_kernel<<<1, dim3(K), 0, stream_adapt>>>(d_centroids, d_old_centroids);
    }

    void _predict() {
        predict_kernel<<<dim3(GRID_DIM), dim3(BLOCK_DIM), 0, stream_predict>>>(d_labels, d_X, d_centroids, n_samples);
    }

    void _count_and_sum() {
        hipMemset(d_counts, 0, K * GRID_DIM * sizeof(int));
        hipMemset(d_sums, 0, K * n_features * sizeof(float));
        count_sum_kernel<<<dim3(GRID_DIM), dim3(BLOCK_DIM), BLOCK_DIM * K * sizeof(int) + BLOCK_DIM * K * n_features * sizeof(float), stream_adapt>>>(d_counts, d_sums, d_labels, d_X, n_samples, GRID_DIM, NULL, NULL);
        count_sum_kernel<<<dim3(1), dim3(BLOCK_DIM),BLOCK_DIM * K * sizeof(int) + BLOCK_DIM * K * n_features * sizeof(float), stream_adapt>>>(d_counts, d_sums, d_labels, d_X, n_samples, GRID_DIM, d_counts, d_sums);
    }

    float _get_energy(py::array_t<float> X_) {
        hipStreamSynchronize(stream_adapt);
        _copy_to_host();
        auto X = X_.unchecked<2>();
        float total_energy = 0.0;

        for (int i = 0; i < n_samples; i++) {
            float min_dist = std::numeric_limits<float>::max();
            for (int k = 0; k < K; k++) {
                float dist = 0;
                for (int f = 0; f < n_features; f++) {
                    float diff = X(i, f) - h_centroids[k][f];
                    dist += diff * diff;
                }
                min_dist = std::min(min_dist, dist);
            }
            total_energy += min_dist;
        }
        return total_energy;
    }

public:
    KMeans(int n_samples_) : GRID_DIM((n_samples_ + BLOCK_DIM-1) / BLOCK_DIM), n_samples(n_samples_), dominant_cluster(-1) {
        _allocate_memory();
        hipStreamCreate(&stream_predict);
        hipStreamCreate(&stream_adapt);
    }

    ~KMeans() {
        _free_memory();
        hipStreamDestroy(stream_predict);
        hipStreamDestroy(stream_adapt);
    }

    void fit(py::array_t<float> X_) {
        hipStreamSynchronize(stream_adapt);
        _copy_to_host();
        auto X = X_.unchecked<2>();

        float best_centroids[K][n_features];
        float best_energy = std::numeric_limits<float>::max();

        for (int i = 0; i < 10; i++) {
            _init_kpp(X.data(0, 0));
            _copy_to_device();

            for (int i = 0; i < 5; i++) {
                hipStreamSynchronize(stream_adapt);
                _fit_iteration(X.data(0, 0));
            }

            float current_energy = _get_energy(X_);

            if (current_energy < best_energy) {
                best_energy = current_energy;
                memcpy(best_centroids, h_centroids, sizeof(h_centroids));
            }
        }

        memcpy(h_centroids, best_centroids, sizeof(h_centroids));
        _copy_to_device();
    }

    py::array_t<int> predict_and_adapt(py::array_t<float> X_) {
        auto X = X_.unchecked<2>();
        py::array_t<int> labels_(n_samples);
        auto labels = labels_.mutable_unchecked<1>();
        hipStreamSynchronize(stream_adapt);
        _fit_iteration(X.data(0,0));
        _count_and_sum();
        _apply_learning_rate();
        hipMemcpy(labels.mutable_data(0), d_labels, n_samples * sizeof(int), hipMemcpyDeviceToHost);
        return labels_;
    }

    py::array_t<int> predict(py::array_t<float> X_) {
        auto X = X_.unchecked<2>();
        py::array_t<int> labels_(n_samples);
        auto labels = labels_.mutable_unchecked<1>();
        hipStreamSynchronize(stream_adapt);
        hipMemcpy(d_X, X.data(0,0), n_samples * n_features * sizeof(float), hipMemcpyHostToDevice);
        _predict();
        hipStreamSynchronize(stream_predict);
        hipMemcpy(labels.mutable_data(0), d_labels, n_samples * sizeof(int), hipMemcpyDeviceToHost);
        return labels_;
    }

    void save(const std::string& filename) {
        hipStreamSynchronize(stream_adapt);
        _copy_to_host();
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<char*> (h_centroids), sizeof (h_centroids));
        file.write(reinterpret_cast<char*> (h_old_centroids), sizeof (h_old_centroids));
        file.close();
    }

    void load(const std::string& filename) {
        hipStreamSynchronize(stream_adapt);
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Could not open file");
        file.read(reinterpret_cast<char*> (h_centroids), sizeof (h_centroids));
        file.read(reinterpret_cast<char*> (h_old_centroids), sizeof (h_old_centroids));
        file.close();
        _copy_to_device();
    }

    int get_dominant_cluster() {
        if (dominant_cluster >= 0) return dominant_cluster;
        int h_counts[K];
        hipMemcpy(h_counts, d_counts, K * sizeof(int), hipMemcpyDeviceToHost);
        int max = std::numeric_limits<int>::min();
        int ret = 0;
        for (int i = 0; i < K; i++) {
            if (h_counts[i] > max) {
                max = h_counts[i];
                ret = i;
            }
        }
        return ret;
    }
};

PYBIND11_MODULE(KMeans, m) {
py::class_<KMeans>(m, "KMeans")
.def(py::init<int>(), py::arg("n_samples"))
.def("fit", &KMeans::fit)
.def("predict_and_adapt", &KMeans::predict_and_adapt)
.def("predict", &KMeans::predict)
.def("save", &KMeans::save)
.def("load", &KMeans::load)
.def("get_dominant_cluster", &KMeans::get_dominant_cluster);
}
