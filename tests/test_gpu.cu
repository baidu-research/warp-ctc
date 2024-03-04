#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>

#include <ctc.h>

#include "test.h"

bool small_test() {
    const int alphabet_size = 5;
    const int T = 2;

    std::vector<float> activations = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                      0.1f, 0.1f, 0.6f, 0.1f, 0.1f};

    // Calculate the score analytically
    float expected_score;
    {
        std::vector<float> probs(activations.size());
        softmax(activations.data(), alphabet_size, T, probs.data());

        // Score calculation is specific to the given activations above
        expected_score = probs[1] * probs[7];
    }

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *activations_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");

#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(activations_gpu, nullptr,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    score = std::exp(-score);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

#ifdef __HIPCC__
    throw_on_error(hipFree(activations_gpu),
                   "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace),
                   "hipFree");
    throw_on_error(hipStreamDestroy(stream),
                   "hipStreamDestroy");
#else
    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
#endif

    return (score > lb && score < ub);
}

int offset(int t, int n, int a) {
    constexpr int minibatch = 2;
    constexpr int alphabet_size = 6;
    return (t * minibatch + n) * alphabet_size + a;
}

bool options_test() {
    const int alphabet_size = 6;
    const int T = 5;
    const int minibatch = 2;

    std::vector<float> activations =
        {0.633766f, 0.221185f, 0.0917319f, 0.0129757f, 0.0142857f, 0.0260553f,
         0.30176f, 0.28562f, 0.0831517f, 0.0862751f, 0.0816851f, 0.161508f,

         0.111121f, 0.588392f, 0.278779f, 0.0055756f, 0.00569609f, 0.010436f,
         0.24082f, 0.397533f, 0.0557226f, 0.0546814f, 0.0557528f, 0.19549f,

         0.0357786f, 0.633813f, 0.321418f, 0.00249248f, 0.00272882f, 0.0037688f,
         0.230246f, 0.450868f, 0.0389607f, 0.038309f, 0.0391602f, 0.202456f,

         0.0663296f, 0.643849f, 0.280111f, 0.00283995f, 0.0035545f, 0.00331533f,
         0.280884f, 0.429522f, 0.0326593f, 0.0339046f, 0.0326856f, 0.190345f,

         0.458235f, 0.396634f, 0.123377f, 0.00648837f, 0.00903441f, 0.00623107f,
         0.423286f, 0.315517f, 0.0338439f, 0.0393744f, 0.0339315f, 0.154046f};

    std::vector<float> expected_grads = // from tensorflow
        {-0.366234f, 0.221185f, 0.0917319f, 0.0129757f, 0.0142857f, 0.0260553f,
         -0.69824f, 0.28562f, 0.0831517f, 0.0862751f, 0.0816851f, 0.161508f,

         0.111121f, -0.411608f, 0.278779f, 0.0055756f, 0.00569609f, 0.010436f,
         0.24082f, -0.602467f, 0.0557226f, 0.0546814f, 0.0557528f, 0.19549f,

         0.0357786f, 0.633813f, -0.678582f, 0.00249248f, 0.00272882f, 0.0037688f,
         0.230246f, 0.450868f, 0.0389607f, 0.038309f, 0.0391602f, -0.797544f,

         0.0663296f, -0.356151f, 0.280111f, 0.00283995f, 0.0035545f, 0.00331533f,
         0.280884f, -0.570478f, 0.0326593f, 0.0339046f, 0.0326856f, 0.190345f,

         -0.541765f, 0.396634f, 0.123377f, 0.00648837f, 0.00903441f, 0.00623107f,
         -0.576714f, 0.315517f, 0.0338439f, 0.0393744f, 0.0339315f, 0.154046f};

    // Calculate the expected scores analytically
    auto& a = activations;
    double expected_score[2];
    expected_score[0] =
            -std::log(a[offset(0, 0, 0)] * a[offset(1, 0, 1)] * a[offset(2, 0, 2)]
                     * a[offset(3, 0, 1)] * a[offset(4, 0, 0)]);
    expected_score[1] = 5.42262f; // from tensorflow

    // now take the log to account for the softmax
    for (auto& a : activations) {
        a = std::log(a);
    }

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *activations_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> labels = {0, 1, 2, 1, 0,
                               0, 1, 1, 0};

    std::vector<int> label_lengths = {5, 4};

    std::vector<int> lengths = {5, 5};

    float score[2];

    float *grads_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "cudaMalloc");
#endif

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;
    options.blank_label = 5;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(activations_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score[0],
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in options_test");

    std::vector<float> grads(alphabet_size * T * minibatch);

#ifdef __HIPCC__
    throw_on_error(hipMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync");
    throw_on_error(hipStreamSynchronize(stream), "hipStreamSynchronize");

    throw_on_error(hipFree(activations_gpu),
                   "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace),
                   "hipFree");
    throw_on_error(hipStreamDestroy(stream),
                   "hipStreamDestroy");
#else
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
#endif

    const double eps = 1e-4;

    bool result = true;
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_grads[i]
                      << " calculated score: " << grads[i]
                      << " !(" << lb << " < " << grads[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_score[i] - eps;
        const double ub = expected_score[i] + eps;

        if (!(score[i] > lb && score[i] < ub)) {
            std::cerr << "score mismatch in options_test"
                      << " expected score: " << expected_score[i]
                      << " calculated score: " << score[i] << std::endl;
            result = false;
        }
    }
    return result;
}


bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L);
    labels[0] = 2;
    std::vector<int> label_lengths = {L};

    std::vector<float> acts = genActs(alphabet_size * T * minibatch);

    for (int i = 0; i < T; ++i)
        acts[alphabet_size * i + 2] = -1e30;

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *acts_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> lengths;
    lengths.push_back(T);

    float *grads_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&grads_gpu, (alphabet_size * T) * sizeof(float)),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T) * sizeof(float)),
                   "cudaMalloc");
#endif

    float cost;

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &cost,
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in inf_test");

    bool status = std::isinf(cost);

    std::vector<float> grads(alphabet_size * T);

#ifdef __HIPCC__
    throw_on_error(hipMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync");
    throw_on_error(hipStreamSynchronize(stream), "hipStreamSynchronize");
#else
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
#endif

    for (int i = 0; i < alphabet_size * T; ++i)
        status &= !std::isnan(grads[i]);

#ifdef __HIPCC__
    throw_on_error(hipFree(acts_gpu), "hipFree");
    throw_on_error(hipFree(grads_gpu), "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace), "hipFree");
    throw_on_error(hipStreamDestroy(stream), "hipStreamDestroy");
#else
    throw_on_error(cudaFree(acts_gpu), "cudaFree");
    throw_on_error(cudaFree(grads_gpu), "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace), "cudaFree");
    throw_on_error(cudaStreamDestroy(stream), "cudaStreamDestroy");
#endif

    return status;
}

float grad_check(int T, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& lengths) {

    float epsilon = 1e-2;

    const int minibatch = labels.size();

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *acts_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    float *grads_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&grads_gpu, acts.size() * sizeof(float)),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&grads_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
#endif

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      lengths.data(),
                                      alphabet_size,
                                      lengths.size(),
                                      options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    flat_labels.data(),
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss (0) in grad_check");

    std::vector<float> grads(acts.size());

#ifdef __HIPCC__
    throw_on_error(hipMemcpyAsync(grads.data(),
                                   grads_gpu, grads.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync");
    throw_on_error(hipStreamSynchronize(stream), "hipStreamSynchronize");
#else
    throw_on_error(cudaMemcpyAsync(grads.data(),
                                   grads_gpu, grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
#endif

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    for (int i = 0; i < T * alphabet_size * minibatch; ++i) {
        acts[i] += epsilon;

#ifdef __HIPCC__
        throw_on_error(hipMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       hipMemcpyHostToDevice, stream),
                       "hipMemcpyAsync");
#else
        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");
#endif

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(),
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        ctc_gpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;

#ifdef __HIPCC__
        throw_on_error(hipMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       hipMemcpyHostToDevice, stream),
                       "hipMemcpyAsync");
#else
        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");
#endif

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(),
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        ctc_gpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;

        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }

    float diff = rel_diff(grads, num_grad);

#ifdef __HIPCC__
    throw_on_error(hipFree(acts_gpu),
                   "hipFree");
    throw_on_error(hipFree(grads_gpu),
                   "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace),
                   "hipFree");
    throw_on_error(hipStreamDestroy(stream),
                   "hipStreamDestroy");
#else
    throw_on_error(cudaFree(acts_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(grads_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
#endif

    return diff;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
        { std::make_tuple(28, 50, 15, 1, 1e-5) };

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts = genActs(alphabet_size * T * minibatch);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        float diff = grad_check(T, alphabet_size, acts, labels, sizes);
        status &= (diff < tol);
    }

    return status;
}

bool use_softmax_test() {
    const int alphabet_size = 4;
    const int T = 2;
    const int minibatch = 2;

    std::vector<float> activations = 
        {0.633766f, 0.221185f, 0.0917319f, 0.0129757f,
         0.30176f, 0.28562f, 0.0831517f, 0.0862751f,
                                      
         0.111121f, 0.588392f, 0.278779f, 0.0055756f,
         0.24082f, 0.397533f, 0.0557226f, 0.0546814f};

    std::vector<float> expected_grads = // from tensorflow
        {1.884695f, 0.247554f, 1.096071f, 1.013060f,
         1.352237f, 1.330587f, 0.086707f, 1.090106f,

         1.117530f, 1.801090f, 0.321515f, 1.005591f,
         1.272292f, 0.488149f, 1.057304f, 1.056204f};

    std::vector<double> expected_scores(2);
    auto& a = activations;
    expected_scores[0] = -0.499964f; // from tensorflow
    expected_scores[1] = -0.480685f; // from tensorflow

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *activations_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> labels = {1, 2, 2, 1};
    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {2, 2};
    std::vector<float> costs(minibatch);
    float *grads_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "cudaMalloc");
#endif

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;
    options.use_softmax = false;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");

#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(activations_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    std::vector<float> grads(alphabet_size * T * minibatch);

#ifdef __HIPCC__
    throw_on_error(hipMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync");
    throw_on_error(hipStreamSynchronize(stream), "hipStreamSynchronize");

    throw_on_error(hipFree(activations_gpu),
                   "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace),
                   "hipFree");
    throw_on_error(hipStreamDestroy(stream),
                   "hipStreamDestroy");
#else
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
#endif
    const double eps = 1e-4;

    bool result = true;
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in use_softmax_test"
                    << " expected grad: " << expected_grads[i]
                    << " calculated score: " << grads[i]
                    << " !(" << lb << " < " << grads[i]
                    << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_scores[i] - eps;
        const double ub = expected_scores[i] + eps;

        if (!(costs[i] > lb && costs[i] < ub)) {
            std::cerr << "score mismatch in use_softmax_test"
                    << " expected score: " << expected_scores[i]
                    << " calculated score: " << costs[i] << std::endl;
            result = false;
        }
    }
    return result;
}

bool zero_infinity_test() {
    const int alphabet_size = 4;
    const int T = 2;
    const int minibatch = 2;

    std::vector<float> activations = 
        {0.633766f, 0.221185f, 0.0917319f, 0.0129757f,
         0.30176f, -1e30, 0.0831517f, 0.0862751f,
                                     
         0.111121f, 0.588392f, 0.278779f, 0.0055756f,
         0.24082f, -1e30, 0.0557226f, 0.0546814f};

    std::vector<float> expected_grads = // from tensorflow
        {1.884695f, 0.247554f, 1.096071f, 1.013060f,
         0.0f, 0.0f, 0.0f, 0.0f,

         1.117530f, 1.801090f, 0.321515f, 1.005591f,
         0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<double> expected_scores(2);
    auto& a = activations;
    expected_scores[0] = -0.5f; // from tensorflow
    expected_scores[1] = 0.0f;

#ifdef __HIPCC__
    hipStream_t stream;
    throw_on_error(hipStreamCreate(&stream),
                   "hipStreamCreate");
#else
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");
#endif

    float *activations_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "hipMalloc");
    throw_on_error(hipMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync");
#else
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
#endif

    std::vector<int> labels = {1, 2, 2, 1};
    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {2, 2};
    std::vector<float> costs(minibatch);
    float *grads_gpu;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "hipMalloc");
#else
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "cudaMalloc");
#endif

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;
    options.use_softmax = false;
    options.zero_infinity = true;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    char *ctc_gpu_workspace;

#ifdef __HIPCC__
    throw_on_error(hipMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "hipMalloc");

#else
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
#endif

    throw_on_error(compute_ctc_loss(activations_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    std::vector<float> grads(alphabet_size * T * minibatch);

#ifdef __HIPCC__
    throw_on_error(hipMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync");
    throw_on_error(hipStreamSynchronize(stream), "hipStreamSynchronize");

    throw_on_error(hipFree(activations_gpu),
                   "hipFree");
    throw_on_error(hipFree(ctc_gpu_workspace),
                   "hipFree");
    throw_on_error(hipStreamDestroy(stream),
                   "hipStreamDestroy");
#else
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
#endif
    const double eps = 1e-4;

    bool result = true;
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in zero_infinity_test"
                    << " expected grad: " << expected_grads[i]
                    << " calculated score: " << grads[i]
                    << " !(" << lb << " < " << grads[i]
                    << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_scores[i] - eps;
        const double ub = expected_scores[i] + eps;

        if (!(costs[i] > lb && costs[i] < ub)) {
            std::cerr << "score mismatch in zero_infinity_test"
                    << " expected score: " << expected_scores[i]
                    << " calculated score: " << costs[i] << std::endl;
            result = false;
        }
    }
    return result;
}

int main(void) {
    if (get_warpctc_version() != 2) {
        std::cerr << "Invalid WarpCTC version." << std::endl;
        return 1;
    }

    std::cout << "Running GPU tests" << std::endl;

#ifdef __HIPCC__
    throw_on_error(hipSetDevice(0), "hipSetDevice");
#else
    throw_on_error(cudaSetDevice(0), "cudaSetDevice");
#endif

    bool status = true;
    status &= small_test();
    status &= options_test();
    status &= inf_test();
    status &= run_tests();
    status &= use_softmax_test();
    status &= zero_infinity_test();

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
