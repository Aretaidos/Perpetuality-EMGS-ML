/**
 * M1 4-Channel Native Inference Engine for XIAO nRF52840
 *
 * Native C++ implementation of M1 4-Channel TinyML model inference.
 * Optimized for ARM Cortex-M4F with hardware FPU.
 *
 * Architecture:
 *   Conv1d(4->72, k=15, s=10) -> BatchNorm -> ReLU
 *   -> LSTM(72->48, 2 layers) -> LayerNorm -> FC(48->9)
 *
 * Input:  4 channels x 2000 samples (1.0s @ 2kHz)
 * Output: 9 gesture classes x 199 timesteps
 *
 * Memory Requirements:
 *   - Weights (INT8): ~47KB (in flash)
 *   - Input buffer: 32KB
 *   - Conv output: 57KB
 *   - LSTM states: 1.5KB
 *   - Total RAM: ~92KB (fits in 220KB usable)
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * MCU: nRF52840 (ARM Cortex-M4F @ 64MHz)
 */

#ifndef M1_INFERENCE_H
#define M1_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Model Configuration (with guards to avoid redefinition from weights.h)
// ============================================================================

// Input configuration
#ifndef M1_INPUT_CHANNELS
#define M1_INPUT_CHANNELS    4
#endif

#ifndef M1_INPUT_SAMPLES
#define M1_INPUT_SAMPLES     2000    // 1.0s @ 2kHz
#endif

// Conv1D configuration
#ifndef M1_CONV_OUT_CHANNELS
#define M1_CONV_OUT_CHANNELS 72
#endif

#ifndef M1_CONV_KERNEL_SIZE
#define M1_CONV_KERNEL_SIZE  15
#endif

#ifndef M1_CONV_STRIDE
#define M1_CONV_STRIDE       10
#endif

#ifndef M1_CONV_OUT_TIMESTEPS
#define M1_CONV_OUT_TIMESTEPS ((M1_INPUT_SAMPLES - M1_CONV_KERNEL_SIZE) / M1_CONV_STRIDE + 1)  // 199
#endif

// LSTM configuration
#ifndef M1_LSTM_INPUT_SIZE
#define M1_LSTM_INPUT_SIZE   72      // Same as conv output channels
#endif

#ifndef M1_LSTM_HIDDEN_SIZE
#define M1_LSTM_HIDDEN_SIZE  48
#endif

#ifndef M1_LSTM_NUM_LAYERS
#define M1_LSTM_NUM_LAYERS   2
#endif

// Output configuration
#ifndef M1_OUTPUT_CLASSES
#define M1_OUTPUT_CLASSES    9
#endif

#ifndef M1_OUTPUT_TIMESTEPS
#define M1_OUTPUT_TIMESTEPS  M1_CONV_OUT_TIMESTEPS  // 199
#endif

// Reinhard compression parameters
#ifndef M1_REINHARD_RANGE
#define M1_REINHARD_RANGE    1.0f
#endif

#ifndef M1_REINHARD_MIDPOINT
#define M1_REINHARD_MIDPOINT 32.0f
#endif

// Batch normalization epsilon
#define M1_BN_EPSILON        1e-5f

// Layer normalization epsilon
#define M1_LN_EPSILON        1e-5f

// ============================================================================
// Gesture Class Names
// ============================================================================

static const char* M1_GESTURE_NAMES[] = {
    "index_press",
    "index_release",
    "middle_press",
    "middle_release",
    "thumb_click",
    "thumb_down",
    "thumb_in",
    "thumb_out",
    "thumb_up"
};

// ============================================================================
// M1Inference Class
// ============================================================================

/**
 * M1Inference - Native inference engine for M1 4-Channel TinyML model.
 *
 * This class implements all layers of the M1 model using native C++ code,
 * optimized for Cortex-M4F with hardware FPU.
 *
 * Usage:
 *   M1Inference inference;
 *   inference.begin();
 *   int gesture = inference.classify(emg_data, confidences);
 */
class M1Inference {
public:
    M1Inference();
    ~M1Inference();

    /**
     * Initialize the inference engine.
     * Allocates working buffers and loads weights.
     *
     * @return true if initialization successful
     */
    bool begin();

    /**
     * Run gesture classification on EMG window.
     *
     * @param emg_data Input EMG data [M1_INPUT_CHANNELS * M1_INPUT_SAMPLES]
     *                 Data in channel-major format: [ch0_t0...ch0_tN, ch1_t0...]
     * @param confidences Output confidence scores [M1_OUTPUT_CLASSES]
     * @return Predicted gesture class (0-8), or -1 on error
     */
    int classify(const float* emg_data, float* confidences);

    /**
     * Get the predicted gesture name.
     *
     * @param class_idx Gesture class index (0-8)
     * @return Gesture name string
     */
    const char* getGestureName(int class_idx) const;

    /**
     * Get last inference latency in milliseconds.
     */
    uint32_t getLastLatency() const { return last_latency_ms_; }

    /**
     * Check if engine is ready.
     */
    bool isReady() const { return initialized_; }

    /**
     * Print memory usage information.
     */
    void printMemoryInfo();

private:
    bool initialized_;
    uint32_t last_latency_ms_;

    // Working buffers (allocated on heap to save stack)
    float* conv_output_;          // [72 x 199]
    float* lstm_input_;           // [199 x 72] (transposed)
    float* lstm_output_;          // [199 x 48]
    float* lstm_hidden_[2];       // [48] for each layer
    float* lstm_cell_[2];         // [48] for each layer
    float* fc_output_;            // [199 x 9]

    // ========================================================================
    // Layer Forward Passes
    // ========================================================================

    /**
     * Apply Reinhard compression to normalize EMG input.
     * output = range * x / (midpoint + |x|)
     */
    void applyReinhardCompression(const float* input, float* output, int size);

    /**
     * Conv1D forward pass with padding.
     * Input: [4 x 2000], Output: [72 x 199]
     */
    void conv1dForward(const float* input, float* output);

    /**
     * BatchNorm forward pass (inference mode).
     * Uses running mean/var, applies gamma/beta.
     */
    void batchNormForward(float* data, int channels, int timesteps);

    /**
     * ReLU activation (in-place).
     */
    void reluForward(float* data, int size);

    /**
     * Transpose [C x T] to [T x C] for LSTM input.
     */
    void transposeForLSTM(const float* input, float* output, int channels, int timesteps);

    /**
     * LSTM forward pass (2 layers).
     * Input: [T x 72], Output: [T x 48]
     */
    void lstmForward(const float* input, float* output, int timesteps);

    /**
     * Single LSTM layer forward pass.
     */
    void lstmLayerForward(
        const float* input,
        float* output,
        int input_size,
        int hidden_size,
        int timesteps,
        int layer_idx,
        float* hidden,
        float* cell
    );

    /**
     * LSTM cell computation for single timestep.
     * Computes: i, f, g, o gates and updates h, c.
     */
    void lstmCell(
        const float* x,           // Input [input_size]
        const float* h_prev,      // Previous hidden [hidden_size]
        const float* c_prev,      // Previous cell [hidden_size]
        float* h_next,            // Next hidden [hidden_size]
        float* c_next,            // Next cell [hidden_size]
        int input_size,
        int hidden_size,
        int layer_idx
    );

    /**
     * LayerNorm forward pass.
     * Normalizes over hidden dimension at each timestep.
     */
    void layerNormForward(float* data, int timesteps, int features);

    /**
     * Fully-connected (linear) layer.
     * Input: [T x 48], Output: [T x 9]
     */
    void fcForward(const float* input, float* output, int timesteps);

    /**
     * Temporal averaging over timesteps.
     * Input: [T x 9], Output: [9]
     */
    void temporalAverage(const float* input, float* output, int timesteps, int classes);

    /**
     * Softmax activation.
     * Converts logits to probabilities.
     */
    void softmax(float* data, int size);

    // ========================================================================
    // Weight Dequantization Helpers
    // ========================================================================

    /**
     * Dequantize INT8 weight to float.
     * output = input * scale
     */
    inline float dequantize(int8_t value, float scale) const {
        return static_cast<float>(value) * scale;
    }

    /**
     * Sigmoid activation.
     */
    inline float sigmoid(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }

    /**
     * Tanh activation.
     */
    inline float tanh_activation(float x) const {
        return tanhf(x);
    }
};

#endif // M1_INFERENCE_H
