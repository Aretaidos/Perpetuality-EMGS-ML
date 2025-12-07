/**
 * M1 4-Channel Native Inference Engine Implementation
 *
 * Native C++ implementation optimized for ARM Cortex-M4F.
 * Uses hardware FPU for fast floating-point operations.
 *
 * Memory Layout:
 *   - Weights stored as INT8 in flash (from m1_4channel_weights.h)
 *   - Dequantized on-the-fly during inference
 *   - Working buffers allocated on heap
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 */

// Include weights first so its macros are defined before m1_inference.h
#include "m1_4channel_weights.h"
#include "m1_inference.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <cstdlib>

// ============================================================================
// Constructor / Destructor
// ============================================================================

M1Inference::M1Inference()
    : initialized_(false)
    , last_latency_ms_(0)
    , conv_output_(nullptr)
    , lstm_input_(nullptr)
    , lstm_output_(nullptr)
    , fc_output_(nullptr)
{
    for (int i = 0; i < M1_LSTM_NUM_LAYERS; i++) {
        lstm_hidden_[i] = nullptr;
        lstm_cell_[i] = nullptr;
    }
}

M1Inference::~M1Inference() {
    // Free allocated buffers
    if (conv_output_) free(conv_output_);
    if (lstm_input_) free(lstm_input_);
    if (lstm_output_) free(lstm_output_);
    if (fc_output_) free(fc_output_);

    for (int i = 0; i < M1_LSTM_NUM_LAYERS; i++) {
        if (lstm_hidden_[i]) free(lstm_hidden_[i]);
        if (lstm_cell_[i]) free(lstm_cell_[i]);
    }
}

// ============================================================================
// Initialization
// ============================================================================

bool M1Inference::begin() {
    Serial.println("[M1] Initializing native inference engine...");

    // Calculate buffer sizes
    size_t conv_size = M1_CONV_OUT_CHANNELS * M1_CONV_OUT_TIMESTEPS * sizeof(float);
    size_t lstm_input_size = M1_CONV_OUT_TIMESTEPS * M1_LSTM_INPUT_SIZE * sizeof(float);
    size_t lstm_output_size = M1_CONV_OUT_TIMESTEPS * M1_LSTM_HIDDEN_SIZE * sizeof(float);
    size_t fc_size = M1_CONV_OUT_TIMESTEPS * M1_OUTPUT_CLASSES * sizeof(float);
    size_t hidden_size = M1_LSTM_HIDDEN_SIZE * sizeof(float);

    Serial.printf("[M1] Buffer sizes:\n");
    Serial.printf("  Conv output: %d bytes\n", conv_size);
    Serial.printf("  LSTM input:  %d bytes\n", lstm_input_size);
    Serial.printf("  LSTM output: %d bytes\n", lstm_output_size);
    Serial.printf("  FC output:   %d bytes\n", fc_size);
    Serial.printf("  LSTM states: %d bytes x 4\n", hidden_size);

    // Allocate buffers
    conv_output_ = (float*)malloc(conv_size);
    lstm_input_ = (float*)malloc(lstm_input_size);
    lstm_output_ = (float*)malloc(lstm_output_size);
    fc_output_ = (float*)malloc(fc_size);

    for (int i = 0; i < M1_LSTM_NUM_LAYERS; i++) {
        lstm_hidden_[i] = (float*)malloc(hidden_size);
        lstm_cell_[i] = (float*)malloc(hidden_size);
    }

    // Check allocations
    if (!conv_output_ || !lstm_input_ || !lstm_output_ || !fc_output_) {
        Serial.println("[M1] ERROR: Failed to allocate main buffers!");
        return false;
    }

    for (int i = 0; i < M1_LSTM_NUM_LAYERS; i++) {
        if (!lstm_hidden_[i] || !lstm_cell_[i]) {
            Serial.println("[M1] ERROR: Failed to allocate LSTM state buffers!");
            return false;
        }
    }

    size_t total_ram = conv_size + lstm_input_size + lstm_output_size + fc_size +
                       4 * hidden_size;
    Serial.printf("[M1] Total RAM allocated: %d bytes (%.1f KB)\n", total_ram, total_ram / 1024.0f);

    initialized_ = true;
    Serial.println("[M1] Initialization complete!");
    return true;
}

// ============================================================================
// Main Inference
// ============================================================================

int M1Inference::classify(const float* emg_data, float* confidences) {
    if (!initialized_) {
        return -1;
    }

    uint32_t start_time = millis();

    // Reset LSTM states to zero
    for (int i = 0; i < M1_LSTM_NUM_LAYERS; i++) {
        memset(lstm_hidden_[i], 0, M1_LSTM_HIDDEN_SIZE * sizeof(float));
        memset(lstm_cell_[i], 0, M1_LSTM_HIDDEN_SIZE * sizeof(float));
    }

    // 1. Apply Reinhard compression to input
    // We'll apply it during Conv1D to save memory

    // 2. Conv1D + BatchNorm + ReLU
    conv1dForward(emg_data, conv_output_);
    batchNormForward(conv_output_, M1_CONV_OUT_CHANNELS, M1_CONV_OUT_TIMESTEPS);
    reluForward(conv_output_, M1_CONV_OUT_CHANNELS * M1_CONV_OUT_TIMESTEPS);

    // 3. Transpose for LSTM: [C x T] -> [T x C]
    transposeForLSTM(conv_output_, lstm_input_, M1_CONV_OUT_CHANNELS, M1_CONV_OUT_TIMESTEPS);

    // 4. LSTM (2 layers)
    lstmForward(lstm_input_, lstm_output_, M1_CONV_OUT_TIMESTEPS);

    // 5. LayerNorm
    layerNormForward(lstm_output_, M1_CONV_OUT_TIMESTEPS, M1_LSTM_HIDDEN_SIZE);

    // 6. Fully-connected layer
    fcForward(lstm_output_, fc_output_, M1_CONV_OUT_TIMESTEPS);

    // 7. Temporal averaging
    temporalAverage(fc_output_, confidences, M1_CONV_OUT_TIMESTEPS, M1_OUTPUT_CLASSES);

    // 8. Softmax
    softmax(confidences, M1_OUTPUT_CLASSES);

    // Find argmax
    int best_class = 0;
    float best_conf = confidences[0];
    for (int i = 1; i < M1_OUTPUT_CLASSES; i++) {
        if (confidences[i] > best_conf) {
            best_conf = confidences[i];
            best_class = i;
        }
    }

    last_latency_ms_ = millis() - start_time;
    return best_class;
}

const char* M1Inference::getGestureName(int class_idx) const {
    if (class_idx >= 0 && class_idx < M1_OUTPUT_CLASSES) {
        return M1_GESTURE_NAMES[class_idx];
    }
    return "unknown";
}

// ============================================================================
// Reinhard Compression
// ============================================================================

void M1Inference::applyReinhardCompression(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        float abs_x = fabsf(x);
        output[i] = M1_REINHARD_RANGE * x / (M1_REINHARD_MIDPOINT + abs_x);
    }
}

// ============================================================================
// Conv1D Forward
// ============================================================================

void M1Inference::conv1dForward(const float* input, float* output) {
    // Input: [4 x 2000] channel-major
    // Output: [72 x 199] channel-major
    // Kernel: [72 x 4 x 15]

    const int in_channels = M1_INPUT_CHANNELS;
    const int out_channels = M1_CONV_OUT_CHANNELS;
    const int kernel_size = M1_CONV_KERNEL_SIZE;
    const int stride = M1_CONV_STRIDE;
    const int in_samples = M1_INPUT_SAMPLES;
    const int out_samples = M1_CONV_OUT_TIMESTEPS;

    // For each output channel
    for (int oc = 0; oc < out_channels; oc++) {
        // Get bias (dequantize)
        float bias = dequantize(conv_bias[oc], conv_bias_scale);

        // For each output timestep
        for (int t = 0; t < out_samples; t++) {
            float sum = bias;

            // Convolution over input channels and kernel
            int start_pos = t * stride;

            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int input_pos = start_pos + k;
                    if (input_pos >= 0 && input_pos < in_samples) {
                        // Input is channel-major: input[ic * in_samples + input_pos]
                        float x = input[ic * in_samples + input_pos];

                        // Apply Reinhard compression inline
                        float abs_x = fabsf(x);
                        float compressed_x = M1_REINHARD_RANGE * x / (M1_REINHARD_MIDPOINT + abs_x);

                        // Weight index: [oc, ic, k] in row-major
                        int weight_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                        float w = dequantize(conv_weight[weight_idx], conv_weight_scale);

                        sum += compressed_x * w;
                    }
                }
            }

            // Output is channel-major: output[oc * out_samples + t]
            output[oc * out_samples + t] = sum;
        }
    }
}

// ============================================================================
// BatchNorm Forward
// ============================================================================

void M1Inference::batchNormForward(float* data, int channels, int timesteps) {
    // BatchNorm inference: y = gamma * (x - mean) / sqrt(var + eps) + beta

    for (int c = 0; c < channels; c++) {
        float gamma = dequantize(bn_gamma[c], bn_gamma_scale);
        float beta = dequantize(bn_beta[c], bn_beta_scale);
        float mean = dequantize(bn_mean[c], bn_mean_scale);
        float var = dequantize(bn_var[c], bn_var_scale);

        // Precompute scale factor
        float inv_std = 1.0f / sqrtf(var + M1_BN_EPSILON);
        float scale = gamma * inv_std;
        float shift = beta - mean * scale;

        // Apply to all timesteps for this channel
        for (int t = 0; t < timesteps; t++) {
            int idx = c * timesteps + t;
            data[idx] = data[idx] * scale + shift;
        }
    }
}

// ============================================================================
// ReLU Forward
// ============================================================================

void M1Inference::reluForward(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// ============================================================================
// Transpose for LSTM
// ============================================================================

void M1Inference::transposeForLSTM(const float* input, float* output, int channels, int timesteps) {
    // Input: [C x T] channel-major
    // Output: [T x C] timestep-major

    for (int t = 0; t < timesteps; t++) {
        for (int c = 0; c < channels; c++) {
            output[t * channels + c] = input[c * timesteps + t];
        }
    }
}

// ============================================================================
// LSTM Forward
// ============================================================================

void M1Inference::lstmForward(const float* input, float* output, int timesteps) {
    // Layer 0: input_size=72, hidden_size=48
    // Layer 1: input_size=48, hidden_size=48

    // Temporary buffer for intermediate layer output
    float* temp_output = (float*)malloc(timesteps * M1_LSTM_HIDDEN_SIZE * sizeof(float));
    if (!temp_output) {
        Serial.println("[M1] ERROR: LSTM temp buffer allocation failed!");
        return;
    }

    // Layer 0
    lstmLayerForward(
        input,
        temp_output,
        M1_LSTM_INPUT_SIZE,    // 72
        M1_LSTM_HIDDEN_SIZE,   // 48
        timesteps,
        0,
        lstm_hidden_[0],
        lstm_cell_[0]
    );

    // Layer 1
    lstmLayerForward(
        temp_output,
        output,
        M1_LSTM_HIDDEN_SIZE,   // 48
        M1_LSTM_HIDDEN_SIZE,   // 48
        timesteps,
        1,
        lstm_hidden_[1],
        lstm_cell_[1]
    );

    free(temp_output);
}

void M1Inference::lstmLayerForward(
    const float* input,
    float* output,
    int input_size,
    int hidden_size,
    int timesteps,
    int layer_idx,
    float* hidden,
    float* cell
) {
    // Process each timestep sequentially
    for (int t = 0; t < timesteps; t++) {
        const float* x = input + t * input_size;
        float* h_out = output + t * hidden_size;

        lstmCell(x, hidden, cell, h_out, cell, input_size, hidden_size, layer_idx);

        // Update hidden state for next timestep
        memcpy(hidden, h_out, hidden_size * sizeof(float));
    }
}

void M1Inference::lstmCell(
    const float* x,
    const float* h_prev,
    const float* c_prev,
    float* h_next,
    float* c_next,
    int input_size,
    int hidden_size,
    int layer_idx
) {
    // LSTM cell: computes i, f, g, o gates
    // Gates are stored concatenated: [i, f, g, o] (4 * hidden_size)

    // Get weight pointers based on layer
    const int8_t* weight_ih;
    const int8_t* weight_hh;
    const int8_t* bias_ih;
    const int8_t* bias_hh;
    float weight_ih_scale, weight_hh_scale, bias_ih_scale, bias_hh_scale;

    if (layer_idx == 0) {
        weight_ih = lstm0_weight_ih;
        weight_hh = lstm0_weight_hh;
        bias_ih = lstm0_bias_ih;
        bias_hh = lstm0_bias_hh;
        weight_ih_scale = lstm0_weight_ih_scale;
        weight_hh_scale = lstm0_weight_hh_scale;
        bias_ih_scale = lstm0_bias_ih_scale;
        bias_hh_scale = lstm0_bias_hh_scale;
    } else {
        weight_ih = lstm1_weight_ih;
        weight_hh = lstm1_weight_hh;
        bias_ih = lstm1_bias_ih;
        bias_hh = lstm1_bias_hh;
        weight_ih_scale = lstm1_weight_ih_scale;
        weight_hh_scale = lstm1_weight_hh_scale;
        bias_ih_scale = lstm1_bias_ih_scale;
        bias_hh_scale = lstm1_bias_hh_scale;
    }

    // Temporary storage for gates
    float gates[4 * 48];  // 4 gates * hidden_size

    // Compute gates = W_ih @ x + b_ih + W_hh @ h + b_hh
    // W_ih: [4*hidden x input], W_hh: [4*hidden x hidden]

    for (int g = 0; g < 4 * hidden_size; g++) {
        float sum = dequantize(bias_ih[g], bias_ih_scale) +
                    dequantize(bias_hh[g], bias_hh_scale);

        // W_ih @ x
        for (int i = 0; i < input_size; i++) {
            int idx = g * input_size + i;
            sum += dequantize(weight_ih[idx], weight_ih_scale) * x[i];
        }

        // W_hh @ h
        for (int h = 0; h < hidden_size; h++) {
            int idx = g * hidden_size + h;
            sum += dequantize(weight_hh[idx], weight_hh_scale) * h_prev[h];
        }

        gates[g] = sum;
    }

    // Apply activations and compute new cell/hidden states
    // gates layout: [i(0..H), f(H..2H), g(2H..3H), o(3H..4H)]
    for (int h = 0; h < hidden_size; h++) {
        float i_gate = sigmoid(gates[h]);
        float f_gate = sigmoid(gates[hidden_size + h]);
        float g_gate = tanh_activation(gates[2 * hidden_size + h]);
        float o_gate = sigmoid(gates[3 * hidden_size + h]);

        // New cell state: c_t = f * c_{t-1} + i * g
        float c_new = f_gate * c_prev[h] + i_gate * g_gate;
        c_next[h] = c_new;

        // New hidden state: h_t = o * tanh(c_t)
        h_next[h] = o_gate * tanh_activation(c_new);
    }
}

// ============================================================================
// LayerNorm Forward
// ============================================================================

void M1Inference::layerNormForward(float* data, int timesteps, int features) {
    // LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
    // Normalize over features for each timestep

    for (int t = 0; t < timesteps; t++) {
        float* row = data + t * features;

        // Compute mean
        float mean = 0.0f;
        for (int f = 0; f < features; f++) {
            mean += row[f];
        }
        mean /= features;

        // Compute variance
        float var = 0.0f;
        for (int f = 0; f < features; f++) {
            float diff = row[f] - mean;
            var += diff * diff;
        }
        var /= features;

        // Normalize and apply scale/shift
        float inv_std = 1.0f / sqrtf(var + M1_LN_EPSILON);
        for (int f = 0; f < features; f++) {
            float gamma = dequantize(ln_weight[f], ln_weight_scale);
            float beta = dequantize(ln_bias[f], ln_bias_scale);
            row[f] = gamma * (row[f] - mean) * inv_std + beta;
        }
    }
}

// ============================================================================
// FC Forward
// ============================================================================

void M1Inference::fcForward(const float* input, float* output, int timesteps) {
    // Linear: output = input @ weight.T + bias
    // weight: [9 x 48], input: [T x 48], output: [T x 9]

    for (int t = 0; t < timesteps; t++) {
        const float* x = input + t * M1_LSTM_HIDDEN_SIZE;
        float* y = output + t * M1_OUTPUT_CLASSES;

        for (int o = 0; o < M1_OUTPUT_CLASSES; o++) {
            float sum = dequantize(fc_bias[o], fc_bias_scale);

            for (int i = 0; i < M1_LSTM_HIDDEN_SIZE; i++) {
                int idx = o * M1_LSTM_HIDDEN_SIZE + i;
                sum += dequantize(fc_weight[idx], fc_weight_scale) * x[i];
            }

            y[o] = sum;
        }
    }
}

// ============================================================================
// Temporal Averaging
// ============================================================================

void M1Inference::temporalAverage(const float* input, float* output, int timesteps, int classes) {
    // Average over timesteps: output[c] = mean(input[:, c])

    for (int c = 0; c < classes; c++) {
        float sum = 0.0f;
        for (int t = 0; t < timesteps; t++) {
            sum += input[t * classes + c];
        }
        output[c] = sum / timesteps;
    }
}

// ============================================================================
// Softmax
// ============================================================================

void M1Inference::softmax(float* data, int size) {
    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

// ============================================================================
// Memory Info
// ============================================================================

void M1Inference::printMemoryInfo() {
    Serial.println("\n=== M1 Native Inference Memory ===");

    size_t conv_size = M1_CONV_OUT_CHANNELS * M1_CONV_OUT_TIMESTEPS * sizeof(float);
    size_t lstm_input_size = M1_CONV_OUT_TIMESTEPS * M1_LSTM_INPUT_SIZE * sizeof(float);
    size_t lstm_output_size = M1_CONV_OUT_TIMESTEPS * M1_LSTM_HIDDEN_SIZE * sizeof(float);
    size_t fc_size = M1_CONV_OUT_TIMESTEPS * M1_OUTPUT_CLASSES * sizeof(float);
    size_t hidden_size = M1_LSTM_HIDDEN_SIZE * sizeof(float) * 4;

    Serial.printf("  Conv buffer:   %5d bytes\n", conv_size);
    Serial.printf("  LSTM input:    %5d bytes\n", lstm_input_size);
    Serial.printf("  LSTM output:   %5d bytes\n", lstm_output_size);
    Serial.printf("  LSTM states:   %5d bytes\n", hidden_size);
    Serial.printf("  FC output:     %5d bytes\n", fc_size);

    size_t total = conv_size + lstm_input_size + lstm_output_size + fc_size + hidden_size;
    Serial.printf("  -------------------------\n");
    Serial.printf("  Total working: %5d bytes (%.1f KB)\n", total, total / 1024.0f);

    // Weights in flash (approximate from INT8 weights)
    Serial.printf("\n  Weights (INT8 in flash): ~47 KB\n");
}
