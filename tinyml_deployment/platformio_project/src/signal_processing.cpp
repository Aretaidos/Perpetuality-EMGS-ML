/**
 * Signal Processing Implementation
 *
 * EMG signal preprocessing for XIAO nRF52840.
 * Optimized for Cortex-M4F with FPU.
 */

#include "signal_processing.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>

// ============================================================================
// Pre-computed filter coefficients
// Computed using scipy.signal for 2kHz sample rate
// ============================================================================

// Bandpass filter: 20-450Hz, 4th order Butterworth
// Implemented as cascade of highpass (20Hz) and lowpass (450Hz)
// Each 2nd order section as biquad

// High-pass 20Hz coefficients (2 sections)
static const BiquadCoeffs HP_COEFFS[2] = {
    // Section 1
    {
        .b0 = 0.9565436765112076f,
        .b1 = -1.9130873530224153f,
        .b2 = 0.9565436765112076f,
        .a1 = -1.9111970674260020f,
        .a2 = 0.9149758386188286f
    },
    // Section 2
    {
        .b0 = 0.9565436765112076f,
        .b1 = -1.9130873530224153f,
        .b2 = 0.9565436765112076f,
        .a1 = -1.9149776386188279f,
        .a2 = 0.9111988674260027f
    }
};

// Low-pass 450Hz coefficients (2 sections)
static const BiquadCoeffs LP_COEFFS[2] = {
    // Section 1
    {
        .b0 = 0.2065720808555728f,
        .b1 = 0.4131441617111456f,
        .b2 = 0.2065720808555728f,
        .a1 = -0.3695259753654073f,
        .a2 = 0.1958142987876985f
    },
    // Section 2
    {
        .b0 = 0.2065720808555728f,
        .b1 = 0.4131441617111456f,
        .b2 = 0.2065720808555728f,
        .a1 = -0.1958125987876978f,
        .a2 = 0.3695276753654080f
    }
};

// Notch filter: 60Hz, Q=30
static const BiquadCoeffs NOTCH_COEFFS[1] = {
    {
        .b0 = 0.9906218464034014f,
        .b1 = -1.8117681178316885f,
        .b2 = 0.9906218464034014f,
        .a1 = -1.8117681178316885f,
        .a2 = 0.9812436928068028f
    }
};

// ============================================================================
// SignalProcessor Implementation
// ============================================================================

SignalProcessor::SignalProcessor()
    : initialized_(false)
    , notch_enabled_(true)
{
    memset(bandpass_state_, 0, sizeof(bandpass_state_));
    memset(notch_state_, 0, sizeof(notch_state_));
    memset(dc_offset_, 0, sizeof(dc_offset_));
}

SignalProcessor::~SignalProcessor() {
    // Nothing to clean up
}

bool SignalProcessor::begin() {
    initBandpassCoeffs();
    initNotchCoeffs();
    reset();
    initialized_ = true;
    return true;
}

void SignalProcessor::reset() {
    memset(bandpass_state_, 0, sizeof(bandpass_state_));
    memset(notch_state_, 0, sizeof(notch_state_));
    memset(dc_offset_, 0, sizeof(dc_offset_));
}

void SignalProcessor::initBandpassCoeffs() {
    // Copy pre-computed coefficients
    // Sections 0-1: High-pass (20Hz)
    memcpy(&bandpass_coeffs_[0], HP_COEFFS, sizeof(HP_COEFFS));
    // Sections 2-3: Low-pass (450Hz)
    memcpy(&bandpass_coeffs_[2], LP_COEFFS, sizeof(LP_COEFFS));
}

void SignalProcessor::initNotchCoeffs() {
    memcpy(notch_coeffs_, NOTCH_COEFFS, sizeof(NOTCH_COEFFS));
}

float SignalProcessor::applyBiquad(float sample, const BiquadCoeffs* coeffs, BiquadState* state) {
    // Direct Form II Transposed implementation
    // y[n] = b0*x[n] + w0
    // w0 = b1*x[n] - a1*y[n] + w1
    // w1 = b2*x[n] - a2*y[n]

    float y = coeffs->b0 * sample + state->w0;
    state->w0 = coeffs->b1 * sample - coeffs->a1 * y + state->w1;
    state->w1 = coeffs->b2 * sample - coeffs->a2 * y;

    return y;
}

void SignalProcessor::processSample(float* samples) {
    for (int ch = 0; ch < EMG_CHANNELS; ch++) {
        float x = samples[ch];

        // Apply bandpass filter (4 sections: 2 HP + 2 LP)
        for (int s = 0; s < BANDPASS_SECTIONS * 2; s++) {
            x = applyBiquad(x, &bandpass_coeffs_[s], &bandpass_state_[ch][s]);
        }

        // Apply notch filter
        if (notch_enabled_) {
            for (int s = 0; s < NOTCH_SECTIONS; s++) {
                x = applyBiquad(x, &notch_coeffs_[s], &notch_state_[ch][s]);
            }
        }

        samples[ch] = x;
    }
}

void SignalProcessor::process(float* data, int num_channels, int num_samples) {
    if (!initialized_) {
        return;
    }

    // Process each sample
    // Data format: channel-major [ch0_t0, ch0_t1, ..., ch1_t0, ...]
    for (int t = 0; t < num_samples; t++) {
        float samples[EMG_CHANNELS];

        // Extract samples for this timestep
        for (int ch = 0; ch < num_channels && ch < EMG_CHANNELS; ch++) {
            samples[ch] = data[ch * num_samples + t];
        }

        // Process
        processSample(samples);

        // Write back
        for (int ch = 0; ch < num_channels && ch < EMG_CHANNELS; ch++) {
            data[ch * num_samples + t] = samples[ch];
        }
    }
}

float SignalProcessor::getDcOffset(int channel) const {
    if (channel >= 0 && channel < EMG_CHANNELS) {
        return dc_offset_[channel];
    }
    return 0.0f;
}

// ============================================================================
// Standalone Functions
// ============================================================================

void applyReinhardCompression(float* data, int size, float range, float midpoint) {
    // Vectorized Reinhard compression
    // output = range * x / (midpoint + |x|)
    //
    // This is the same normalization used in the PyTorch model:
    // class ReinhardCompression(nn.Module):
    //     def forward(self, x):
    //         return self.range_val * x / (self.midpoint + torch.abs(x))

    for (int i = 0; i < size; i++) {
        float x = data[i];
        float abs_x = fabsf(x);
        data[i] = range * x / (midpoint + abs_x);
    }
}

float computeRMS(const float* data, int size) {
    if (size <= 0) return 0.0f;

    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += data[i] * data[i];
    }
    return sqrtf(sum_sq / size);
}

float computeMAV(const float* data, int size) {
    if (size <= 0) return 0.0f;

    float sum_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_abs += fabsf(data[i]);
    }
    return sum_abs / size;
}

void applyMovingAverage(float* data, int size, int window_size) {
    if (size <= 0 || window_size <= 1) return;

    // In-place moving average using running sum
    float* temp = new float[size];
    if (!temp) return;

    memcpy(temp, data, size * sizeof(float));

    int half_window = window_size / 2;

    for (int i = 0; i < size; i++) {
        float sum = 0.0f;
        int count = 0;

        int start = max(0, i - half_window);
        int end = min(size - 1, i + half_window);

        for (int j = start; j <= end; j++) {
            sum += temp[j];
            count++;
        }

        data[i] = sum / count;
    }

    delete[] temp;
}
