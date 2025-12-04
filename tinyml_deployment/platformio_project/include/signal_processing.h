/**
 * Signal Processing for EMG Data
 *
 * Implements filtering and preprocessing for sEMG signals
 * optimized for Cortex-M4F (XIAO nRF52840).
 *
 * Signal Chain:
 * 1. DC offset removal (high-pass @ 20Hz)
 * 2. Anti-aliasing (low-pass @ 450Hz)
 * 3. Powerline notch filter (60Hz)
 * 4. Optional: Envelope extraction
 */

#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include <stdint.h>
#include <stdbool.h>

// Configuration
#ifndef EMG_CHANNELS
#define EMG_CHANNELS 7
#endif

#ifndef EMG_SAMPLE_RATE
#define EMG_SAMPLE_RATE 2000  // Hz
#endif

// Filter order
#define BANDPASS_ORDER 4  // 4th order Butterworth
#define NOTCH_ORDER 2     // 2nd order notch

// Number of biquad sections (order / 2)
#define BANDPASS_SECTIONS (BANDPASS_ORDER / 2)
#define NOTCH_SECTIONS (NOTCH_ORDER / 2)

/**
 * Biquad filter state
 */
typedef struct {
    float w0;  // State variable 0
    float w1;  // State variable 1
} BiquadState;

/**
 * Biquad filter coefficients (Direct Form II)
 *
 * Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
 */
typedef struct {
    float b0, b1, b2;  // Numerator coefficients
    float a1, a2;      // Denominator coefficients (a0 = 1)
} BiquadCoeffs;

/**
 * Signal Processor class
 *
 * Manages per-channel filter states and provides preprocessing.
 */
class SignalProcessor {
public:
    SignalProcessor();
    ~SignalProcessor();

    /**
     * Initialize the signal processor.
     *
     * Sets up filter coefficients and clears state.
     *
     * @return true on success
     */
    bool begin();

    /**
     * Reset all filter states.
     *
     * Call when starting a new recording session.
     */
    void reset();

    /**
     * Process EMG buffer in-place.
     *
     * Applies the full signal processing chain:
     * - Bandpass filter (20-450Hz)
     * - Notch filter (60Hz)
     *
     * @param data Input/output buffer [channels * samples]
     *             Data format: channel-major [ch0_t0, ch0_t1, ..., ch1_t0, ...]
     * @param num_channels Number of EMG channels
     * @param num_samples Number of samples per channel
     */
    void process(float* data, int num_channels, int num_samples);

    /**
     * Process single sample (for real-time streaming).
     *
     * @param samples Input/output array [EMG_CHANNELS]
     */
    void processSample(float* samples);

    /**
     * Enable/disable 60Hz notch filter.
     */
    void enableNotchFilter(bool enable) { notch_enabled_ = enable; }

    /**
     * Get DC offset estimate (for debugging).
     */
    float getDcOffset(int channel) const;

private:
    bool initialized_;
    bool notch_enabled_;

    // Filter coefficients
    BiquadCoeffs bandpass_coeffs_[BANDPASS_SECTIONS * 2];  // High-pass + low-pass
    BiquadCoeffs notch_coeffs_[NOTCH_SECTIONS];

    // Per-channel filter states
    BiquadState bandpass_state_[EMG_CHANNELS][BANDPASS_SECTIONS * 2];
    BiquadState notch_state_[EMG_CHANNELS][NOTCH_SECTIONS];

    // DC offset tracking
    float dc_offset_[EMG_CHANNELS];

    /**
     * Initialize bandpass filter coefficients.
     *
     * Butterworth bandpass 20-450Hz at 2kHz sample rate.
     */
    void initBandpassCoeffs();

    /**
     * Initialize notch filter coefficients.
     *
     * Notch at 60Hz with Q=30.
     */
    void initNotchCoeffs();

    /**
     * Apply biquad filter section.
     *
     * @param sample Input sample
     * @param coeffs Filter coefficients
     * @param state Filter state (modified)
     * @return Filtered sample
     */
    float applyBiquad(float sample, const BiquadCoeffs* coeffs, BiquadState* state);
};

// ============================================================================
// Standalone functions for use without class wrapper
// ============================================================================

/**
 * Apply Reinhard compression to buffer.
 *
 * output = range * x / (midpoint + |x|)
 *
 * @param data Input/output buffer
 * @param size Number of elements
 * @param range Compression range (default 1.0)
 * @param midpoint Midpoint value (default 32.0)
 */
void applyReinhardCompression(float* data, int size, float range = 1.0f, float midpoint = 32.0f);

/**
 * Compute RMS of buffer.
 *
 * @param data Input buffer
 * @param size Number of elements
 * @return RMS value
 */
float computeRMS(const float* data, int size);

/**
 * Compute Mean Absolute Value.
 *
 * @param data Input buffer
 * @param size Number of elements
 * @return MAV
 */
float computeMAV(const float* data, int size);

/**
 * Apply simple moving average filter.
 *
 * @param data Input/output buffer
 * @param size Number of elements
 * @param window_size Moving average window
 */
void applyMovingAverage(float* data, int size, int window_size);

#endif // SIGNAL_PROCESSING_H
