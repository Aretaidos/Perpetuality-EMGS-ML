/**
 * EMG Acquisition Interface
 *
 * Handles EMG data acquisition for XIAO nRF52840.
 * Supports both simulated data (for testing) and real ADC input.
 *
 * For real EMG hardware, you'll typically use:
 * - External ADC via SPI (e.g., ADS1299 for research-grade)
 * - External ADC via I2C (e.g., ADS1115 for hobby-grade)
 * - Direct analog pins (limited to fewer channels, lower quality)
 */

#ifndef EMG_ACQUISITION_H
#define EMG_ACQUISITION_H

#include <stdint.h>
#include <stdbool.h>

// Configuration
#ifndef EMG_CHANNELS
#define EMG_CHANNELS 7
#endif

#ifndef EMG_SAMPLE_RATE
#define EMG_SAMPLE_RATE 2000  // Hz
#endif

#ifndef WINDOW_SAMPLES
#define WINDOW_SAMPLES 2000  // Samples per channel per window
#endif

// Acquisition mode
enum EmgAcquisitionMode {
    EMG_MODE_SIMULATED,   // Generate synthetic EMG-like data
    EMG_MODE_ANALOG,      // Use built-in ADC (limited channels)
    EMG_MODE_SPI_ADC,     // External ADC via SPI
    EMG_MODE_I2C_ADC,     // External ADC via I2C
    EMG_MODE_SERIAL,      // Receive data via serial from external frontend
};

/**
 * EMG Acquisition class
 *
 * Manages data acquisition and buffering.
 */
class EmgAcquisition {
public:
    EmgAcquisition();
    ~EmgAcquisition();

    /**
     * Initialize acquisition system.
     *
     * @param mode Acquisition mode (default: simulated for testing)
     * @return true on success
     */
    bool begin(EmgAcquisitionMode mode = EMG_MODE_SIMULATED);

    /**
     * Check if a complete window is ready.
     *
     * @return true if window buffer is full
     */
    bool isWindowReady() const { return window_ready_; }

    /**
     * Get the current window data.
     *
     * @param buffer Output buffer [EMG_CHANNELS * WINDOW_SAMPLES]
     *               Data is in channel-major format
     */
    void getWindow(float* buffer);

    /**
     * Start acquiring next window.
     *
     * Resets the buffer and begins acquisition.
     */
    void startNextWindow();

    /**
     * Get current sample count in buffer.
     */
    int getSampleCount() const { return sample_count_; }

    /**
     * Get acquisition mode.
     */
    EmgAcquisitionMode getMode() const { return mode_; }

    /**
     * Update acquisition (call from main loop or timer).
     *
     * For simulated and serial modes, this should be called frequently.
     * For timer-driven ADC, this may be a no-op.
     */
    void update();

    /**
     * Set callback for new sample (optional).
     *
     * @param callback Function to call when new sample acquired
     */
    typedef void (*SampleCallback)(const float* samples, int num_channels);
    void setSampleCallback(SampleCallback callback) { sample_callback_ = callback; }

private:
    EmgAcquisitionMode mode_;
    bool initialized_;
    bool window_ready_;
    bool acquiring_;

    // Double-buffered window storage
    float buffer_a_[EMG_CHANNELS * WINDOW_SAMPLES];
    float buffer_b_[EMG_CHANNELS * WINDOW_SAMPLES];
    float* active_buffer_;
    float* ready_buffer_;

    int sample_count_;
    uint32_t last_sample_time_;
    uint32_t sample_interval_us_;

    SampleCallback sample_callback_;

    // Simulated data generation
    float sim_phase_[EMG_CHANNELS];
    float sim_freq_[EMG_CHANNELS];

    /**
     * Add sample to buffer.
     *
     * @param samples Array of [EMG_CHANNELS] values
     */
    void addSample(const float* samples);

    /**
     * Generate simulated EMG sample.
     *
     * @param samples Output array [EMG_CHANNELS]
     */
    void generateSimulatedSample(float* samples);

    /**
     * Read from analog pins.
     *
     * @param samples Output array [EMG_CHANNELS]
     */
    void readAnalogSample(float* samples);

    /**
     * Read from serial.
     *
     * @param samples Output array [EMG_CHANNELS]
     * @return true if sample received
     */
    bool readSerialSample(float* samples);
};

#endif // EMG_ACQUISITION_H
