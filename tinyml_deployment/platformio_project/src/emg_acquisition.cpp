/**
 * EMG Acquisition Implementation
 *
 * Data acquisition for XIAO nRF52840.
 */

#include "emg_acquisition.h"
#include <Arduino.h>
#include <cstring>
#include <cmath>

// Analog pin mapping for XIAO nRF52840
// Note: XIAO has limited analog pins (A0-A5)
// For 7 channels, you'll need external ADC
static const int ANALOG_PINS[] = {A0, A1, A2, A3, A4, A5, A0};  // A0 repeated as placeholder

// ADC configuration
static const float ADC_VREF = 3.3f;         // Reference voltage
static const int ADC_RESOLUTION = 4096;      // 12-bit ADC
static const float ADC_TO_UV = (ADC_VREF / ADC_RESOLUTION) * 1000000.0f;  // Convert to microvolts

EmgAcquisition::EmgAcquisition()
    : mode_(EMG_MODE_SIMULATED)
    , initialized_(false)
    , window_ready_(false)
    , acquiring_(false)
    , active_buffer_(buffer_a_)
    , ready_buffer_(buffer_b_)
    , sample_count_(0)
    , last_sample_time_(0)
    , sample_interval_us_(1000000 / EMG_SAMPLE_RATE)  // 500us for 2kHz
    , sample_callback_(nullptr)
{
    memset(buffer_a_, 0, sizeof(buffer_a_));
    memset(buffer_b_, 0, sizeof(buffer_b_));
    memset(sim_phase_, 0, sizeof(sim_phase_));

    // Initialize simulated signal frequencies
    for (int i = 0; i < EMG_CHANNELS; i++) {
        sim_freq_[i] = 20.0f + i * 15.0f;  // 20-110Hz range
    }
}

EmgAcquisition::~EmgAcquisition() {
    // Nothing to clean up
}

bool EmgAcquisition::begin(EmgAcquisitionMode mode) {
    mode_ = mode;

    switch (mode) {
        case EMG_MODE_SIMULATED:
            Serial.println("EMG: Using simulated data");
            break;

        case EMG_MODE_ANALOG:
            Serial.println("EMG: Using analog ADC");
            // Configure analog pins
            for (int i = 0; i < min(EMG_CHANNELS, 6); i++) {
                pinMode(ANALOG_PINS[i], INPUT);
            }
            analogReadResolution(12);
            break;

        case EMG_MODE_SERIAL:
            Serial.println("EMG: Using serial input");
            // Serial should already be initialized
            break;

        case EMG_MODE_SPI_ADC:
        case EMG_MODE_I2C_ADC:
            Serial.println("EMG: External ADC not implemented");
            return false;
    }

    initialized_ = true;
    startNextWindow();
    return true;
}

void EmgAcquisition::startNextWindow() {
    // Swap buffers
    float* temp = active_buffer_;
    active_buffer_ = ready_buffer_;
    ready_buffer_ = temp;

    // Reset state
    sample_count_ = 0;
    window_ready_ = false;
    acquiring_ = true;
    last_sample_time_ = micros();
}

void EmgAcquisition::getWindow(float* buffer) {
    if (ready_buffer_ && buffer) {
        memcpy(buffer, ready_buffer_, EMG_CHANNELS * WINDOW_SAMPLES * sizeof(float));
    }
}

void EmgAcquisition::update() {
    if (!initialized_ || !acquiring_) {
        return;
    }

    // Check if it's time for next sample
    uint32_t now = micros();
    uint32_t elapsed = now - last_sample_time_;

    // Handle micros() overflow
    if (elapsed < 0) {
        elapsed = now + (UINT32_MAX - last_sample_time_);
    }

    if (elapsed >= sample_interval_us_) {
        float samples[EMG_CHANNELS];

        switch (mode_) {
            case EMG_MODE_SIMULATED:
                generateSimulatedSample(samples);
                addSample(samples);
                break;

            case EMG_MODE_ANALOG:
                readAnalogSample(samples);
                addSample(samples);
                break;

            case EMG_MODE_SERIAL:
                if (readSerialSample(samples)) {
                    addSample(samples);
                }
                break;

            default:
                break;
        }

        last_sample_time_ = now;
    }
}

void EmgAcquisition::addSample(const float* samples) {
    if (sample_count_ >= WINDOW_SAMPLES) {
        return;
    }

    // Store in channel-major format
    for (int ch = 0; ch < EMG_CHANNELS; ch++) {
        active_buffer_[ch * WINDOW_SAMPLES + sample_count_] = samples[ch];
    }

    sample_count_++;

    // Callback
    if (sample_callback_) {
        sample_callback_(samples, EMG_CHANNELS);
    }

    // Check if window complete
    if (sample_count_ >= WINDOW_SAMPLES) {
        window_ready_ = true;
        acquiring_ = false;
    }
}

void EmgAcquisition::generateSimulatedSample(float* samples) {
    // Generate realistic EMG-like signals
    // EMG characteristics:
    // - Broadband noise (20-450Hz)
    // - Amplitude varies with muscle activation
    // - Some baseline noise

    float time = millis() / 1000.0f;

    // Simulate varying muscle activation
    float activation = 0.5f + 0.4f * sinf(2.0f * PI * 0.2f * time);  // Slow 0.2Hz modulation

    for (int ch = 0; ch < EMG_CHANNELS; ch++) {
        // Update phase
        sim_phase_[ch] += 2.0f * PI * sim_freq_[ch] / EMG_SAMPLE_RATE;
        if (sim_phase_[ch] > 2.0f * PI) {
            sim_phase_[ch] -= 2.0f * PI;
        }

        // Base signal components
        float signal = 0.0f;

        // Motor unit action potentials (multiple frequencies)
        signal += sinf(sim_phase_[ch]) * 20.0f;
        signal += sinf(sim_phase_[ch] * 2.3f) * 15.0f;
        signal += sinf(sim_phase_[ch] * 5.7f) * 10.0f;

        // Modulate by activation level
        signal *= activation;

        // Add noise
        float noise = (random(-1000, 1000) / 1000.0f) * 5.0f;
        signal += noise;

        // Simulate occasional gesture events
        static int gesture_timer = 0;
        static int current_gesture = 0;
        gesture_timer++;
        if (gesture_timer > EMG_SAMPLE_RATE * 2) {  // Every 2 seconds
            gesture_timer = 0;
            current_gesture = (current_gesture + 1) % 9;

            // Add burst for gesture
            if (ch == current_gesture % EMG_CHANNELS) {
                signal += 100.0f * (1.0f - (gesture_timer / 500.0f));
            }
        }

        // Output in microvolts (typical EMG range: 0-1000uV)
        samples[ch] = signal;
    }
}

void EmgAcquisition::readAnalogSample(float* samples) {
    // Read from analog pins
    // Note: XIAO nRF52840 ADC is single-ended, 0-3.3V range

    for (int ch = 0; ch < EMG_CHANNELS; ch++) {
        if (ch < 6) {  // Only 6 analog pins available
            int raw = analogRead(ANALOG_PINS[ch]);

            // Convert to microvolts
            // Assuming EMG frontend with gain and offset
            // Typical setup: EMG signal centered at VCC/2 with gain
            float voltage = (raw - ADC_RESOLUTION / 2) * ADC_TO_UV;

            // Assuming frontend gain of 1000x
            samples[ch] = voltage / 1000.0f;
        } else {
            samples[ch] = 0.0f;
        }
    }
}

bool EmgAcquisition::readSerialSample(float* samples) {
    // Protocol: Expect binary packets
    // Format: [START_BYTE] [CH0_H] [CH0_L] ... [CH6_H] [CH6_L] [CHECKSUM]
    // Or: ASCII CSV format "ch0,ch1,ch2,ch3,ch4,ch5,ch6\n"

    // Simple ASCII implementation for testing
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');

        int ch = 0;
        int start = 0;
        for (int i = 0; i <= line.length() && ch < EMG_CHANNELS; i++) {
            if (i == line.length() || line[i] == ',') {
                String val = line.substring(start, i);
                samples[ch] = val.toFloat();
                ch++;
                start = i + 1;
            }
        }

        // Fill remaining channels with zeros
        while (ch < EMG_CHANNELS) {
            samples[ch++] = 0.0f;
        }

        return true;
    }

    return false;
}
