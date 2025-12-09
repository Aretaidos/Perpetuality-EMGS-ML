/**
 * Gesture Classifier - TFLite Micro Interface
 *
 * Wraps TensorFlow Lite Micro for gesture classification
 * on Seeed Studio XIAO nRF52840.
 *
 * Supports:
 * - Compressed M1 CNN+LSTM (RECOMMENDED): ~45K params, fits in RAM
 * - M2 CNN-only: ~160K params, fits in RAM
 * - Original M1 CNN+LSTM: 411K params, DOES NOT FIT (398KB needed)
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * MCU: nRF52840 (ARM Cortex-M4F @ 64MHz)
 * RAM: 256KB (~220KB usable)
 */

#ifndef GESTURE_CLASSIFIER_H
#define GESTURE_CLASSIFIER_H

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Model Configuration
// ============================================================================

#ifndef EMG_CHANNELS
#define EMG_CHANNELS 7
#endif

// Window size - reduced for compressed M1
// 1000 samples = 500ms @ 2kHz (fits in RAM)
// Original M1 used 10000 samples but doesn't fit
#ifndef WINDOW_SAMPLES
#define WINDOW_SAMPLES 1000  // 500ms @ 2kHz
#endif

#ifndef NUM_GESTURES
#define NUM_GESTURES 9
#endif

// Output timesteps (input / conv_stride)
// For compressed M1: 1000 / 10 = 100 timesteps (minus kernel overhead)
#define OUTPUT_TIMESTEPS ((WINDOW_SAMPLES - 14) / 10)

// ============================================================================
// Memory Configuration
// ============================================================================

// Memory arena size for TFLite Micro (in bytes)
// CRITICAL: Must fit in nRF52840's 256KB RAM (~220KB usable)
//
// Memory budget for Compressed M1:
//   - Tensor arena: 80KB (reduced for smaller model)
//   - Input buffer: 28KB (7 channels * 1000 samples * 4 bytes)
//   - TFLite runtime: 25KB
//   - LSTM states: 0.3KB (32 hidden * 2 states * 4 bytes)
//   - Stack/heap: 40KB
//   - Total: ~173KB (fits in 220KB usable)
//
// Model comparison:
//   Compressed M1: 64 conv channels, 32 LSTM hidden, 1 layer -> ~80KB arena
//   Original M1:   128 conv channels, 128 LSTM hidden, 3 layers -> ~398KB (NO FIT)
//   M2 CNN-only:   No LSTM -> ~80-100KB arena
//
#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (80 * 1024)  // 80KB for compressed M1
#endif

// ============================================================================
// Model Selection
// ============================================================================

// Select which model header to use
// Options:
//   - m1_compressed_model.h  (RECOMMENDED for M1 architecture)
//   - m2_model.h             (CNN-only alternative)
//
// The selected model should match WINDOW_SAMPLES and architecture

/**
 * GestureClassifier class
 *
 * Manages TFLite Micro interpreter and provides inference API.
 */
class GestureClassifier {
public:
    GestureClassifier();
    ~GestureClassifier();

    /**
     * Initialize the classifier.
     *
     * Loads the TFLite model and allocates tensors.
     *
     * @return true if initialization successful
     */
    bool begin();

    /**
     * Run gesture classification on EMG window.
     *
     * @param emg_data Input EMG data, shape [EMG_CHANNELS * WINDOW_SAMPLES]
     *                 Data should be in channel-major format: [ch0_t0, ch0_t1, ..., ch0_tN, ch1_t0, ...]
     * @param confidences Output confidence scores for each gesture [NUM_GESTURES]
     * @return Predicted gesture class (0-8), or -1 on error
     */
    int classify(const float* emg_data, float* confidences);

    /**
     * Get the tensor arena size.
     *
     * @return Arena size in bytes
     */
    int getArenaSize() const { return TENSOR_ARENA_SIZE; }

    /**
     * Check if classifier is ready.
     *
     * @return true if model loaded and ready for inference
     */
    bool isReady() const { return initialized_; }

    /**
     * Get last inference latency in milliseconds.
     *
     * @return Latency of last inference
     */
    uint32_t getLastLatency() const { return last_latency_ms_; }

    /**
     * Get free heap memory.
     *
     * @return Free heap in bytes
     */
    static int getFreeHeap();

    /**
     * Print memory information to Serial.
     */
    void printMemoryInfo();

private:
    bool initialized_;
    uint32_t last_latency_ms_;

    // TFLite Micro objects (forward declarations)
    // Actual implementations use TFLite Micro types
    void* model_;
    void* interpreter_;
    void* tensor_arena_;
    void* input_tensor_;
    void* output_tensor_;

    /**
     * Apply Reinhard compression to input data.
     *
     * Implements: output = range * x / (midpoint + |x|)
     * With range=1.0, midpoint=32.0
     *
     * @param data Input/output data buffer
     * @param size Number of elements
     */
    void applyReinhardCompression(float* data, int size);

    /**
     * Post-process output logits to confidence scores.
     *
     * Applies softmax and temporal averaging.
     *
     * @param logits Raw output logits [NUM_GESTURES * OUTPUT_TIMESTEPS]
     * @param confidences Output confidence scores [NUM_GESTURES]
     * @return Predicted class index
     */
    int postProcess(const float* logits, float* confidences);
};

#endif // GESTURE_CLASSIFIER_H
