/**
 * sEMG Gesture Classifier - Main Application
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * Board: seeed-xiao-afruitnrf52-nrf52840
 * MCU: nRF52840 (ARM Cortex-M4F @ 64MHz)
 * RAM: 256KB (~220KB usable), Flash: 1MB + 2MB onboard
 *
 * Model Options:
 *   - Compressed M1 CNN+LSTM (DEFAULT): 45K params, ~80KB arena
 *   - M2 CNN-only: 160K params, ~100KB arena
 *   - Original M1: 411K params - DOES NOT FIT (needs 398KB)
 *
 * Input: 7 EMG channels @ 2kHz, 500ms windows (1000 samples)
 * Output: 9 gesture classes with confidence scores
 *
 * Memory Budget (Compressed M1):
 *   - Tensor arena: 80KB
 *   - Input buffer: 28KB (7ch × 1000 × 4 bytes)
 *   - TFLite runtime: 25KB
 *   - LSTM states: 0.3KB
 *   - Stack/heap: 40KB
 *   - Total: ~173KB (fits in 220KB)
 *
 * Gesture Classes:
 *   0: index_press      4: thumb_click     8: thumb_up
 *   1: index_release    5: thumb_down
 *   2: middle_press     6: thumb_in
 *   3: middle_release   7: thumb_out
 */

#include <Arduino.h>
#include "gesture_classifier.h"
#include "emg_acquisition.h"
#include "signal_processing.h"

// ============================================================================
// Configuration
// ============================================================================

#define SERIAL_BAUD_RATE 115200

// LED pin - XIAO nRF52840 has built-in LED
#ifndef LED_BUILTIN
#define LED_BUILTIN 13
#endif
#define LED_PIN LED_BUILTIN

// Inference settings
#define CONFIDENCE_THRESHOLD 0.5f    // Minimum confidence to report gesture
#define DEBOUNCE_MS 200              // Minimum time between gesture reports

// ============================================================================
// Gesture Names
// ============================================================================

const char* GESTURE_NAMES[] = {
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
// Global Objects
// ============================================================================

// EMG data buffer - channel-major format [ch0_t0...ch0_tN, ch1_t0...ch1_tN, ...]
static float emg_buffer[EMG_CHANNELS * WINDOW_SAMPLES];

// Output confidence scores
static float output_buffer[NUM_GESTURES];

// Core components
static GestureClassifier classifier;
static EmgAcquisition emg;
static SignalProcessor signal_proc;

// State tracking
static bool classifier_ready = false;
static uint32_t inference_count = 0;
static uint32_t last_inference_time = 0;
static uint32_t last_gesture_time = 0;
static int last_gesture = -1;

// ============================================================================
// Function Declarations
// ============================================================================

void setup_hardware();
void run_inference();
void output_results(const float* confidences, int predicted_class);
void blink_gesture(int gesture_class);
void print_startup_info();
void handle_serial_commands();

// ============================================================================
// Setup
// ============================================================================

void setup() {
    // Initialize serial with timeout
    Serial.begin(SERIAL_BAUD_RATE);

    // Wait for serial connection (important for USB CDC on nRF52840)
    uint32_t start = millis();
    while (!Serial && (millis() - start) < 5000) {
        delay(10);
    }

    // Print startup banner
    print_startup_info();

    // Setup hardware peripherals
    setup_hardware();

    // Initialize signal processor
    Serial.print("[INIT] Signal processor... ");
    if (signal_proc.begin()) {
        Serial.println("OK");
    } else {
        Serial.println("FAILED");
    }

    // Initialize EMG acquisition (default: simulated data)
    Serial.print("[INIT] EMG acquisition... ");
    if (emg.begin(EMG_MODE_SIMULATED)) {
        Serial.println("OK (simulated mode)");
    } else {
        Serial.println("FAILED");
    }

    // Initialize gesture classifier (TFLite Micro)
    Serial.print("[INIT] ML model... ");
    if (classifier.begin()) {
        Serial.println("OK");
        classifier_ready = true;
        classifier.printMemoryInfo();
    } else {
        Serial.println("FAILED");
        Serial.println("ERROR: Could not initialize TFLite model!");
        Serial.println("Make sure model header is included in project.");
    }

    Serial.println("\n========================================");
    Serial.println("Ready for inference!");
    Serial.println("Commands: 's'=status, 'r'=reset, 't'=test, 'm'=memory");
    Serial.println("========================================\n");
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
    // Handle classifier not ready
    if (!classifier_ready) {
        // Blink error pattern (fast)
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
        return;
    }

    // Update EMG acquisition
    emg.update();

    // Check if window is ready for inference
    if (emg.isWindowReady()) {
        // Get EMG window data
        emg.getWindow(emg_buffer);

        // Apply signal preprocessing (bandpass filter, notch filter)
        signal_proc.process(emg_buffer, EMG_CHANNELS, WINDOW_SAMPLES);

        // Run ML inference
        run_inference();

        // Start acquiring next window
        emg.startNextWindow();
    }

    // Handle serial commands
    handle_serial_commands();
}

// ============================================================================
// Hardware Setup
// ============================================================================

void setup_hardware() {
    // Configure LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Configure ADC for EMG input (if using analog)
    // XIAO nRF52840 has 12-bit ADC, 6 analog channels
    analogReadResolution(12);

    // Note: For high-quality EMG, use external ADC via SPI/I2C
    // - ADS1299 (8-channel, 24-bit, research-grade)
    // - ADS1115 (4-channel, 16-bit, hobby-grade)

    Serial.println("[HW] Peripherals configured");
}

// ============================================================================
// Inference
// ============================================================================

void run_inference() {
    uint32_t start_time = millis();

    // Run ML classifier
    int predicted_class = classifier.classify(emg_buffer, output_buffer);

    last_inference_time = millis() - start_time;
    inference_count++;

    // Find max confidence
    float max_conf = 0.0f;
    for (int i = 0; i < NUM_GESTURES; i++) {
        if (output_buffer[i] > max_conf) {
            max_conf = output_buffer[i];
        }
    }

    // Apply debouncing and threshold
    uint32_t now = millis();
    bool should_report = (max_conf >= CONFIDENCE_THRESHOLD) &&
                         (predicted_class != last_gesture ||
                          (now - last_gesture_time) > DEBOUNCE_MS);

    if (should_report) {
        output_results(output_buffer, predicted_class);
        blink_gesture(predicted_class);
        last_gesture = predicted_class;
        last_gesture_time = now;
    }
}

// ============================================================================
// Output Results
// ============================================================================

void output_results(const float* confidences, int predicted_class) {
    // Find max confidence
    float max_conf = 0.0f;
    for (int i = 0; i < NUM_GESTURES; i++) {
        if (confidences[i] > max_conf) {
            max_conf = confidences[i];
        }
    }

    // Output in JSON format for easy parsing
    Serial.print("{");
    Serial.printf("\"n\":%lu,", inference_count);
    Serial.printf("\"ms\":%lu,", last_inference_time);
    Serial.printf("\"class\":%d,", predicted_class);
    Serial.printf("\"gesture\":\"%s\",", GESTURE_NAMES[predicted_class]);
    Serial.printf("\"conf\":%.3f,", max_conf);
    Serial.print("\"scores\":[");
    for (int i = 0; i < NUM_GESTURES; i++) {
        Serial.printf("%.2f", confidences[i]);
        if (i < NUM_GESTURES - 1) Serial.print(",");
    }
    Serial.println("]}");
}

// ============================================================================
// Visual Feedback
// ============================================================================

void blink_gesture(int gesture_class) {
    // Blink pattern based on gesture class
    if (gesture_class >= 0 && gesture_class < NUM_GESTURES) {
        // Short blink for detection
        digitalWrite(LED_PIN, HIGH);
        delay(30);
        digitalWrite(LED_PIN, LOW);
    }
}

// ============================================================================
// Serial Command Handler
// ============================================================================

void handle_serial_commands() {
    if (!Serial.available()) return;

    char cmd = Serial.read();

    switch (cmd) {
        case 's':  // Status
            Serial.println("\n=== Status ===");
            Serial.printf("Inferences: %lu\n", inference_count);
            Serial.printf("Last latency: %lu ms\n", last_inference_time);
            Serial.printf("Last gesture: %s\n",
                          last_gesture >= 0 ? GESTURE_NAMES[last_gesture] : "none");
            Serial.printf("Classifier ready: %s\n", classifier_ready ? "yes" : "no");
            break;

        case 'r':  // Reset
            inference_count = 0;
            last_gesture = -1;
            last_gesture_time = 0;
            Serial.println("Stats reset");
            break;

        case 't':  // Test inference
            Serial.println("\n=== Test Inference ===");
            // Generate synthetic EMG-like data
            for (int ch = 0; ch < EMG_CHANNELS; ch++) {
                for (int t = 0; t < WINDOW_SAMPLES; t++) {
                    float phase = (float)t / WINDOW_SAMPLES * 6.28f * (ch + 1);
                    emg_buffer[ch * WINDOW_SAMPLES + t] = sinf(phase) * 50.0f + (rand() % 10);
                }
            }
            {
                uint32_t start = millis();
                int result = classifier.classify(emg_buffer, output_buffer);
                uint32_t elapsed = millis() - start;
                Serial.printf("Result: %s (class %d)\n",
                              GESTURE_NAMES[result], result);
                Serial.printf("Latency: %lu ms\n", elapsed);
            }
            break;

        case 'm':  // Memory info
            classifier.printMemoryInfo();
            break;

        case 'h':  // Help
        case '?':
            Serial.println("\n=== Commands ===");
            Serial.println("s - Show status");
            Serial.println("r - Reset stats");
            Serial.println("t - Test inference");
            Serial.println("m - Memory info");
            Serial.println("h - This help");
            break;

        default:
            break;
    }
}

// ============================================================================
// Startup Info
// ============================================================================

void print_startup_info() {
    Serial.println();
    Serial.println("========================================");
    Serial.println("  sEMG Gesture Classifier v1.0");
    Serial.println("========================================");
    Serial.println();
    Serial.println("Target: Seeed Studio XIAO nRF52840");
    Serial.println("MCU:    nRF52840 (Cortex-M4F @ 64MHz)");
    Serial.println("RAM:    256KB");
    Serial.println("Flash:  1MB + 2MB onboard");
    Serial.println();
    Serial.printf("Input:  %d channels x %d samples\n", EMG_CHANNELS, WINDOW_SAMPLES);
    Serial.printf("Output: %d gesture classes\n", NUM_GESTURES);
    Serial.printf("Window: %d ms @ 2kHz\n", WINDOW_SAMPLES / 2);
    Serial.println();
}
