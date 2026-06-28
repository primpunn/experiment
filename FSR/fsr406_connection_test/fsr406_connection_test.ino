/*
 * FSR406 + MCP6004 UGA Connection Test
 *
 * Circuit (from wiring diagram):
 *   3.3V ──── FSR406.P1
 *              FSR406.P2 ── Node A ──┬── R_M(10kΩ) ── GND
 *                                    └── MCP6004.pin3 (IN_A+)
 *   MCP6004.pin1 (OUT_A) ── MCP6004.pin2 (IN_A−)  [feedback → unity gain]
 *   MCP6004.pin1 (OUT_A) ── ESP32 GPIO34 (ADC1_CH6)
 *   MCP6004.pin4 (VDD)   = 3.3V,  pin11 (VSS) = GND
 *
 * Expected behavior:
 *   No force  → R_FSR > 10 MΩ → V_OUT ≈ 0 V  (raw ≈ 0)
 *   Light ~1N → R_FSR ≈ 3 kΩ  → V_OUT ≈ 0.76 V
 *   Heavy ~10N→ R_FSR ≈ 1 kΩ  → V_OUT ≈ 2.02 V
 *   V_A = 3.3 × R_M / (R_FSR + R_M)
 *
 * Force estimation (FSR406 datasheet curve):
 *   F_grams ≈ FSR_FORCE_K / R_FSR_ohm  (valid range: 20 g – 10 kg)
 *
 *   FSR_FORCE_K default = 3,000,000 (Interlink FSR406 typical).
 *   For better accuracy, calibrate with two known weights:
 *     1. Press a known weight W1 (grams) → read R1 (Ω)
 *     2. Press a known weight W2 (grams) → read R2 (Ω)
 *     3. FSR_FORCE_K = (W1×R1 + W2×R2) / 2
 */

#include "driver/adc.h"
#include "esp_adc_cal.h"

// ── Pin & circuit constants ─────────────────────────────────────────
#define FSR_ADC_CHANNEL   ADC1_CHANNEL_6   // GPIO34
#define V_CC_MV           3300             // mV
#define R_M_OHM           10000.0f         // 10 kΩ pull-down
#define SAMPLES           64               // oversampling count
#define PRINT_INTERVAL_MS 300

// ── Force calibration constant ───────────────────────────────────────
// FSR406 datasheet: conductance is roughly proportional to force.
// F_grams ≈ FSR_FORCE_K / R_FSR_ohm
// Adjust this value if you have known reference weights (see header comment).
#define FSR_FORCE_K       3000000.0f

// ── Sensor valid range ───────────────────────────────────────────────
#define FSR_MIN_G         20.0f
#define FSR_MAX_G         10000.0f

// ── ADC calibration ─────────────────────────────────────────────────
static esp_adc_cal_characteristics_t adc_chars;

void setup() {
    Serial.begin(115200);
    delay(800);

    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(FSR_ADC_CHANNEL, ADC_ATTEN_DB_11);  // full 0–3.3 V range

    esp_adc_cal_value_t cal = esp_adc_cal_characterize(
        ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);

    Serial.println("\n╔══════════════════════════════════════════╗");
    Serial.println("║   FSR406 + MCP6004 UGA — Connection Test  ║");
    Serial.println("╚══════════════════════════════════════════╝");
    Serial.printf("ADC cal source: %s\n",
        cal == ESP_ADC_CAL_VAL_EFUSE_VREF   ? "eFuse Vref (best)" :
        cal == ESP_ADC_CAL_VAL_EFUSE_TP     ? "eFuse two-point"   :
                                               "default 1100 mV");
    Serial.printf("Force constant K = %.0f  (F_g ≈ K / R_FSR)\n", FSR_FORCE_K);
    Serial.println("GPIO34 (ADC1_CH6) ← MCP6004 OUT_A");
    Serial.println("──────────────────────────────────────────────────────────");
    Serial.println("RAW    V_OUT    R_FSR          Force         Status");
    Serial.println("──────────────────────────────────────────────────────────");
}

// ── Estimate force from R_FSR (returns grams, -1 if out of range) ────
float fsrToGrams(float r_fsr_ohm) {
    if (r_fsr_ohm <= 0.0f) return -1.0f;
    float g = FSR_FORCE_K / r_fsr_ohm;
    if (g < FSR_MIN_G || g > FSR_MAX_G) return -1.0f;
    return g;
}

// ── Format force for display ─────────────────────────────────────────
void formatForce(float grams, char* buf, size_t len) {
    if (grams < 0.0f) {
        snprintf(buf, len, "  out of range ");
    } else if (grams >= 1000.0f) {
        snprintf(buf, len, "%6.2f kg      ", grams / 1000.0f);
    } else {
        snprintf(buf, len, "%6.0f g       ", grams);
    }
}

void loop() {
    // ── Oversample ───────────────────────────────────────────────────
    uint32_t raw_sum = 0;
    for (int i = 0; i < SAMPLES; i++) {
        raw_sum += adc1_get_raw(FSR_ADC_CHANNEL);
        delayMicroseconds(80);
    }
    uint32_t raw = raw_sum / SAMPLES;

    // ── Calibrated voltage ───────────────────────────────────────────
    uint32_t v_mv = esp_adc_cal_raw_to_voltage(raw, &adc_chars);
    float v_out   = v_mv / 1000.0f;

    // ── Diagnostics ──────────────────────────────────────────────────
    if (v_mv < 30) {
        Serial.printf("%4lu   %.3f V  > 5 MΩ          —             No force / check wiring\n",
                      raw, v_out);

    } else if (v_mv > 3250) {
        Serial.printf("%4lu   %.3f V  < 100 Ω          —             SATURATED — check R_M & short\n",
                      raw, v_out);

    } else {
        // R_FSR = R_M × (V_CC / V_A − 1)
        float ratio  = (float)V_CC_MV / (float)v_mv - 1.0f;
        float r_fsr  = R_M_OHM * ratio;
        float grams  = fsrToGrams(r_fsr);

        char r_str[16];
        if (r_fsr >= 1e6f)
            snprintf(r_str, sizeof(r_str), "%.2f MΩ", r_fsr / 1e6f);
        else if (r_fsr >= 1e3f)
            snprintf(r_str, sizeof(r_str), "%.1f kΩ", r_fsr / 1e3f);
        else
            snprintf(r_str, sizeof(r_str), "%.0f Ω ", r_fsr);

        char f_str[20];
        formatForce(grams, f_str, sizeof(f_str));

        const char* status;
        if      (grams < 0)       status = "below/above range";
        else if (grams < 50)      status = "barely touching";
        else if (grams < 500)     status = "light press";
        else if (grams < 2000)    status = "moderate press";
        else                      status = "heavy press";

        Serial.printf("%4lu   %.3f V  %-13s  %s  %s\n",
                      raw, v_out, r_str, f_str, status);
    }

    delay(PRINT_INTERVAL_MS);
}
