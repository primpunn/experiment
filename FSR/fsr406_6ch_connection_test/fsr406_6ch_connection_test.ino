/*
 * FSR406 × 6 — MCP6004 UGA — Connection Test (6-channel)
 *
 * Wiring (from wiring_6ch.png):
 *   3.3V ──── FSR406.P1
 *              FSR406.P2 ── Node A ──┬── R_M (10 kΩ) ── GND
 *                                    └── MCP6004 IN+ → MCP6004 OUT → ESP32 ADCx
 *
 *   MCP6004 #1 (3 buffers used, 1 NC):
 *     CH1 → GPIO34 (ADC1_CH6)
 *     CH2 → GPIO35 (ADC1_CH7)
 *     CH3 → GPIO32 (ADC1_CH4)
 *
 *   MCP6004 #2 (3 buffers used, 1 NC):
 *     CH4 → GPIO33 (ADC1_CH5)
 *     CH5 → GPIO39 (ADC1_CH3)
 *     CH6 → GPIO36 (ADC1_CH0)
 *
 * Expected ADC reading per channel:
 *   No force  → V_OUT ≈ 0 V   (raw ≈ 0)     R_FSR > 5 MΩ
 *   Light ~1N → V_OUT ≈ 0.76 V              R_FSR ≈ 3 kΩ
 *   Hard ~10N → V_OUT ≈ 2.02 V              R_FSR ≈ 1 kΩ
 *
 * Pass/Fail criteria:
 *   IDLE  : V_OUT < 30 mV   → sensor unloaded and wiring intact
 *   PRESS : 30–3250 mV      → measurable force, R_FSR and grams shown
 *   SHORT : V_OUT > 3250 mV → R_FSR shorted or R_M missing
 *   STUCK : Raw never moves → ADC pin disconnected or MCP6004 dead
 */

#include "driver/adc.h"
#include "esp_adc_cal.h"

// ── Circuit constants ────────────────────────────────────────────────
#define V_CC_MV        3300
#define R_M_OHM        10000.0f
#define SAMPLES        64
#define PRINT_MS       500
#define FSR_FORCE_K    3000000.0f   // grams · Ω  (FSR406 typical)
#define FSR_MIN_G      20.0f
#define FSR_MAX_G      10000.0f

// ── 6-channel pin table ──────────────────────────────────────────────
struct FsrCh {
    const char*     label;
    adc1_channel_t  adc_ch;
    uint8_t         gpio;
};

static const FsrCh CHANS[] = {
    { "FSR1", ADC1_CHANNEL_6, 34 },   // MCP6004 #1, buffer A
    { "FSR2", ADC1_CHANNEL_7, 35 },   // MCP6004 #1, buffer B
    { "FSR3", ADC1_CHANNEL_4, 32 },   // MCP6004 #1, buffer C
    { "FSR4", ADC1_CHANNEL_5, 33 },   // MCP6004 #2, buffer A
    { "FSR5", ADC1_CHANNEL_3, 39 },   // MCP6004 #2, buffer B
    { "FSR6", ADC1_CHANNEL_0, 36 },   // MCP6004 #2, buffer C
};
static const int N_CHANS = sizeof(CHANS) / sizeof(CHANS[0]);

// ── ADC calibration ──────────────────────────────────────────────────
static esp_adc_cal_characteristics_t adc_chars;

// ── Helpers ──────────────────────────────────────────────────────────
static uint32_t readOversampled(adc1_channel_t ch) {
    uint32_t s = 0;
    for (int i = 0; i < SAMPLES; i++) {
        s += adc1_get_raw(ch);
        delayMicroseconds(80);
    }
    return s / SAMPLES;
}

static float fsrToGrams(float r_ohm) {
    if (r_ohm <= 0.0f) return -1.0f;
    float g = FSR_FORCE_K / r_ohm;
    return (g >= FSR_MIN_G && g <= FSR_MAX_G) ? g : -1.0f;
}

// ════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(800);

    adc1_config_width(ADC_WIDTH_BIT_12);
    for (int i = 0; i < N_CHANS; i++)
        adc1_config_channel_atten(CHANS[i].adc_ch, ADC_ATTEN_DB_11);

    esp_adc_cal_value_t cal = esp_adc_cal_characterize(
        ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);

    Serial.println();
    Serial.println("╔══════════════════════════════════════════════════════════╗");
    Serial.println("║   FSR406 × 6 — MCP6004 UGA — 6-Channel Connection Test  ║");
    Serial.println("╚══════════════════════════════════════════════════════════╝");
    Serial.printf("ADC cal: %s\n",
        cal == ESP_ADC_CAL_VAL_EFUSE_VREF ? "eFuse Vref (best)" :
        cal == ESP_ADC_CAL_VAL_EFUSE_TP   ? "eFuse two-point"   :
                                             "default 1100 mV");
    Serial.println();
    Serial.println("Channel  GPIO  RAW    V_OUT    R_FSR          Force          Status");
    Serial.println("────────────────────────────────────────────────────────────────────");
}

// ════════════════════════════════════════════════════════════════════
void loop() {
    for (int i = 0; i < N_CHANS; i++) {
        uint32_t raw  = readOversampled(CHANS[i].adc_ch);
        uint32_t v_mv = esp_adc_cal_raw_to_voltage(raw, &adc_chars);
        float    v    = v_mv / 1000.0f;

        char r_str[16] = "-              ";
        char f_str[18] = "-              ";
        const char* status;

        if (v_mv < 30) {
            status = "IDLE — no force / wiring OK";

        } else if (v_mv > 3250) {
            status = "SHORT — check R_M & connections";

        } else {
            float ratio = (float)V_CC_MV / (float)v_mv - 1.0f;
            float r_fsr = R_M_OHM * ratio;

            if (r_fsr >= 1e6f)      snprintf(r_str, sizeof(r_str), "%.2f MΩ", r_fsr / 1e6f);
            else if (r_fsr >= 1e3f) snprintf(r_str, sizeof(r_str), "%.1f kΩ", r_fsr / 1e3f);
            else                    snprintf(r_str, sizeof(r_str), "%.0f Ω",   r_fsr);

            float g = fsrToGrams(r_fsr);
            if (g < 0)              snprintf(f_str, sizeof(f_str), "out-of-range   ");
            else if (g >= 1000.0f)  snprintf(f_str, sizeof(f_str), "%.2f kg        ", g / 1000.0f);
            else                    snprintf(f_str, sizeof(f_str), "%.0f g          ", g);

            if      (g < 0)    status = "below/above range";
            else if (g < 50)   status = "barely touching";
            else if (g < 500)  status = "light press";
            else if (g < 2000) status = "moderate press";
            else               status = "heavy press";
        }

        Serial.printf("%-8s GPIO%-2u  %4lu   %.3f V  %-14s %-16s %s\n",
                      CHANS[i].label, CHANS[i].gpio,
                      raw, v, r_str, f_str, status);
    }

    Serial.println("────────────────────────────────────────────────────────────────────");
    delay(PRINT_MS);
}
