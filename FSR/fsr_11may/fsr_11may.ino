/*
 * fsr_11may.ino — FSR406 × 6  100 Hz Streaming
 *
 * Hardware (same wiring as fsr406_6ch_calibration):
 *   3.3V ── FSR406_P1; FSR406_P2 ── Node A ──┬── R_M (10 kΩ) ── GND
 *                                             └── MCP6004 buffer ── ESP32 ADC
 *
 *   MCP6004 #1:  FSR1 → GPIO34  FSR2 → GPIO35  FSR3 → GPIO32
 *   MCP6004 #2:  FSR4 → GPIO33  FSR5 → GPIO39  FSR6 → GPIO36
 *
 * Serial protocol (115200 baud):
 *   Commands (send as a line + newline):
 *     START  — begin 100 Hz streaming (M lines)
 *     STOP   — stop streaming
 *     X      — alias for STOP
 *     D      — single-shot: one DATA line (all 6 channels)
 *     ?      — help
 *
 *   Output line formats:
 *     M,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv     streaming (100 Hz)
 *     DATA,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv  single-shot
 *     # ...   comment / status
 *
 * Calibration and recording are fully controlled from Python (calibrate.py,
 * record.py). This firmware is intentionally simple — it is just a data source.
 *
 * Timing:
 *   OVERSAMPLE = 4 × 6 channels × ~130 µs/read ≈ 3.1 ms ADC time per loop.
 *   Output at 100 Hz (10 ms interval) → ~69% headroom.
 *   Serial bandwidth: 100 Hz × ~45 bytes = 4500 B/s < 11520 B/s at 115200 baud.
 */

#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ── Configuration ──────────────────────────────────────────────────────────────
#define N_CHANS          6
#define OVERSAMPLE       4          // ADC reads averaged per channel per sample
#define STREAM_INTERVAL  10         // ms between M lines = 100 Hz

// ── Channel table ──────────────────────────────────────────────────────────────
struct FsrCh { adc1_channel_t ch; uint8_t gpio; };

static const FsrCh CHANS[N_CHANS] = {
    { ADC1_CHANNEL_6, 34 },    // FSR1 (calf 1)
    { ADC1_CHANNEL_7, 35 },    // FSR2 (calf 2)
    { ADC1_CHANNEL_4, 32 },    // FSR3 (calf 3)
    { ADC1_CHANNEL_5, 33 },    // FSR4 (calf 4)
    { ADC1_CHANNEL_3, 39 },    // FSR5 (foot 1)
    { ADC1_CHANNEL_0, 36 },    // FSR6 (foot 2)
};

// ── State ──────────────────────────────────────────────────────────────────────
static esp_adc_cal_characteristics_t adc_chars;
static bool     streaming = false;
static uint32_t next_ms   = 0;
static char     cmd_buf[16];
static int      cmd_len   = 0;

// ── ADC helper ─────────────────────────────────────────────────────────────────
static uint32_t readMv(int i) {
    uint32_t s = 0;
    for (int k = 0; k < OVERSAMPLE; k++) {
        s += adc1_get_raw(CHANS[i].ch);
        delayMicroseconds(60);
    }
    return esp_adc_cal_raw_to_voltage(s / OVERSAMPLE, &adc_chars);
}

static void sendAllChannels(const char* prefix) {
    Serial.printf("%s,%lu", prefix, millis());
    for (int i = 0; i < N_CHANS; i++) {
        Serial.printf(",%lu", readMv(i));
    }
    Serial.println();
}

// ── Command handler ────────────────────────────────────────────────────────────
static void processCmd(const char* buf) {
    // Case-insensitive compare
    char up[16];
    int len = 0;
    while (buf[len] && len < 15) { up[len] = toupper(buf[len]); len++; }
    up[len] = '\0';

    if (strcmp(up, "START") == 0) {
        streaming = true;
        next_ms   = millis();
        Serial.println(F("# Streaming 100 Hz — M,ts_ms,v0_mv..v5_mv"));

    } else if (strcmp(up, "STOP") == 0 || strcmp(up, "X") == 0) {
        streaming = false;
        Serial.println(F("# Stopped"));

    } else if (up[0] == 'D') {
        sendAllChannels("DATA");

    } else if (up[0] == '?') {
        Serial.println(F("# ── FSR406×6 fsr_11may ─────────────────────────"));
        Serial.println(F("#   START  begin 100 Hz streaming (M lines)"));
        Serial.println(F("#   STOP   stop streaming"));
        Serial.println(F("#   D      single-shot read (DATA line)"));
        Serial.println(F("# Channels: FSR1-4 = calf, FSR5-6 = foot"));
        Serial.println(F("# M,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv"));

    } else {
        Serial.printf("# Unknown command: %s  (? for help)\n", buf);
    }
}

// ── Setup ──────────────────────────────────────────────────────────────────────
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);
    delay(2000);

    adc1_config_width(ADC_WIDTH_BIT_12);
    for (int i = 0; i < N_CHANS; i++) {
        adc1_config_channel_atten(CHANS[i].ch, ADC_ATTEN_DB_11);
    }
    esp_adc_cal_value_t cal = esp_adc_cal_characterize(
        ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);

    Serial.println(F("\n# ════════════════════════════════════════"));
    Serial.println(F("#  FSR406×6  fsr_11may  100 Hz streaming"));
    Serial.println(F("# ════════════════════════════════════════"));
    Serial.printf( "# ADC cal: %s\n",
        cal == ESP_ADC_CAL_VAL_EFUSE_VREF ? "eFuse Vref (best)" :
        cal == ESP_ADC_CAL_VAL_EFUSE_TP   ? "eFuse two-point"   :
                                             "default 1100 mV");
    Serial.println(F("# Send START to begin streaming, D for a single read, ? for help."));
}

// ── Loop ───────────────────────────────────────────────────────────────────────
void loop() {
    // Non-blocking serial command reader
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmd_len > 0) {
                cmd_buf[cmd_len] = '\0';
                processCmd(cmd_buf);
                cmd_len = 0;
            }
        } else if (cmd_len < (int)(sizeof(cmd_buf) - 1)) {
            cmd_buf[cmd_len++] = c;
        }
    }

    // 100 Hz streaming
    if (streaming && millis() >= next_ms) {
        next_ms += STREAM_INTERVAL;
        sendAllChannels("M");
    }
}
