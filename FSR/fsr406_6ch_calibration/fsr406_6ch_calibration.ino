/*
 * FSR406 × 6  Calibration Protocol — 6-Channel
 *
 * Hardware (from wiring_6ch.png):
 *   3.3V ──── FSR406.P1
 *              FSR406.P2 ── Node A ──┬── R_M (10 kΩ) ── GND
 *                                    └── MCP6004 IN+ → MCP6004 OUT → ESP32 ADCx
 *
 *   MCP6004 #1:  FSR1→GPIO34  FSR2→GPIO35  FSR3→GPIO32
 *   MCP6004 #2:  FSR4→GPIO33  FSR5→GPIO39  FSR6→GPIO36
 *
 * ═══════════════════════════════════════════════════════════════════════
 * Serial commands (115200 baud) — send as a full line, press Enter
 * ═══════════════════════════════════════════════════════════════════════
 *   ?         — help
 *   A1..A6    — select active channel (default A1)
 *   B         — Phase 2: Baseline (10 samples × 30 s) for active channel
 *   L         — Phase 3: Start loading log (10 Hz) for active channel
 *   C         — Phase 3: Capture stable window (send AFTER 10 s creep wait)
 *   R         — Phase 3: Mark start of next rep
 *   G         — Phase 6: Monitor all 6 channels at 10 Hz (Python reads this)
 *   X         — Stop → IDLE
 *
 * ═══════════════════════════════════════════════════════════════════════
 * Output line prefixes
 * ═══════════════════════════════════════════════════════════════════════
 *   #  — human-readable comment
 *   D  — data: D,ts_ms,ch,raw_adc,v_mv,tag       ch = 1..6
 *   S  — stats: S,BASELINE,CH<n>_result,mean_mv,sd_mv,n,PASS|FAIL
 *              S,LOADING,CH<n>_R<rep>_cap,mean_mv,sd_mv,n,OK
 *   M  — monitor (G mode): M,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv
 *
 * ═══════════════════════════════════════════════════════════════════════
 * Workflow
 * ═══════════════════════════════════════════════════════════════════════
 *  For each sensor i (1..6):
 *    1. Select: A<i>
 *    2. Baseline: B  (wait ~5 min)
 *    3. Loading:  L → apply each force → wait 10 s → C → (3 reps, send R between)
 *    4. Stop:     X
 *  Save full serial log to file (use: python -m serial.tools.miniterm --raw
 *                                      /dev/ttyUSB0 115200 | tee session.txt)
 *  Fit:     python fsr406_6ch_phase4.py fit session.txt
 *  Validate: python fsr406_6ch_phase4.py validate --port /dev/ttyUSB0
 *  Monitor:  python fsr406_6ch_phase4.py monitor  --port /dev/ttyUSB0
 *  Recheck:  python fsr406_6ch_phase4.py recheck  --port /dev/ttyUSB0
 */

// ── Timing & integral window ────────────────────────────────────────────────
#define LOG_INTERVAL_MS      100    // 10 Hz
#define INTEGRAL_WINDOW_N    10     // 1.0 s at 10 Hz  (must match Python --window)
#define BASELINE_COUNT       10
#define BASELINE_INTERVAL_S  30
#define BASELINE_SD_THRESH   0.02f  // 2% of 3300 mV = 66 mV
#define CAPTURE_SAMPLES      10
#define CAPTURE_INTERVAL_MS  500

// ── Circuit constants ────────────────────────────────────────────────────────
#define V_CC_MV   3300
#define OVERSAMPLE  64

#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ── Type declarations (must precede all functions so Arduino forward-decls work)
struct FsrCh { const char* label; adc1_channel_t adc_ch; uint8_t gpio; };
struct Stats  { float mean; float sd; };

// ── Channel table ────────────────────────────────────────────────────────────
static const FsrCh CHANS[6] = {
    { "FSR1", ADC1_CHANNEL_6, 34 },   // MCP6004 #1 buffer A
    { "FSR2", ADC1_CHANNEL_7, 35 },   // MCP6004 #1 buffer B
    { "FSR3", ADC1_CHANNEL_4, 32 },   // MCP6004 #1 buffer C
    { "FSR4", ADC1_CHANNEL_5, 33 },   // MCP6004 #2 buffer A
    { "FSR5", ADC1_CHANNEL_3, 39 },   // MCP6004 #2 buffer B
    { "FSR6", ADC1_CHANNEL_0, 36 },   // MCP6004 #2 buffer C
};
static const int N_CHANS = 6;

static esp_adc_cal_characteristics_t adc_chars;

// ── State ────────────────────────────────────────────────────────────────────
enum Phase { IDLE, BASELINE, LOADING, MONITOR };
static Phase    currentPhase = IDLE;
static int      active_ch    = 0;    // 0-based

// ── Baseline state ───────────────────────────────────────────────────────────
static uint32_t bl_samples[BASELINE_COUNT];
static int      bl_idx     = 0;
static uint32_t bl_next_ms = 0;

// ── Loading state ────────────────────────────────────────────────────────────
static uint32_t log_next_ms = 0;
static int      rep_num     = 1;

// ── Integral buffer (active channel only, resets on L) ───────────────────────
static uint32_t v_buf[INTEGRAL_WINDOW_N];
static int      v_buf_head  = 0;
static int      v_buf_count = 0;

// ── Serial line buffer ───────────────────────────────────────────────────────
static char cmd_buf[16];
static int  cmd_len = 0;

// ── ADC helpers ──────────────────────────────────────────────────────────────
static uint32_t readRaw(adc1_channel_t ch) {
    uint32_t s = 0;
    for (int i = 0; i < OVERSAMPLE; i++) {
        s += adc1_get_raw(ch);
        delayMicroseconds(80);
    }
    return s / OVERSAMPLE;
}
static uint32_t toMv(uint32_t raw) {
    return esp_adc_cal_raw_to_voltage(raw, &adc_chars);
}

// ── Stats ────────────────────────────────────────────────────────────────────
static Stats computeStats(uint32_t* arr, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    float m = sum / n, sq = 0;
    for (int i = 0; i < n; i++) sq += (arr[i] - m) * (arr[i] - m);
    return { m, (n > 1) ? sqrtf(sq / (n - 1)) : 0.0f };
}

// ── Integral buffer ──────────────────────────────────────────────────────────
static void bufReset() { v_buf_head = 0; v_buf_count = 0; }
static void bufPush(uint32_t v_mv) {
    v_buf[v_buf_head] = v_mv;
    v_buf_head = (v_buf_head + 1) % INTEGRAL_WINDOW_N;
    if (v_buf_count < INTEGRAL_WINDOW_N) v_buf_count++;
}

// ── Help ─────────────────────────────────────────────────────────────────────
static void printHelp() {
    Serial.println(F("#"));
    Serial.println(F("# ── FSR406 × 6 Calibration Protocol ──────────────────"));
    Serial.printf("# Active channel: FSR%d (GPIO%u)\n",
                  active_ch + 1, CHANS[active_ch].gpio);
    Serial.println(F("#"));
    Serial.println(F("#  A1..A6  Select active channel for Phase 2 / 3"));
    Serial.println(F("#  B       Phase 2: Baseline (10 × 30 s ≈ 5 min)"));
    Serial.println(F("#  L       Phase 3: Start loading log (10 Hz CSV)"));
    Serial.println(F("#  C       Phase 3: Capture stable window (after 10 s creep)"));
    Serial.println(F("#  R       Phase 3: New rep (press after completing one rep)"));
    Serial.println(F("#  G       Phase 6: Monitor all 6 channels at 10 Hz"));
    Serial.println(F("#  X       Stop → IDLE    ?  Help"));
    Serial.println(F("#"));
    Serial.println(F("# D,ts_ms,ch,raw_adc,v_mv,tag"));
    Serial.println(F("# M,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv  (G mode)"));
    Serial.println(F("#"));
}

// ── Command processor ─────────────────────────────────────────────────────────
static void processCommand(const char* buf, int len) {
    if (len == 0) return;
    char cmd = toupper(buf[0]);

    if (cmd == 'A') {
        // Parse digit: accept "A1", "A 1", "a1"
        for (int i = 1; i < len; i++) {
            if (isdigit(buf[i])) {
                int ch = buf[i] - '0';
                if (ch >= 1 && ch <= N_CHANS) {
                    active_ch    = ch - 1;
                    currentPhase = IDLE;
                    Serial.printf("# Active → FSR%d (GPIO%u)\n",
                                  ch, CHANS[active_ch].gpio);
                } else {
                    Serial.printf("# Channel must be 1-%d\n", N_CHANS);
                }
                return;
            }
        }
        Serial.println(F("# Usage: A1 .. A6"));
        return;
    }

    switch (cmd) {
        case '?': printHelp();                                              break;

        case 'B':
            currentPhase = BASELINE;
            bl_idx       = 0;
            bl_next_ms   = millis();
            Serial.printf("# ── Phase 2 BASELINE — FSR%d ───────────────────────\n",
                          active_ch + 1);
            Serial.printf("# %d samples × %d s — sensor warm, no extra force\n",
                          BASELINE_COUNT, BASELINE_INTERVAL_S);
            Serial.println(F("# D,ts_ms,ch,raw_adc,v_mv,tag"));
            break;

        case 'L':
            currentPhase = LOADING;
            rep_num      = 1;
            log_next_ms  = millis();
            bufReset();
            Serial.printf("# ── Phase 3 LOADING — FSR%d ─────────────────────────\n",
                          active_ch + 1);
            Serial.println(F("# 10 Hz CSV.  C=capture after 10 s  R=new rep  X=stop"));
            Serial.printf("# Rep %d\n", rep_num);
            Serial.println(F("# D,ts_ms,ch,raw_adc,v_mv,tag"));
            break;

        case 'C':
            if (currentPhase != LOADING) {
                Serial.println(F("# C only valid during Phase 3 (send L first)"));
                break;
            }
            {
                Serial.printf("# Capture: %d × %d ms — CH%d Rep%d\n",
                              CAPTURE_SAMPLES, CAPTURE_INTERVAL_MS,
                              active_ch + 1, rep_num);
                uint32_t cap[CAPTURE_SAMPLES];
                for (int i = 0; i < CAPTURE_SAMPLES; i++) {
                    uint32_t raw  = readRaw(CHANS[active_ch].adc_ch);
                    uint32_t v_mv = toMv(raw);
                    bufPush(v_mv);
                    cap[i] = v_mv;
                    Serial.printf("D,%lu,%d,%lu,%lu,CH%d_CAP_R%d_%02d\n",
                                  millis(), active_ch + 1, raw, v_mv,
                                  active_ch + 1, rep_num, i + 1);
                    if (i < CAPTURE_SAMPLES - 1) delay(CAPTURE_INTERVAL_MS);
                }
                Stats s = computeStats(cap, CAPTURE_SAMPLES);
                Serial.printf("S,LOADING,CH%d_R%d_cap,%.1f,%.1f,%d,OK\n",
                              active_ch + 1, rep_num, s.mean, s.sd, CAPTURE_SAMPLES);
                Serial.printf("# CH%d Rep%d mean=%.1f mV  SD=%.1f mV\n",
                              active_ch + 1, rep_num, s.mean, s.sd);
                log_next_ms = millis() + LOG_INTERVAL_MS;
            }
            break;

        case 'R':
            if (currentPhase != LOADING) {
                Serial.println(F("# R only valid during Phase 3"));
                break;
            }
            rep_num++;
            Serial.printf("# ── Rep %d ──\n", rep_num);
            break;

        case 'D': {
            // Single read of all 6 channels — used by calibrate.py
            Serial.print("DATA");
            for (int i = 0; i < N_CHANS; i++) {
                Serial.print(',');
                Serial.print(toMv(readRaw(CHANS[i].adc_ch)));
            }
            Serial.println();
            break;
        }

        case 'G':
            currentPhase = MONITOR;
            log_next_ms  = millis();
            Serial.println(F("# ── Phase 6 MONITOR — all 6 channels at 10 Hz ────"));
            Serial.println(F("# Run Python: python fsr406_6ch_phase4.py monitor --port ..."));
            Serial.println(F("# M,ts_ms,v0_mv,v1_mv,v2_mv,v3_mv,v4_mv,v5_mv"));
            Serial.println(F("# X to stop"));
            break;

        case 'X':
            currentPhase = IDLE;
            Serial.println(F("# Stopped → IDLE"));
            break;

        default:
            Serial.printf("# Unknown command '%c' (? for help)\n", cmd);
            break;
    }
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);
    delay(2000);

    adc1_config_width(ADC_WIDTH_BIT_12);
    for (int i = 0; i < N_CHANS; i++)
        adc1_config_channel_atten(CHANS[i].adc_ch, ADC_ATTEN_DB_11);

    esp_adc_cal_value_t cal = esp_adc_cal_characterize(
        ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);

    Serial.println(F("\n# ══════════════════════════════════════════════════════"));
    Serial.println(F("#   FSR406 × 6  Calibration Protocol  (6-Channel)"));
    Serial.println(F("# ══════════════════════════════════════════════════════"));
    Serial.printf( "# ADC cal: %s\n",
        cal == ESP_ADC_CAL_VAL_EFUSE_VREF ? "eFuse Vref (best)" :
        cal == ESP_ADC_CAL_VAL_EFUSE_TP   ? "eFuse two-point"   :
                                             "default 1100 mV");
    printHelp();
}

// ── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    // Non-blocking serial line reader
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmd_len > 0) {
                cmd_buf[cmd_len] = '\0';
                processCommand(cmd_buf, cmd_len);
                cmd_len = 0;
            }
        } else if (cmd_len < (int)(sizeof(cmd_buf) - 1)) {
            cmd_buf[cmd_len++] = c;
        }
    }

    // State machine
    switch (currentPhase) {

        case BASELINE: {
            if (millis() < bl_next_ms) break;
            bl_next_ms += (uint32_t)BASELINE_INTERVAL_S * 1000;

            uint32_t raw  = readRaw(CHANS[active_ch].adc_ch);
            uint32_t v_mv = toMv(raw);
            bl_samples[bl_idx] = v_mv;

            Serial.printf("D,%lu,%d,%lu,%lu,CH%d_BL_%02d\n",
                          millis(), active_ch + 1, raw, v_mv,
                          active_ch + 1, bl_idx + 1);

            int left = BASELINE_COUNT - bl_idx - 1;
            if (left > 0)
                Serial.printf("# %d remaining, next in %d s\n", left, BASELINE_INTERVAL_S);
            bl_idx++;

            if (bl_idx >= BASELINE_COUNT) {
                Stats s      = computeStats(bl_samples, BASELINE_COUNT);
                float thresh = BASELINE_SD_THRESH * V_CC_MV;
                bool  pass   = (s.sd <= thresh);
                Serial.printf("S,BASELINE,CH%d_result,%.1f,%.1f,%d,%s\n",
                              active_ch + 1, s.mean, s.sd, BASELINE_COUNT,
                              pass ? "PASS" : "FAIL");
                Serial.printf("# CH%d V₀=%.1f mV  SD=%.1f mV  thresh=%.1f mV → %s\n",
                              active_ch + 1, s.mean, s.sd, thresh,
                              pass ? "PASS" : "FAIL");
                if (!pass)
                    Serial.println(F("# FAIL: check strap tension, housing, preload"));
                currentPhase = IDLE;
            }
            break;
        }

        case LOADING: {
            if (millis() < log_next_ms) break;
            log_next_ms += LOG_INTERVAL_MS;

            uint32_t raw  = readRaw(CHANS[active_ch].adc_ch);
            uint32_t v_mv = toMv(raw);
            bufPush(v_mv);
            Serial.printf("D,%lu,%d,%lu,%lu,CH%d_LOG_R%d\n",
                          millis(), active_ch + 1, raw, v_mv,
                          active_ch + 1, rep_num);
            break;
        }

        case MONITOR: {
            if (millis() < log_next_ms) break;
            log_next_ms += LOG_INTERVAL_MS;

            Serial.printf("M,%lu", millis());
            for (int i = 0; i < N_CHANS; i++) {
                uint32_t raw  = readRaw(CHANS[i].adc_ch);
                uint32_t v_mv = toMv(raw);
                Serial.printf(",%lu", v_mv);
            }
            Serial.println();
            break;
        }

        default: break;
    }
}
