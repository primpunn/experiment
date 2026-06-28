/*
 * FSR406 Calibration Protocol — Phases 2–5
 * Force model: Option B — 4th-order polynomial + linearly-weighted moving integral
 *
 * Hardware (same circuit as connection_test):
 *   3.3V → FSR406 → Node A ──┬── R_M(10 kΩ) → GND
 *                             └── MCP6004 IN+ → MCP6004 OUT → GPIO34
 *
 * ══════════════════════════════════════════════════════════════════════
 * Serial commands (115200 baud)
 * ══════════════════════════════════════════════════════════════════════
 *   ?   — show this help
 *   B   — Phase 2: Baseline recording (10 samples × 30 s ≈ 5 min)
 *   L   — Phase 3: Loading log start (10 Hz continuous CSV)
 *   C   — Phase 3: Capture stable window (send AFTER 10 s creep wait)
 *   R   — Phase 3: Mark start of next Rep
 *   V   — Phase 5: Validation (needs CALIB_OB_* constants filled in)
 *   X   — Stop → IDLE
 *   ?   — Help
 *
 * ══════════════════════════════════════════════════════════════════════
 * Output line prefixes
 * ══════════════════════════════════════════════════════════════════════
 *   #   — human-readable comment
 *   D   — data CSV  (see per-phase headers)
 *   S   — stats CSV
 *
 * ══════════════════════════════════════════════════════════════════════
 * Calibration workflow
 * ══════════════════════════════════════════════════════════════════════
 *   Phase 2 (B)  →  record V₀  →  set CALIB_V0_MV
 *   Phase 3 (L/C/R)  →  save log file
 *   Phase 4 (Python)  →  fsr406_phase4_calibration.py  →  get 9 coefficients
 *   Phase 5 (V)  →  set all CALIB_OB_* + CALIB_V0_MV  →  reflash  →  validate
 *
 * ══════════════════════════════════════════════════════════════════════
 * Phase 4 calibration constants — fill in after offline curve fitting
 * ══════════════════════════════════════════════════════════════════════
 *   Model: F [N] = a0 + a1·V + a2·V² + a3·V³ + a4·V⁴
 *                     + b1·I + b2·I² + b3·I³ + b4·I⁴
 *   V = voltage in Volts (v_mv / 1000)
 *   I = linearly-weighted integral of V over past INTEGRAL_WINDOW_S seconds (V·s)
 */
#define CALIB_V0_MV      0        // baseline mV — from Phase 2

#define CALIB_OB_A0      0.0f     // constant term
#define CALIB_OB_A1      0.0f     // linear voltage
#define CALIB_OB_A2      0.0f     // quadratic
#define CALIB_OB_A3      0.0f     // cubic
#define CALIB_OB_A4      0.0f     // quartic

#define CALIB_OB_B1      0.0f     // linear integral
#define CALIB_OB_B2      0.0f     // quadratic
#define CALIB_OB_B3      0.0f     // cubic
#define CALIB_OB_B4      0.0f     // quartic

// ── Circuit & sampling constants ─────────────────────────────────────
#define FSR_ADC_CHANNEL     ADC1_CHANNEL_6   // GPIO34
#define V_CC_MV             3300
#define OVERSAMPLE          64
#define LOG_INTERVAL_MS     100    // 10 Hz — matches Python WINDOW_S assumption

// ── Phase 2 — Baseline ───────────────────────────────────────────────
#define BASELINE_COUNT      10
#define BASELINE_INTERVAL_S 30
#define BASELINE_SD_THRESH  0.02f  // 2% of full-scale (66 mV)

// ── Phase 3 — Loading log ────────────────────────────────────────────
#define CAPTURE_SAMPLES     10
#define CAPTURE_INTERVAL_MS 500

// ── Circular buffer for moving integral ──────────────────────────────
// 1.0 s window at 10 Hz = 10 samples (must match Python --window default)
#define INTEGRAL_WINDOW_S   1.0f
#define INTEGRAL_WINDOW_N   10          // = INTEGRAL_WINDOW_S / (LOG_INTERVAL_MS/1000)

// ────────────────────────────────────────────────────────────────────
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include <math.h>

static esp_adc_cal_characteristics_t adc_chars;

enum Phase { IDLE, BASELINE, LOADING, VALIDATION };
static Phase currentPhase = IDLE;

// ── Baseline state ───────────────────────────────────────────────────
static uint32_t bl_samples[BASELINE_COUNT];
static int      bl_idx     = 0;
static uint32_t bl_next_ms = 0;

// ── Loading state ────────────────────────────────────────────────────
static uint32_t log_next_ms = 0;
static int      rep_num     = 1;

// ── Circular buffer (mV values for integral) ─────────────────────────
static uint32_t v_buf[INTEGRAL_WINDOW_N];
static int      v_buf_head  = 0;   // next write position
static int      v_buf_count = 0;   // how many valid entries (caps at INTEGRAL_WINDOW_N)

static void bufPush(uint32_t v_mv) {
    v_buf[v_buf_head] = v_mv;
    v_buf_head = (v_buf_head + 1) % INTEGRAL_WINDOW_N;
    if (v_buf_count < INTEGRAL_WINDOW_N) v_buf_count++;
}

/*
 * Linearly-weighted moving integral of V over the last INTEGRAL_WINDOW_N samples.
 * Weight rises linearly from 0 (oldest sample) to 1 (most recent sample).
 * Uses trapezoidal rule with uniform dt = LOG_INTERVAL_MS.
 * Returns I in Volt·seconds.
 */
static float computeIntegral() {
    int N = v_buf_count;
    if (N < 2) return 0.0f;

    float dt = LOG_INTERVAL_MS / 1000.0f;   // seconds per sample
    float sum = 0.0f;

    // Collect N samples oldest-first into a local array
    // v_buf_head points to the NEXT write slot → (head - 1) is the newest.
    // Oldest is at (head - count + INTEGRAL_WINDOW_N) % INTEGRAL_WINDOW_N.
    for (int j = 0; j < N - 1; j++) {
        // j=0 is oldest sample, j=N-1 is newest
        int idx0 = (v_buf_head - N + j     + INTEGRAL_WINDOW_N * 2) % INTEGRAL_WINDOW_N;
        int idx1 = (v_buf_head - N + j + 1 + INTEGRAL_WINDOW_N * 2) % INTEGRAL_WINDOW_N;

        float v0 = v_buf[idx0] / 1000.0f;   // mV → V
        float v1 = v_buf[idx1] / 1000.0f;

        // Linear weights: w rises from 0 at j=0 to 1 at j=N-1
        float w0 = (float)j       / (float)(N - 1);
        float w1 = (float)(j + 1) / (float)(N - 1);

        // Trapezoidal: 0.5 * (w0*v0 + w1*v1) * dt
        sum += 0.5f * (w0 * v0 + w1 * v1) * dt;
    }
    return sum;
}

// ── Hall model evaluation (Horner's method for numerical stability) ───
static float evalHallModel(float v_V, float I_Vs) {
    // F = a0 + V*(a1 + V*(a2 + V*(a3 + V*a4)))
    //       + I*(b1 + I*(b2 + I*(b3 + I*b4)))
    float poly = CALIB_OB_A0
               + v_V * (CALIB_OB_A1
               + v_V * (CALIB_OB_A2
               + v_V * (CALIB_OB_A3
               + v_V *  CALIB_OB_A4)));

    float intg = I_Vs * (CALIB_OB_B1
               + I_Vs * (CALIB_OB_B2
               + I_Vs * (CALIB_OB_B3
               + I_Vs *  CALIB_OB_B4)));

    float F = poly + intg;
    return (F > 0.0f) ? F : 0.0f;   // clamp to zero (no tension)
}

static bool optionBReady() {
    // True if any non-zero coefficient is set
    return (CALIB_OB_A0 != 0.0f || CALIB_OB_A1 != 0.0f ||
            CALIB_OB_A2 != 0.0f || CALIB_OB_A3 != 0.0f ||
            CALIB_OB_A4 != 0.0f || CALIB_OB_B1 != 0.0f ||
            CALIB_OB_B2 != 0.0f || CALIB_OB_B3 != 0.0f ||
            CALIB_OB_B4 != 0.0f);
}

// ── ADC helpers ──────────────────────────────────────────────────────
static uint32_t readRaw() {
    uint32_t s = 0;
    for (int i = 0; i < OVERSAMPLE; i++) {
        s += adc1_get_raw(FSR_ADC_CHANNEL);
        delayMicroseconds(80);
    }
    return s / OVERSAMPLE;
}

static uint32_t rawToMv(uint32_t raw) {
    return esp_adc_cal_raw_to_voltage(raw, &adc_chars);
}

// ── Stats ─────────────────────────────────────────────────────────────
struct Stats { float mean; float sd; };
static Stats computeStats(uint32_t* arr, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    float mean = sum / n, sq = 0;
    for (int i = 0; i < n; i++) sq += (arr[i] - mean) * (arr[i] - mean);
    return { mean, (n > 1) ? sqrtf(sq / (n - 1)) : 0.0f };
}

// ── Help ──────────────────────────────────────────────────────────────
static void printHelp() {
    Serial.println(F("#"));
    Serial.println(F("# ── FSR406 Calibration Protocol ────────────────────"));
    Serial.println(F("#  B  Phase 2: Baseline (10 × 30 s, ~5 min)"));
    Serial.println(F("#  L  Phase 3: Loading log start (10 Hz CSV)"));
    Serial.println(F("#  C  Phase 3: Capture stable window (after 10 s wait)"));
    Serial.println(F("#  R  Phase 3: New rep"));
    Serial.println(F("#  V  Phase 5: Validation (needs CALIB_OB_* constants)"));
    Serial.println(F("#  X  Stop → IDLE    ?  Help"));
    Serial.println(F("#"));
    Serial.println(F("# Data:  D,ts_ms,raw,v_mv,[I_Vs,f_N,]tag"));
    Serial.println(F("# Stats: S,phase,label,mean_mv,sd_mv,n,PASS|FAIL|OK"));
    Serial.printf( "# Integral window: %.1f s (%d samples at 10 Hz)\n",
                   INTEGRAL_WINDOW_S, INTEGRAL_WINDOW_N);
    Serial.println(F("#"));
}

// ════════════════════════════════════════════════════════════════════
// Phase 2 — Baseline
// ════════════════════════════════════════════════════════════════════
static void startBaseline() {
    currentPhase = BASELINE;
    bl_idx       = 0;
    bl_next_ms   = millis();
    Serial.println(F("# ── Phase 2 BASELINE ──────────────────────────────"));
    Serial.printf( "# %d samples × %d s  (~%d min)\n",
                   BASELINE_COUNT, BASELINE_INTERVAL_S,
                   (BASELINE_COUNT - 1) * BASELINE_INTERVAL_S / 60);
    Serial.println(F("# Sensor must be warm, no external force on it."));
    Serial.println(F("# D,timestamp_ms,raw_adc,v_mv,tag"));
    Serial.println(F("#"));
}

static void runBaseline() {
    if (millis() < bl_next_ms) return;
    bl_next_ms += (uint32_t)BASELINE_INTERVAL_S * 1000;

    uint32_t raw  = readRaw();
    uint32_t v_mv = rawToMv(raw);
    bl_samples[bl_idx] = v_mv;
    Serial.printf("D,%lu,%lu,%lu,BL_%02d\n", millis(), raw, v_mv, bl_idx + 1);

    int left = BASELINE_COUNT - bl_idx - 1;
    if (left > 0)
        Serial.printf("# %d sample(s) remaining — next in %d s\n",
                      left, BASELINE_INTERVAL_S);
    bl_idx++;

    if (bl_idx >= BASELINE_COUNT) {
        Stats s        = computeStats(bl_samples, BASELINE_COUNT);
        float thresh   = BASELINE_SD_THRESH * V_CC_MV;
        bool  pass     = (s.sd <= thresh);
        Serial.println(F("#"));
        Serial.println(F("# ── Phase 2 BASELINE complete ─────────────────────"));
        Serial.printf( "S,BASELINE,result,%.1f,%.1f,%d,%s\n",
                       s.mean, s.sd, BASELINE_COUNT, pass ? "PASS" : "FAIL");
        Serial.printf( "# Mean V₀ = %.1f mV   SD = %.1f mV   threshold = %.1f mV → %s\n",
                       s.mean, s.sd, thresh, pass ? "PASS" : "FAIL");
        if (pass)
            Serial.printf("# → Set  #define CALIB_V0_MV  %u\n",
                          (unsigned)roundf(s.mean));
        else
            Serial.println(F("# FAIL: check strap tension, housing seating, unusual preload"));
        Serial.println(F("#"));
        currentPhase = IDLE;
    }
}

// ════════════════════════════════════════════════════════════════════
// Phase 3 — Loading log
// ════════════════════════════════════════════════════════════════════
static void startLoading() {
    currentPhase = LOADING;
    rep_num      = 1;
    log_next_ms  = millis();
    v_buf_head   = 0;
    v_buf_count  = 0;
    Serial.println(F("# ── Phase 3 LOADING ───────────────────────────────"));
    Serial.println(F("# 10 Hz CSV log.  C = capture window,  R = new rep,  X = stop"));
    Serial.printf( "# Rep %d\n", rep_num);
    Serial.println(F("# D,timestamp_ms,raw_adc,v_mv,tag"));
    Serial.println(F("#"));
}

static void runLoadingLog() {
    if (millis() < log_next_ms) return;
    log_next_ms += LOG_INTERVAL_MS;

    uint32_t raw  = readRaw();
    uint32_t v_mv = rawToMv(raw);
    bufPush(v_mv);
    Serial.printf("D,%lu,%lu,%lu,LOG_R%d\n", millis(), raw, v_mv, rep_num);
}

// Blocking capture window — send after the 10 s creep stabilisation
static void runCapture() {
    Serial.printf("# Capture window: %d × %d ms  (Rep %d)\n",
                  CAPTURE_SAMPLES, CAPTURE_INTERVAL_MS, rep_num);

    uint32_t cap[CAPTURE_SAMPLES];
    for (int i = 0; i < CAPTURE_SAMPLES; i++) {
        uint32_t raw  = readRaw();
        uint32_t v_mv = rawToMv(raw);
        bufPush(v_mv);
        cap[i] = v_mv;
        Serial.printf("D,%lu,%lu,%lu,CAP_R%d_%02d\n",
                      millis(), raw, v_mv, rep_num, i + 1);
        if (i < CAPTURE_SAMPLES - 1) delay(CAPTURE_INTERVAL_MS);
    }

    Stats s = computeStats(cap, CAPTURE_SAMPLES);
    Serial.printf("S,LOADING,R%d_cap,%.1f,%.1f,%d,OK\n",
                  rep_num, s.mean, s.sd, CAPTURE_SAMPLES);
    Serial.printf("# Rep %d mean V = %.1f mV  SD = %.1f mV\n",
                  rep_num, s.mean, s.sd);
    Serial.println(F("# Record this V in your (F_N, V_mV) table, then next force level."));
    Serial.println(F("#"));

    log_next_ms = millis() + LOG_INTERVAL_MS;
}

// ════════════════════════════════════════════════════════════════════
// Phase 5 — Validation
// ════════════════════════════════════════════════════════════════════
static void startValidation() {
    if (!optionBReady()) {
        Serial.println(F("# ERROR: all CALIB_OB_* are 0."));
        Serial.println(F("#   1. Run Phase 3 and save the serial log."));
        Serial.println(F("#   2. Run: python fsr406_phase4_calibration.py <log.txt>"));
        Serial.println(F("#   3. Copy the printed #define lines into this file."));
        Serial.println(F("#   4. Reflash, then send V again."));
        return;
    }

    currentPhase = VALIDATION;
    v_buf_head   = 0;
    v_buf_count  = 0;

    Serial.println(F("# ── Phase 5 VALIDATION ────────────────────────────"));
    Serial.println(F("# Option B: 4th-order polynomial + moving integral"));
    Serial.printf( "# F = a0+a1V+a2V²+a3V³+a4V⁴ + b1I+b2I²+b3I³+b4I⁴\n");
    Serial.printf( "# V₀ = %d mV   window = %.1f s\n",
                   CALIB_V0_MV, INTEGRAL_WINDOW_S);
    Serial.println(F("# Warming up buffer ..."));

    // Fill buffer before showing output
    for (int i = 0; i < INTEGRAL_WINDOW_N; i++) {
        uint32_t v_mv = rawToMv(readRaw());
        bufPush(v_mv);
        delay(LOG_INTERVAL_MS);
    }

    Serial.println(F("# Ready. Apply known forces and compare F_pred vs F_true."));
    Serial.println(F("# Acceptance: RMSE < 3 N,  |error| < 10% per point.  X to stop."));
    Serial.println(F("# D,timestamp_ms,raw_adc,v_mv,I_Vs,f_pred_N,tag"));
    Serial.println(F("#"));
}

static void runValidation() {
    uint32_t raw  = readRaw();
    uint32_t v_mv = rawToMv(raw);
    bufPush(v_mv);

    float v_V  = v_mv / 1000.0f;
    float I    = computeIntegral();
    float f_N  = evalHallModel(v_V, I);

    Serial.printf("D,%lu,%lu,%lu,%.4f,%.3f,VAL_B\n",
                  millis(), raw, v_mv, I, f_N);
    delay(LOG_INTERVAL_MS);
}

// ════════════════════════════════════════════════════════════════════
// Setup / Loop
// ════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(800);

    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(FSR_ADC_CHANNEL, ADC_ATTEN_DB_11);
    esp_adc_cal_value_t cal = esp_adc_cal_characterize(
        ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);

    Serial.println(F("\n# ══════════════════════════════════════════════════"));
    Serial.println(F("#   FSR406 Calibration Protocol — Option B ready"));
    Serial.println(F("# ══════════════════════════════════════════════════"));
    Serial.printf( "# ADC cal: %s\n",
        cal == ESP_ADC_CAL_VAL_EFUSE_VREF ? "eFuse Vref (best)" :
        cal == ESP_ADC_CAL_VAL_EFUSE_TP   ? "eFuse two-point"   :
                                             "default 1100 mV");
    Serial.printf("# CALIB_V0_MV = %d   Option B: %s\n",
                  CALIB_V0_MV, optionBReady() ? "READY" : "not set (run Phase 4 first)");
    printHelp();
}

void loop() {
    if (Serial.available()) {
        char cmd = (char)toupper(Serial.read());
        while (Serial.available()) Serial.read();

        switch (cmd) {
            case '?':                    printHelp();       break;
            case 'B':                    startBaseline();   break;
            case 'L':                    startLoading();    break;
            case 'C':
                if (currentPhase == LOADING) runCapture();
                else Serial.println(F("# C only valid during Phase 3 (send L first)"));
                break;
            case 'R':
                if (currentPhase == LOADING) {
                    rep_num++;
                    Serial.printf("# ── Rep %d ──\n", rep_num);
                } else {
                    Serial.println(F("# R only valid during Phase 3 (send L first)"));
                }
                break;
            case 'V':                    startValidation(); break;
            case 'X':
                currentPhase = IDLE;
                Serial.println(F("# Stopped → IDLE"));
                break;
            default:
                Serial.printf("# Unknown command '%c'  (send ? for help)\n", cmd);
                break;
        }
    }

    switch (currentPhase) {
        case BASELINE:   runBaseline();    break;
        case LOADING:    runLoadingLog();  break;
        case VALIDATION: runValidation();  break;
        default:                           break;
    }
}
