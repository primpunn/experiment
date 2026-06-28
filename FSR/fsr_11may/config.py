N_SENSORS    = 6
SENSOR_NAMES = ["FSR1", "FSR2", "FSR3", "FSR4", "FSR5", "FSR6"]

# Two arrays on body
SENSOR_GROUPS = {
    "calf": [0, 1, 2, 3],  # FSR1–FSR4 arranged around the calf
    "foot": [4, 5],         # FSR5–FSR6 arranged around the foot
}

# Hardware
V_CC_MV  = 3300.0   # mV (esp_adc_cal reference)
BAUD     = 115200
FS       = 100      # Hz
DT       = 1.0 / FS

# Calibration load levels in grams (log-biased spacing, 9 levels, max 2150 g)
LOAD_G   = [0, 80, 200, 390, 685, 1000, 1350, 1750, 2170]
UNLOAD_G = [2170, 1750, 1350, 1000, 685, 390, 200, 80, 0]
N_REPS   = 1

CREEP_WAIT_S       = 5    # seconds to wait for creep stabilisation after placing weight
POST_UNLOAD_WAIT_S = 5    # seconds to wait after removing weight
CAPTURE_N          = 5    # samples to collect in the stable window per level
CAPTURE_INTERVAL_S = 0.2  # spacing between capture samples → 1 s total window

# Baseline (PDF Phase 1)
BASELINE_N_READINGS  = 10   # number of baseline readings per sensor
BASELINE_INTERVAL_S  = 30   # seconds between consecutive baseline readings
BASELINE_SD_MAX_PCT  = 2.0  # SD must be < 2% of full scale to pass

# Hall 2008 moving integral (PDF Section 6)
WINDOW_SEC = 0.5
N_WINDOW   = int(FS * WINDOW_SEC)  # 50 samples

# Validation (PDF Phase 3) — grams NOT used in calibration set
VAL_LOADS_G      = [170, 570, 880, 1220, 1530]
RMSE_MAX_N       = 3.0
PCT_ERROR_MAX    = 10.0
RMSE_TOTAL_MAX_N = 5.0

# Directories
CALIBRATIONS_DIR = "calibrations"   # each run → calibrations/YYYY-MM-DD_HH-MM-SS/
SESSIONS_DIR     = "sessions"       # each run → sessions/YYYY-MM-DD_HH-MM-SS.csv
