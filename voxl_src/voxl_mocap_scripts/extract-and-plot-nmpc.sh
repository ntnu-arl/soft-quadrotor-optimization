#!/bin/bash
# Extract NMPC solver CSVs and latest PX4 ULog from VOXL (SSH/SCP),
# then run extract_px4_motor_log.py, extract_px4_flight_log.py, and plot_hitl.py.
#
# Usage:
#   ./extract-and-plot-nmpc.sh                         # pull from VOXL + plot
#   ./extract-and-plot-nmpc.sh --log-dir <dir>         # pull into specific dir + plot
#   ./extract-and-plot-nmpc.sh --replot <dir>          # skip SSH, re-plot existing log dir
#   ./extract-and-plot-nmpc.sh --skip_non_essential    # pull and process data without plotting
#   ./extract-and-plot-nmpc.sh --from-raw [--replot <dir>]  # plot from raw logs, keeping all intermediates in memory

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch-relay.sh"
load_voxl_connection_config

SCP="sshpass -p ${VOXL_PASS} scp -q -o StrictHostKeyChecking=no"
SSH="sshpass -p ${VOXL_PASS} ssh -o StrictHostKeyChecking=no ${VOXL_USER}@${VOXL_IP}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NMPC_CONFIG_BASENAME="nmpc_controller_cfg.yaml"
NMPC_CONFIG_REMOTE="/home/root/${NMPC_CONFIG_BASENAME}"
LOG_DIR=""
REPLOT=0
SKIP_NON_ESSENTIAL=0
FROM_RAW=0
RAW_QUALISYS_CSV_NAME="qualisys_odom.csv"
RAW_QUALISYS_CSV_REMOTE="/tmp/${RAW_QUALISYS_CSV_NAME}"
BATTERY_LOG_NAME="battery_status_log.csv"
BATTERY_LOG_REMOTE="/tmp/${BATTERY_LOG_NAME}"
QUALISYS_MATCH_MAX_POSITION_ERROR_M="0.0001"
DOMINANT_COST_WINDOW_START_S="12.0"
DOMINANT_COST_WINDOW_END_S="15.5"
DOMINANT_COST_TOP_COUNT="12"
DOMINANT_COST_DISTRIBUTION_SLICE_COUNT="20"

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir) LOG_DIR=$2; shift 2 ;;
        --replot)  REPLOT=1; LOG_DIR=$2; shift 2 ;;
        --skip_non_essential) SKIP_NON_ESSENTIAL=1; shift ;;
        --from-raw) FROM_RAW=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$SCRIPT_DIR/../results/real_world_experiments/voxl_mocap/$(date +%Y_%m_%d_%H%M%S)"
fi

if [ -f "$LOG_DIR" ] && [ $REPLOT -eq 0 ]; then
    echo "ERROR: $LOG_DIR is a file; use --replot to plot from an existing run archive"
    exit 1
fi

RUN_ARCHIVE_MODE=0
if [ $REPLOT -eq 1 ] && [ -f "$LOG_DIR" ]; then
    case "$LOG_DIR" in
        *.tar.zst|*.tar.xz|*.tar.gz|*.tgz|*.tar) RUN_ARCHIVE_MODE=1 ;;
        *) echo "ERROR: --replot target is a file but not a supported run archive: $LOG_DIR"; exit 1 ;;
    esac
fi
if [ $RUN_ARCHIVE_MODE -eq 1 ]; then
    FROM_RAW=1
fi

if [ $RUN_ARCHIVE_MODE -eq 0 ]; then
    mkdir -p "$LOG_DIR"
fi

if [ $REPLOT -eq 0 ]; then
    if ! ${SSH} "test -f ${NMPC_CONFIG_REMOTE}"; then
        echo "ERROR: Missing remote NMPC config at ${NMPC_CONFIG_REMOTE}"
        exit 1
    fi
    echo "Extracting ${NMPC_CONFIG_BASENAME}"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:${NMPC_CONFIG_REMOTE}" "$LOG_DIR/$NMPC_CONFIG_BASENAME"
fi

if [ $REPLOT -eq 0 ]; then
    RUN_MANIFEST_REMOTE="/tmp/nmpc_solver_current_run.txt"
    if ! ${SSH} "test -f ${RUN_MANIFEST_REMOTE}"; then
        echo "ERROR: Missing ${RUN_MANIFEST_REMOTE} on VOXL"
        echo "  The nmpc-pipe-solver must publish the active run manifest before extraction."
        exit 1
    fi

    REMOTE_SOLVER_CSV=$(${SSH} "awk -F= '/^solver_csv=/{print \$2}' ${RUN_MANIFEST_REMOTE}" | tr -d '\r\n')

    if [ -z "$REMOTE_SOLVER_CSV" ]; then
        echo "ERROR: ${RUN_MANIFEST_REMOTE} is incomplete"
        exit 1
    fi

    if ! ${SSH} "test -f ${REMOTE_SOLVER_CSV}"; then
        echo "ERROR: Manifest solver CSV does not exist: ${REMOTE_SOLVER_CSV}"
        exit 1
    fi

    mapfile -t REMOTE_SOLVER_CSVS < <(${SSH} "find /tmp -maxdepth 1 -type f -name 'nmpc_solver_data_*.csv' | sort" | tr -d '\r')
    if [ ${#REMOTE_SOLVER_CSVS[@]} -eq 0 ]; then
        echo "ERROR: No nmpc_solver_data_*.csv files found on VOXL under /tmp"
        exit 1
    fi

    for REMOTE_SOLVER_CSV_PATH in "${REMOTE_SOLVER_CSVS[@]}"; do
        SOLVER_BASENAME=$(basename "$REMOTE_SOLVER_CSV_PATH")
        echo "Extracting ${SOLVER_BASENAME}"
        ${SCP} "${VOXL_USER}@${VOXL_IP}:${REMOTE_SOLVER_CSV_PATH}" "$LOG_DIR/${SOLVER_BASENAME}"
    done
    echo "Extracting nmpc_solver_current_run.txt"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:${RUN_MANIFEST_REMOTE}" "$LOG_DIR/nmpc_solver_current_run.txt"
    echo "Extracting nmpc_solver.log"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:/tmp/nmpc_solver.log" "$LOG_DIR/nmpc_solver.log"

    PX4_LOG_REMOTE=$(${SSH} "find /data/px4/log -type f -name '*.ulg' | sort | tail -n 1" | tr -d '\r\n')

    if [ -z "$PX4_LOG_REMOTE" ]; then
        echo "ERROR: No PX4 .ulg file found on VOXL under /data/px4/log"
        exit 1
    fi

    echo "Extracting px4_run.ulg"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:${PX4_LOG_REMOTE}" "$LOG_DIR/px4_run.ulg"

    if ! ${SSH} "test -f ${RAW_QUALISYS_CSV_REMOTE}"; then
        echo "ERROR: Missing raw Qualisys CSV on VOXL at ${RAW_QUALISYS_CSV_REMOTE}"
        exit 1
    fi

    echo "Extracting ${RAW_QUALISYS_CSV_NAME}"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:${RAW_QUALISYS_CSV_REMOTE}" "$LOG_DIR/${RAW_QUALISYS_CSV_NAME}"

    if ! ${SSH} "test -f ${BATTERY_LOG_REMOTE}"; then
        echo "ERROR: Missing battery status log on VOXL at ${BATTERY_LOG_REMOTE}"
        exit 1
    fi

    echo "Extracting ${BATTERY_LOG_NAME}"
    ${SCP} "${VOXL_USER}@${VOXL_IP}:${BATTERY_LOG_REMOTE}" "$LOG_DIR/${BATTERY_LOG_NAME}"

    echo "Extracting px4_motor_log.csv"
    python3 "$SCRIPT_DIR/extract_px4_motor_log.py" "$LOG_DIR/px4_run.ulg" "$LOG_DIR/px4_motor_log.csv" >/dev/null

    echo "Extracting px4_flight_log.csv"
    python3 "$SCRIPT_DIR/extract_px4_flight_log.py" "$LOG_DIR/px4_run.ulg" "$LOG_DIR/px4_flight_log.csv" >/dev/null
fi

if [ $RUN_ARCHIVE_MODE -eq 0 ]; then
if [ ! -f "$LOG_DIR/${RAW_QUALISYS_CSV_NAME}" ]; then
    echo "ERROR: Missing $LOG_DIR/${RAW_QUALISYS_CSV_NAME}"
    exit 1
fi
if [ ! -f "$LOG_DIR/px4_motor_log.csv" ]; then
    echo "ERROR: Missing $LOG_DIR/px4_motor_log.csv"
    exit 1
fi
if [ ! -f "$LOG_DIR/px4_flight_log.csv" ]; then
    echo "ERROR: Missing $LOG_DIR/px4_flight_log.csv"
    exit 1
fi
if [ $REPLOT -eq 0 ] && [ ! -f "$LOG_DIR/${BATTERY_LOG_NAME}" ]; then
    echo "ERROR: Missing $LOG_DIR/${BATTERY_LOG_NAME}"
    exit 1
fi
if [ ! -f "$LOG_DIR/$NMPC_CONFIG_BASENAME" ]; then
    echo "ERROR: Missing $LOG_DIR/$NMPC_CONFIG_BASENAME"
    exit 1
fi
fi

if [ $FROM_RAW -eq 0 ]; then
SOLVER_CSVS=()
for SOLVER_CSV in "$LOG_DIR"/nmpc_solver_data_*.csv; do
    [ -f "$SOLVER_CSV" ] || continue
    if [[ "$SOLVER_CSV" == *_offline_disturbance.csv ]]; then
        continue
    fi
    if [[ "$SOLVER_CSV" == */nmpc_solver_data_combined.csv ]]; then
        continue
    fi
    if [[ "$SOLVER_CSV" == *_normalized.csv ]]; then
        continue
    fi
    SOLVER_CSVS+=("$SOLVER_CSV")
done

if [ ${#SOLVER_CSVS[@]} -eq 0 ]; then
    echo "ERROR: No nmpc_solver_data_*.csv files found in $LOG_DIR"
    exit 1
fi

COMBINED_SOLVER_CSV="$LOG_DIR/nmpc_solver_data_combined.csv"
echo "Extracting nmpc_solver_data_combined.csv"
python3 "$SCRIPT_DIR/combine_nmpc_solver_logs.py" "$COMBINED_SOLVER_CSV" "${SOLVER_CSVS[@]}" >/dev/null

if [ ! -f "$LOG_DIR/px4_run.ulg" ]; then
    echo "ERROR: Missing $LOG_DIR/px4_run.ulg"
    exit 1
fi

NORMALIZE_BATTERY_ARGS=()
if [ -f "$LOG_DIR/${BATTERY_LOG_NAME}" ]; then
    NORMALIZE_BATTERY_ARGS=(--battery-csv-path "$LOG_DIR/${BATTERY_LOG_NAME}")
else
    echo "[WARN] No $LOG_DIR/${BATTERY_LOG_NAME} found; normalizing without a battery status log"
fi

echo "Extracting normalized aggregate logs"
python3 "$SCRIPT_DIR/normalize_mocap_logs.py" \
    --ulog-path "$LOG_DIR/px4_run.ulg" \
    --solver-csv-path "$COMBINED_SOLVER_CSV" \
    --px4-motor-csv-path "$LOG_DIR/px4_motor_log.csv" \
    --px4-flight-csv-path "$LOG_DIR/px4_flight_log.csv" \
    --qualisys-csv-path "$LOG_DIR/${RAW_QUALISYS_CSV_NAME}" \
    "${NORMALIZE_BATTERY_ARGS[@]}" \
    --output-dir "$LOG_DIR" \
    --qualisys-match-max-position-error-m "$QUALISYS_MATCH_MAX_POSITION_ERROR_M" >/dev/null
fi

if [ $SKIP_NON_ESSENTIAL -eq 1 ]; then
    RESULTS_DIR_REL=$(realpath --relative-to="$PWD" "$LOG_DIR")
    echo "results directory: $RESULTS_DIR_REL"
    exit 0
fi

PLOT_OUT_DIR="$LOG_DIR"
if [ $RUN_ARCHIVE_MODE -eq 1 ]; then
    # Every supported archive suffix (.tar, .tar.zst, .tar.xz, .tar.gz, .tgz) starts with ".t"
    PLOT_OUT_DIR="${LOG_DIR%.t*}_plots"
    mkdir -p "$PLOT_OUT_DIR"
fi
PDF_PATH="$PLOT_OUT_DIR/nmpc_solver_data_combined_diagnostics.pdf"
HTML_PATH="$PLOT_OUT_DIR/nmpc_solver_data_combined_diagnostics.html"
OFFLINE_CSV_PATH="$PLOT_OUT_DIR/nmpc_solver_data_combined_offline_disturbance.csv"
OFFLINE_PDF_PATH="$PLOT_OUT_DIR/nmpc_solver_data_combined_offline_disturbance.pdf"
OFFLINE_HTML_PATH="$PLOT_OUT_DIR/nmpc_solver_data_combined_offline_disturbance.html"
DOMINANT_COST_DISTRIBUTION_LOG_PATH="$PLOT_OUT_DIR/cost_distribution_after_collision.log"

PLOT_BATTERY_ARGS=()
if [ -f "$LOG_DIR/battery_status_log_normalized.csv" ]; then
    PLOT_BATTERY_ARGS=(--battery-csv-path "$LOG_DIR/battery_status_log_normalized.csv")
else
    echo "[WARN] No $LOG_DIR/battery_status_log_normalized.csv found; plotting without a battery status page"
fi

echo "Plotting nmpc_solver_data_combined_diagnostics.pdf"
if [ $FROM_RAW -eq 1 ]; then
    python3 "$SCRIPT_DIR/plot_hitl.py" \
        --raw-run-dir "$LOG_DIR" \
        --qualisys-match-max-position-error-m "$QUALISYS_MATCH_MAX_POSITION_ERROR_M" \
        "$PDF_PATH" \
        --nmpc-config-path "$LOG_DIR/$NMPC_CONFIG_BASENAME" \
        --html-path "$HTML_PATH" \
        --offline-disturbance-pdf "$OFFLINE_PDF_PATH" \
        --offline-disturbance-html "$OFFLINE_HTML_PATH" \
        --dominant-cost-window-start-s "$DOMINANT_COST_WINDOW_START_S" \
        --dominant-cost-window-end-s "$DOMINANT_COST_WINDOW_END_S" \
        --dominant-cost-top-count "$DOMINANT_COST_TOP_COUNT" \
        --dominant-cost-distribution-log "$DOMINANT_COST_DISTRIBUTION_LOG_PATH" \
        --dominant-cost-distribution-slice-count "$DOMINANT_COST_DISTRIBUTION_SLICE_COUNT" >/dev/null
else
    python3 "$SCRIPT_DIR/plot_hitl.py" \
        "$LOG_DIR/nmpc_solver_data_combined_normalized.csv" \
        "$LOG_DIR/px4_motor_log_normalized.csv" \
        "$LOG_DIR/px4_flight_log_normalized.csv" \
        "$PDF_PATH" \
        "${PLOT_BATTERY_ARGS[@]}" \
        --nmpc-config-path "$LOG_DIR/$NMPC_CONFIG_BASENAME" \
        --html-path "$HTML_PATH" \
        --offline-disturbance-csv "$OFFLINE_CSV_PATH" \
        --offline-disturbance-pdf "$OFFLINE_PDF_PATH" \
        --offline-disturbance-html "$OFFLINE_HTML_PATH" \
        --dominant-cost-window-start-s "$DOMINANT_COST_WINDOW_START_S" \
        --dominant-cost-window-end-s "$DOMINANT_COST_WINDOW_END_S" \
        --dominant-cost-top-count "$DOMINANT_COST_TOP_COUNT" \
        --dominant-cost-distribution-log "$DOMINANT_COST_DISTRIBUTION_LOG_PATH" \
        --dominant-cost-distribution-slice-count "$DOMINANT_COST_DISTRIBUTION_SLICE_COUNT" >/dev/null
fi

RESULTS_DIR_REL=$(realpath --relative-to="$PWD" "$LOG_DIR")
PDF_PATH_REL=$(realpath --relative-to="$PWD" "$PDF_PATH")
echo "results directory: $RESULTS_DIR_REL"
echo "diagnostics pdf: $PDF_PATH_REL"
