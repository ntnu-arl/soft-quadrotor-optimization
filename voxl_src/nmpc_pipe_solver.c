#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <pthread.h>
#include <ctype.h>

#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_quadrotor_nmpc.h"
#include "acados_solver_quadrotor_nmpc_recovery.h"

#include <math.h>
#include "protocol.h"

#define QUADROTOR_NX QUADROTOR_NMPC_NX
#define QUADROTOR_NU QUADROTOR_NMPC_NU
#define QUADROTOR_NP QUADROTOR_NMPC_NP
#define QUADROTOR_N  QUADROTOR_NMPC_N

#if QUADROTOR_NMPC_RECOVERY_NX != QUADROTOR_NMPC_NX
#error "nominal and recovery NMPC solvers must have identical NX"
#endif
#if QUADROTOR_NMPC_RECOVERY_NU != QUADROTOR_NMPC_NU
#error "nominal and recovery NMPC solvers must have identical NU"
#endif
#if QUADROTOR_NMPC_RECOVERY_NP != QUADROTOR_NMPC_NP
#error "nominal and recovery NMPC solvers must have identical NP"
#endif
#if QUADROTOR_NMPC_RECOVERY_N != QUADROTOR_NMPC_N
#error "nominal and recovery NMPC solvers must have identical horizon length"
#endif
#if QUADROTOR_NMPC_RECOVERY_NY != QUADROTOR_NMPC_NY
#error "nominal and recovery NMPC solvers must have identical NY"
#endif
#if QUADROTOR_NMPC_RECOVERY_NY0 != QUADROTOR_NMPC_NY0
#error "nominal and recovery NMPC solvers must have identical NY0"
#endif
#if QUADROTOR_NMPC_RECOVERY_NYN != QUADROTOR_NMPC_NYN
#error "nominal and recovery NMPC solvers must have identical NYN"
#endif
#if QUADROTOR_NMPC_NY0 != 23
#error "regenerate nominal NMPC solver with acceleration tracking cost"
#endif
#if QUADROTOR_NMPC_NY != 23
#error "regenerate nominal NMPC solver with acceleration tracking cost"
#endif
#if QUADROTOR_NMPC_NYN != 12
#error "unexpected nominal NMPC terminal cost dimension"
#endif

#define STATE_SINK_PATH   "/run/mpa/nmpc_state"
#define CONTROL_SINK_PATH "/run/mpa/nmpc_control"
#define MPA_DIR           "/run/mpa"
#define CSV_LOG_PATH_FMT  "/tmp/nmpc_solver_data_%Y%m%d_%H%M%S.csv"
#define RUN_MANIFEST_PATH "/tmp/nmpc_solver_current_run.txt"
#define RUN_MANIFEST_TMP_PATH "/tmp/nmpc_solver_current_run.txt.tmp"
#define NMPC_CONFIG_PATH "/home/root/nmpc_controller_cfg.yaml"
#define ENABLE_REALTIME_DISTURBANCE_ESTIMATION 1

#define PARAM_MASS 0
#define PARAM_IXX 1
#define PARAM_IXY 2
#define PARAM_IXZ 3
#define PARAM_IYY 4
#define PARAM_IYZ 5
#define PARAM_IZZ 6
#define PARAM_GRAVITY 7
#define PARAM_POS_REF 10
#define PARAM_VEL_REF 13
#define PARAM_ACC_REF 16
#define PARAM_QUAT_REF 19
#define PARAM_ALLOC 23
#define PARAM_ALLOC_SIZE (6 * QUADROTOR_NU)
#define PARAM_KF (PARAM_ALLOC + PARAM_ALLOC_SIZE)
#define PARAM_TAU (PARAM_KF + QUADROTOR_NU)
#define PARAM_COM (PARAM_TAU + QUADROTOR_NU)
#define PARAM_DIST_FORCE (PARAM_COM + 3)
#define PARAM_DIST_TORQUE (PARAM_DIST_FORCE + 3)
#define PARAM_U_PREV (PARAM_DIST_TORQUE + 3)

#define DIST_WARMUP_TIME_S 2.0
#define DIST_WARMUP_MIN_SAMPLES 40
#define DIST_INITIAL_SIGMA_RATIO 0.50
#define DIST_READY_SIGMA_RATIO 0.15
#define DIST_VARIANCE_FLOOR_SIGMA_RATIO 0.05
#define DIST_FORCE_PROCESS_SIGMA_RATIO 0.03
#define DIST_TORQUE_PROCESS_SIGMA_RATIO 0.03
#define DIST_FORCE_MEAS_SIGMA_RATIO 0.10
#define DIST_TORQUE_MEAS_SIGMA_RATIO 0.20
#define DIST_FORCE_ABS_GATE_RATIO 2.0
#define DIST_TORQUE_ABS_GATE_RATIO 2.0
#define DIST_INNOVATION_GATE_NIS 25.0
// At handoff the motors are still spooling up from the pre-NMPC RPM and the mocap/EKF
// pipeline can serve stale samples, so the wrench residual reads a large phantom force.
// Measurements inside this window after estimator (re)initialization are discarded.
#define DIST_STARTUP_BLANK_TIME_S 0.5
// Extra process noise injected when a plausible measurement (inside the absolute gate)
// is rejected by the NIS gate. A wrong estimate with small variance would otherwise
// reject every honest measurement forever; this re-opens the gate after ~0.6 s of
// persistent disagreement. Abs-gated transients (e.g. collision) do not inflate.
#define DIST_GATED_PROCESS_SIGMA_RATIO 0.15
#define DIST_SLEW_RATE_RATIO_PER_S 0.6
#define DIST_FORCE_HORIZON_DECAY_TAU_S 2.0

#define DIST_FORCE_BOUND_X 1.0
#define DIST_FORCE_BOUND_Y 1.0
#define DIST_FORCE_BOUND_Z 1.0
#define DIST_TORQUE_BOUND_X 0.1
#define DIST_TORQUE_BOUND_Y 0.1
#define DIST_TORQUE_BOUND_Z 0.1
// voxl_esc refreshes only one ESC per actuator cycle, so a healthy motor sample can
// be a few control ticks old; 50 ms rejects stale telemetry without dropping that path.
#define MOTOR_RPS_MEAS_TIMEOUT_US 50000ULL
#define MOTOR_RPS_CORRECTION_GAIN_MIN 0.15
#define MOTOR_RPS_CORRECTION_GAIN_MAX 0.60
#define BODY_DRAG_COEFF_BX 0.130144170
#define BODY_DRAG_COEFF_BY 0.118901478
#define BODY_DRAG_COEFF_BZ 0.0
#define TWO_PI 6.28318530717958647692

typedef struct {
    double rps_hat[QUADROTOR_NU];
    double commanded_rps_prev[QUADROTOR_NU];
    uint64_t last_sample_timestamp_us;
    int initialized;
} motor_state_estimator_t;

typedef struct {
    const char *name;
    void *capsule;
    ocp_nlp_solver *nlp_solver;
    ocp_nlp_config *config;
    ocp_nlp_dims *dims;
    ocp_nlp_in *in;
    ocp_nlp_out *out;
    int (*solve)(void *capsule);
    int (*update_params)(void *capsule, int stage, double *params, int np);
} nmpc_solver_runtime_t;

typedef struct {
    double q_pos[3];
    double q_vel[3];
    double q_acc[3];
    double q_att[3];
    double q_omega[3];
    double r;
    double r_delta_u;
    double terminal_factor;
} nmpc_weight_config_t;

typedef enum {
    TRAJECTORY_MODE_COLLISION = 0,
    TRAJECTORY_MODE_CIRCLE = 1,
} trajectory_mode_t;

typedef struct {
    double peak_velocity;
    double max_acceleration;
    double trajectory_diameter_x;
    double hold_before_s;
    double hold_after_s;
    int repeat_count;
    double radius;
    double omega;
    double period_s;
    double circumference;
    double path_length;
    double accel_time_s;
    double accel_distance;
    double cruise_distance;
    double cruise_time_s;
    double moving_duration_s;
    double total_duration_s;
} circle_trajectory_config_t;

typedef struct {
    trajectory_mode_t mode;
    double nmpc_dt;
    circle_trajectory_config_t circle;
    double start_position[3];
    uint64_t start_timestamp_us;
    int initialized;
    double angle_offset_deg;
    double angle_offset_rad;
    double gap_center[2];
    double cos_neg;
    double sin_neg;
} trajectory_runtime_t;

typedef struct {
    double estimated_force[3];
    double estimated_torque[3];
    double applied_force[3];
    double applied_torque[3];
    double force_variance[3];
    double torque_variance[3];
    double prev_velocity[3];
    double prev_omega[3];
    uint64_t last_force_timestamp_us;
    uint64_t last_torque_timestamp_us;
    double warmup_force_s;
    double warmup_torque_s;
    int sample_count_force;
    int sample_count_torque;
    int initialized;
} disturbance_estimator_t;

static int solve_nominal(void *capsule) {
    return quadrotor_nmpc_acados_solve((quadrotor_nmpc_solver_capsule *)capsule);
}

static int update_params_nominal(void *capsule, int stage, double *params, int np) {
    return quadrotor_nmpc_acados_update_params((quadrotor_nmpc_solver_capsule *)capsule, stage, params, np);
}

static int solve_recovery(void *capsule) {
    return quadrotor_nmpc_recovery_acados_solve((quadrotor_nmpc_recovery_solver_capsule *)capsule);
}

static int update_params_recovery(void *capsule, int stage, double *params, int np) {
    return quadrotor_nmpc_recovery_acados_update_params(
        (quadrotor_nmpc_recovery_solver_capsule *)capsule,
        stage,
        params,
        np
    );
}

static void require_finite_array(const char *name, const double *values, int count) {
    for (int i = 0; i < count; i++) {
        if (!isfinite(values[i])) {
            fprintf(stderr, "Non-finite value in %s[%d]=%.9f\n", name, i, values[i]);
            exit(EXIT_FAILURE);
        }
    }
}

static char *read_required_text_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        perror("fseek config end");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    const long size = ftell(file);
    if (size <= 0) {
        fprintf(stderr, "Config file is empty or unreadable: %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        perror("fseek config start");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    char *text = malloc((size_t)size + 1u);
    if (!text) {
        fprintf(stderr, "Failed to allocate config buffer for %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    const size_t got = fread(text, 1u, (size_t)size, file);
    if (got != (size_t)size) {
        fprintf(stderr, "Failed to read full config file %s\n", path);
        free(text);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    text[size] = '\0';
    fclose(file);
    return text;
}

static const char *next_line(const char *line, const char *end) {
    const char *newline = memchr(line, '\n', (size_t)(end - line));
    return newline ? newline + 1 : end;
}

static int yaml_line_indent(const char *line, const char *line_end) {
    int indent = 0;
    while (line + indent < line_end && line[indent] == ' ') {
        indent++;
    }
    return indent;
}

static const char *yaml_line_value(
    const char *start,
    const char *end,
    int required_indent,
    const char *key
) {
    const size_t key_len = strlen(key);

    for (const char *line = start; line < end; line = next_line(line, end)) {
        const char *line_end = memchr(line, '\n', (size_t)(end - line));
        if (!line_end) {
            line_end = end;
        }

        if (yaml_line_indent(line, line_end) != required_indent) {
            continue;
        }

        const char *trimmed = line + required_indent;
        if ((size_t)(line_end - trimmed) < key_len + 1u) {
            continue;
        }

        if (strncmp(trimmed, key, key_len) == 0 && trimmed[key_len] == ':') {
            return trimmed + key_len + 1u;
        }
    }

    fprintf(stderr, "Required YAML key missing: %s\n", key);
    exit(EXIT_FAILURE);
}

static const char *find_yaml_section(const char *start, const char *end, int indent, const char *key) {
    return yaml_line_value(start, end, indent, key);
}

static const char *find_yaml_section_end(const char *section_body, const char *end, int section_indent) {
    for (const char *line = section_body; line < end; line = next_line(line, end)) {
        const char *line_end = memchr(line, '\n', (size_t)(end - line));
        if (!line_end) {
            line_end = end;
        }

        const char *cursor = line;
        while (cursor < line_end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r')) {
            cursor++;
        }
        if (cursor == line_end || *cursor == '#') {
            continue;
        }

        if (yaml_line_indent(line, line_end) <= section_indent) {
            return line;
        }
    }

    return end;
}

static void skip_spaces(const char **cursor) {
    while (**cursor == ' ' || **cursor == '\t') {
        (*cursor)++;
    }
}

static double parse_yaml_double_or_nan(const char *value, const char *key) {
    const char *cursor = value;
    skip_spaces(&cursor);

    if (strncmp(cursor, "null", 4) == 0) {
        const char next = cursor[4];
        if (next == '\0' || next == '\n' || next == '\r' || next == ' ' || next == '#') {
            return NAN;
        }
    }

    char *parse_end = NULL;
    errno = 0;
    const double parsed = strtod(cursor, &parse_end);
    if (parse_end == cursor || errno != 0 || !isfinite(parsed)) {
        fprintf(stderr, "Invalid numeric YAML value for %s\n", key);
        exit(EXIT_FAILURE);
    }

    return parsed;
}

static void parse_required_yaml_string(
    const char *start,
    const char *end,
    int indent,
    const char *key,
    char *out,
    size_t out_size
) {
    if (out_size < 2u) {
        fprintf(stderr, "String output buffer is too small for YAML key: %s\n", key);
        exit(EXIT_FAILURE);
    }

    const char *cursor = yaml_line_value(start, end, indent, key);
    skip_spaces(&cursor);

    const char *value_end = cursor;
    while (*value_end != '\0' && *value_end != '\n' && *value_end != '\r' && *value_end != '#') {
        value_end++;
    }
    while (value_end > cursor && (value_end[-1] == ' ' || value_end[-1] == '\t')) {
        value_end--;
    }

    if (value_end <= cursor) {
        fprintf(stderr, "YAML string value must not be empty: %s\n", key);
        exit(EXIT_FAILURE);
    }

    if (((*cursor == '"' && value_end[-1] == '"') || (*cursor == '\'' && value_end[-1] == '\'')) && value_end - cursor >= 2) {
        cursor++;
        value_end--;
    }

    const size_t value_len = (size_t)(value_end - cursor);
    if (value_len == 0u || value_len >= out_size) {
        fprintf(stderr, "Invalid YAML string length for %s\n", key);
        exit(EXIT_FAILURE);
    }

    memcpy(out, cursor, value_len);
    out[value_len] = '\0';
}

static double parse_required_yaml_double(
    const char *start,
    const char *end,
    int indent,
    const char *key
) {
    const double value = parse_yaml_double_or_nan(yaml_line_value(start, end, indent, key), key);
    if (!isfinite(value)) {
        fprintf(stderr, "YAML key must be finite: %s\n", key);
        exit(EXIT_FAILURE);
    }
    return value;
}

static int parse_required_yaml_positive_int(
    const char *start,
    const char *end,
    int indent,
    const char *key
) {
    const char *cursor = yaml_line_value(start, end, indent, key);
    skip_spaces(&cursor);

    char *parse_end = NULL;
    errno = 0;
    const long parsed = strtol(cursor, &parse_end, 10);
    if (parse_end == cursor || errno != 0 || parsed <= 0 || parsed > 2147483647L) {
        fprintf(stderr, "YAML key must be a positive integer: %s\n", key);
        exit(EXIT_FAILURE);
    }

    while (*parse_end == ' ' || *parse_end == '\t') {
        parse_end++;
    }
    if (*parse_end != '\0' && *parse_end != '\n' && *parse_end != '\r' && *parse_end != '#') {
        fprintf(stderr, "YAML integer value contains trailing characters: %s\n", key);
        exit(EXIT_FAILURE);
    }

    return (int)parsed;
}

static void parse_required_yaml_vector3(
    const char *start,
    const char *end,
    int indent,
    const char *key,
    double out[3]
) {
    const char *cursor = yaml_line_value(start, end, indent, key);
    skip_spaces(&cursor);

    if (*cursor != '[') {
        fprintf(stderr, "YAML key must be a 3-vector: %s\n", key);
        exit(EXIT_FAILURE);
    }

    cursor++;
    for (int axis = 0; axis < 3; axis++) {
        skip_spaces(&cursor);
        char *parse_end = NULL;
        errno = 0;
        out[axis] = strtod(cursor, &parse_end);
        if (parse_end == cursor || errno != 0 || !isfinite(out[axis])) {
            fprintf(stderr, "Invalid numeric YAML vector value for %s[%d]\n", key, axis);
            exit(EXIT_FAILURE);
        }
        cursor = parse_end;
        skip_spaces(&cursor);
        if (axis < 2) {
            if (*cursor != ',') {
                fprintf(stderr, "YAML vector must contain exactly three comma-separated values: %s\n", key);
                exit(EXIT_FAILURE);
            }
            cursor++;
        }
    }

    skip_spaces(&cursor);
    if (*cursor != ']') {
        fprintf(stderr, "YAML vector must contain exactly three values: %s\n", key);
        exit(EXIT_FAILURE);
    }
}

static void parse_weight_config(
    const char *start,
    const char *end,
    int indent,
    nmpc_weight_config_t *weights
) {
    parse_required_yaml_vector3(start, end, indent, "Q_pos", weights->q_pos);
    parse_required_yaml_vector3(start, end, indent, "Q_vel", weights->q_vel);
    parse_required_yaml_vector3(start, end, indent, "Q_acc", weights->q_acc);
    parse_required_yaml_vector3(start, end, indent, "Q_att", weights->q_att);
    parse_required_yaml_vector3(start, end, indent, "Q_omega", weights->q_omega);
    weights->r = parse_required_yaml_double(start, end, indent, "R");
    weights->r_delta_u = parse_required_yaml_double(start, end, indent, "R_delta_u");
    weights->terminal_factor = parse_required_yaml_double(start, end, indent, "terminal_factor");

    if (!(weights->r >= 0.0) || !(weights->r_delta_u >= 0.0) || !(weights->terminal_factor >= 0.0)) {
        fprintf(stderr, "NMPC scalar weights must be non-negative\n");
        exit(EXIT_FAILURE);
    }
}

static void load_nmpc_weight_configs(
    const char *path,
    nmpc_weight_config_t *nominal_weights,
    nmpc_weight_config_t *recovery_weights
) {
    char *text = read_required_text_file(path);
    const char *text_end = text + strlen(text);
    const char *nmpc_start = find_yaml_section(text, text_end, 0, "nmpc");
    const char *trajectory_start = find_yaml_section(nmpc_start, text_end, 0, "trajectory");
    const char *recovery_start = find_yaml_section(nmpc_start, trajectory_start, 2, "recovery");

    parse_weight_config(nmpc_start, recovery_start, 2, nominal_weights);
    parse_weight_config(recovery_start, trajectory_start, 4, recovery_weights);

    free(text);
}

static double load_motor_rps_min(const char *path) {
    char *text = read_required_text_file(path);
    const char *text_end = text + strlen(text);
    const char *nmpc_start = find_yaml_section(text, text_end, 0, "nmpc");
    const char *trajectory_start = find_yaml_section(nmpc_start, text_end, 0, "trajectory");
    const double motor_rps_min = parse_required_yaml_double(nmpc_start, trajectory_start, 2, "motor_rps_min");

    free(text);

    if (!(motor_rps_min > 0.0)) {
        fprintf(stderr, "nmpc.motor_rps_min must be positive\n");
        exit(EXIT_FAILURE);
    }

    return motor_rps_min;
}

static void validate_circle_trajectory_config(circle_trajectory_config_t *circle) {
    if (!(circle->peak_velocity > 0.0)) {
        fprintf(stderr, "trajectory.circle.peak_velocity must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (!(circle->max_acceleration > 0.0)) {
        fprintf(stderr, "trajectory.circle.max_acceleration must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (!(circle->trajectory_diameter_x > 0.0)) {
        fprintf(stderr, "trajectory.circle.trajectory_diameter_x must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (!(circle->hold_before_s >= 0.0)) {
        fprintf(stderr, "trajectory.circle.hold_before_s must be non-negative\n");
        exit(EXIT_FAILURE);
    }
    if (!(circle->hold_after_s >= 0.0)) {
        fprintf(stderr, "trajectory.circle.hold_after_s must be non-negative\n");
        exit(EXIT_FAILURE);
    }

    circle->radius = 0.5 * circle->trajectory_diameter_x;
    circle->omega = circle->peak_velocity / circle->radius;
    circle->period_s = TWO_PI / circle->omega;
    circle->circumference = TWO_PI * circle->radius;
    circle->path_length = (double)circle->repeat_count * circle->circumference;
    circle->accel_time_s = 0.25 * TWO_PI * circle->peak_velocity / circle->max_acceleration;
    circle->accel_distance = 0.5 * circle->peak_velocity * circle->accel_time_s;
    if (2.0 * circle->accel_distance > circle->path_length) {
        fprintf(stderr, "trajectory.circle path is too short to reach peak_velocity with max_acceleration\n");
        exit(EXIT_FAILURE);
    }
    circle->cruise_distance = circle->path_length - 2.0 * circle->accel_distance;
    circle->cruise_time_s = circle->cruise_distance / circle->peak_velocity;
    circle->moving_duration_s = 2.0 * circle->accel_time_s + circle->cruise_time_s;
    circle->total_duration_s = circle->hold_before_s + circle->moving_duration_s + circle->hold_after_s;

    const double values[] = {
        circle->radius,
        circle->omega,
        circle->period_s,
        circle->circumference,
        circle->path_length,
        circle->accel_time_s,
        circle->accel_distance,
        circle->cruise_distance,
        circle->cruise_time_s,
        circle->moving_duration_s,
        circle->total_duration_s,
    };
    require_finite_array("trajectory.circle", values, (int)(sizeof(values) / sizeof(values[0])));
}

static void parse_circle_trajectory_config(
    const char *start,
    const char *end,
    int indent,
    circle_trajectory_config_t *circle
) {
    circle->peak_velocity = parse_required_yaml_double(start, end, indent, "peak_velocity");
    circle->max_acceleration = parse_required_yaml_double(start, end, indent, "max_acceleration");
    circle->trajectory_diameter_x = parse_required_yaml_double(start, end, indent, "trajectory_diameter_x");
    circle->hold_before_s = parse_required_yaml_double(start, end, indent, "hold_before_s");
    circle->hold_after_s = parse_required_yaml_double(start, end, indent, "hold_after_s");
    circle->repeat_count = parse_required_yaml_positive_int(start, end, indent, "repeat_count");
    validate_circle_trajectory_config(circle);
}

static void load_trajectory_runtime_config(const char *path, trajectory_runtime_t *trajectory) {
    memset(trajectory, 0, sizeof(*trajectory));

    char *text = read_required_text_file(path);
    const char *text_end = text + strlen(text);
    const char *nmpc_start = find_yaml_section(text, text_end, 0, "nmpc");
    const char *trajectory_start = find_yaml_section(nmpc_start, text_end, 0, "trajectory");
    const char *trajectory_end = find_yaml_section_end(trajectory_start, text_end, 0);
    const char *circle_start = find_yaml_section(trajectory_start, trajectory_end, 2, "circle");
    const char *circle_end = find_yaml_section_end(circle_start, trajectory_end, 2);

    const int configured_horizon = parse_required_yaml_positive_int(nmpc_start, trajectory_start, 2, "N");
    if (configured_horizon != QUADROTOR_N) {
        fprintf(
            stderr,
            "nmpc.N=%d does not match compiled acados horizon QUADROTOR_N=%d; regenerate and redeploy the solver\n",
            configured_horizon,
            QUADROTOR_N
        );
        exit(EXIT_FAILURE);
    }

    trajectory->nmpc_dt = parse_required_yaml_double(nmpc_start, trajectory_start, 2, "nmpc_dt");
    if (!(trajectory->nmpc_dt > 0.0)) {
        fprintf(stderr, "nmpc.nmpc_dt must be positive\n");
        exit(EXIT_FAILURE);
    }

    char active[32];
    parse_required_yaml_string(trajectory_start, trajectory_end, 2, "active", active, sizeof(active));
    if (strcmp(active, "collision") == 0) {
        trajectory->mode = TRAJECTORY_MODE_COLLISION;
    } else if (strcmp(active, "circle") == 0) {
        trajectory->mode = TRAJECTORY_MODE_CIRCLE;
    } else {
        fprintf(stderr, "Unsupported trajectory.active: %s\n", active);
        exit(EXIT_FAILURE);
    }

    trajectory->angle_offset_deg = parse_required_yaml_double(nmpc_start, trajectory_start, 2, "angle_offset_deg");
    if (trajectory->angle_offset_deg != 0.0 && trajectory->mode != TRAJECTORY_MODE_COLLISION) {
        fprintf(
            stderr,
            "nmpc.angle_offset_deg=%.6f is only supported for trajectory.active=collision (got %s)\n",
            trajectory->angle_offset_deg,
            active
        );
        exit(EXIT_FAILURE);
    }
    const double pi = atan2(0.0, -1.0);
    trajectory->angle_offset_rad = trajectory->angle_offset_deg * pi / 180.0;
    trajectory->cos_neg = cos(-trajectory->angle_offset_rad);
    trajectory->sin_neg = sin(-trajectory->angle_offset_rad);

    const char *collision_start = find_yaml_section(trajectory_start, trajectory_end, 2, "collision");
    const char *collision_end = find_yaml_section_end(collision_start, trajectory_end, 2);
    double gap_ref1_mm[3];
    double gap_ref2_mm[3];
    parse_required_yaml_vector3(collision_start, collision_end, 4, "gap_ref1_mm", gap_ref1_mm);
    parse_required_yaml_vector3(collision_start, collision_end, 4, "gap_ref2_mm", gap_ref2_mm);
    trajectory->gap_center[0] = 0.5 * (gap_ref1_mm[0] + gap_ref2_mm[0]) / 1000.0;
    trajectory->gap_center[1] = 0.5 * (gap_ref1_mm[1] + gap_ref2_mm[1]) / 1000.0;
    require_finite_array("trajectory.collision.gap_center", trajectory->gap_center, 2);

    parse_circle_trajectory_config(circle_start, circle_end, 4, &trajectory->circle);

    printf(
        "Loaded trajectory config from %s: active=%s nmpc_dt=%.6f "
        "angle_offset_deg=%.6f gap_center=[%.6f %.6f] "
        "diameter_x=%.6f peak_velocity=%.6f max_acceleration=%.6f period_s=%.6f "
        "repeat_count=%d hold_before_s=%.6f hold_after_s=%.6f moving_duration_s=%.6f total_duration_s=%.6f\n",
        path,
        active,
        trajectory->nmpc_dt,
        trajectory->angle_offset_deg,
        trajectory->gap_center[0],
        trajectory->gap_center[1],
        trajectory->circle.trajectory_diameter_x,
        trajectory->circle.peak_velocity,
        trajectory->circle.max_acceleration,
        trajectory->circle.period_s,
        trajectory->circle.repeat_count,
        trajectory->circle.hold_before_s,
        trajectory->circle.hold_after_s,
        trajectory->circle.moving_duration_s,
        trajectory->circle.total_duration_s
    );

    free(text);
}

static void apply_nmpc_weights(nmpc_solver_runtime_t *solver, const nmpc_weight_config_t *weights) {
    double stage_w[QUADROTOR_NMPC_NY * QUADROTOR_NMPC_NY];
    double terminal_w[QUADROTOR_NMPC_NYN * QUADROTOR_NMPC_NYN];
    memset(stage_w, 0, sizeof(stage_w));
    memset(terminal_w, 0, sizeof(terminal_w));

    const double stage_diag[15] = {
        weights->q_pos[0], weights->q_pos[1], weights->q_pos[2],
        weights->q_vel[0], weights->q_vel[1], weights->q_vel[2],
        weights->q_acc[0], weights->q_acc[1], weights->q_acc[2],
        weights->q_att[0], weights->q_att[1], weights->q_att[2],
        weights->q_omega[0], weights->q_omega[1], weights->q_omega[2],
    };
    const double terminal_diag[12] = {
        weights->q_pos[0], weights->q_pos[1], weights->q_pos[2],
        weights->q_vel[0], weights->q_vel[1], weights->q_vel[2],
        weights->q_att[0], weights->q_att[1], weights->q_att[2],
        weights->q_omega[0], weights->q_omega[1], weights->q_omega[2],
    };

    for (int i = 0; i < 15; i++) {
        if (!(stage_diag[i] >= 0.0)) {
            fprintf(stderr, "NMPC stage weight for %s[%d] is negative\n", solver->name, i);
            exit(EXIT_FAILURE);
        }
        stage_w[i + QUADROTOR_NMPC_NY * i] = stage_diag[i];
    }

    for (int i = 0; i < 12; i++) {
        if (!(terminal_diag[i] >= 0.0)) {
            fprintf(stderr, "NMPC terminal weight for %s[%d] is negative\n", solver->name, i);
            exit(EXIT_FAILURE);
        }
        terminal_w[i + QUADROTOR_NMPC_NYN * i] = terminal_diag[i] * weights->terminal_factor;
    }

    for (int i = 0; i < QUADROTOR_NU; i++) {
        stage_w[(15 + i) + QUADROTOR_NMPC_NY * (15 + i)] = weights->r;
        stage_w[(15 + QUADROTOR_NU + i) + QUADROTOR_NMPC_NY * (15 + QUADROTOR_NU + i)] = weights->r_delta_u;
    }

    for (int stage = 0; stage < QUADROTOR_N; stage++) {
        ocp_nlp_cost_model_set(solver->config, solver->dims, solver->in, stage, "W", stage_w);
    }

    ocp_nlp_cost_model_set(solver->config, solver->dims, solver->in, QUADROTOR_N, "W", terminal_w);

    const int status = ocp_nlp_precompute(solver->nlp_solver, solver->in, solver->out);
    if (status != 0) {
        fprintf(stderr, "Failed to precompute %s solver after loading YAML weights: status=%d\n", solver->name, status);
        exit(EXIT_FAILURE);
    }

    printf(
        "Loaded %s NMPC weights from %s: Q_pos=[%.6f %.6f %.6f] Q_vel=[%.6f %.6f %.6f] "
        "Q_acc=[%.6f %.6f %.6f] Q_att=[%.6f %.6f %.6f] Q_omega=[%.6f %.6f %.6f] "
        "R=%.6f R_delta_u=%.6f terminal_factor=%.6f\n",
        solver->name,
        NMPC_CONFIG_PATH,
        weights->q_pos[0], weights->q_pos[1], weights->q_pos[2],
        weights->q_vel[0], weights->q_vel[1], weights->q_vel[2],
        weights->q_acc[0], weights->q_acc[1], weights->q_acc[2],
        weights->q_att[0], weights->q_att[1], weights->q_att[2],
        weights->q_omega[0], weights->q_omega[1], weights->q_omega[2],
        weights->r,
        weights->r_delta_u,
        weights->terminal_factor
    );
}

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void reset_trajectory_runtime(
    trajectory_runtime_t *trajectory,
    uint64_t sample_timestamp_us,
    const double start_position[3]
) {
    trajectory->start_timestamp_us = sample_timestamp_us;
    memcpy(trajectory->start_position, start_position, 3 * sizeof(double));
    require_finite_array("trajectory.start_position", trajectory->start_position, 3);
    trajectory->initialized = 1;
}

static double get_trajectory_elapsed_s(const trajectory_runtime_t *trajectory, uint64_t sample_timestamp_us) {
    if (!trajectory->initialized) {
        fprintf(stderr, "Trajectory runtime is not initialized\n");
        exit(EXIT_FAILURE);
    }
    if (sample_timestamp_us < trajectory->start_timestamp_us) {
        fprintf(
            stderr,
            "Trajectory timestamp moved backwards: current=%llu start=%llu\n",
            (unsigned long long)sample_timestamp_us,
            (unsigned long long)trajectory->start_timestamp_us
        );
        exit(EXIT_FAILURE);
    }

    return (double)(sample_timestamp_us - trajectory->start_timestamp_us) * 1e-6;
}

static void compute_circle_reference(
    const circle_trajectory_config_t *circle,
    const double start_position[3],
    double elapsed_s,
    double position[3],
    double velocity[3],
    double acceleration[3]
) {
    if (!(elapsed_s >= 0.0) || !isfinite(elapsed_s)) {
        fprintf(stderr, "Invalid trajectory elapsed_s=%.9f\n", elapsed_s);
        exit(EXIT_FAILURE);
    }

    position[0] = start_position[0];
    position[1] = start_position[1];
    position[2] = start_position[2];
    velocity[0] = 0.0;
    velocity[1] = 0.0;
    velocity[2] = 0.0;
    acceleration[0] = 0.0;
    acceleration[1] = 0.0;
    acceleration[2] = 0.0;

    if (elapsed_s < circle->hold_before_s) {
        return;
    }

    const double moving_elapsed_s = elapsed_s - circle->hold_before_s;
    if (!(moving_elapsed_s < circle->moving_duration_s)) {
        return;
    }

    double distance = 0.0;
    double speed = 0.0;
    double tangential_acceleration = 0.0;
    if (moving_elapsed_s < circle->accel_time_s) {
        const double phase = 0.5 * TWO_PI * moving_elapsed_s / circle->accel_time_s;
        speed = 0.5 * circle->peak_velocity * (1.0 - cos(phase));
        distance = 0.5 * circle->peak_velocity * (
            moving_elapsed_s - 2.0 * circle->accel_time_s * sin(phase) / TWO_PI
        );
        tangential_acceleration = circle->max_acceleration * sin(phase);
    } else {
        const double cruise_end_s = circle->accel_time_s + circle->cruise_time_s;
        if (moving_elapsed_s < cruise_end_s) {
            const double cruise_elapsed_s = moving_elapsed_s - circle->accel_time_s;
            speed = circle->peak_velocity;
            distance = circle->accel_distance + circle->peak_velocity * cruise_elapsed_s;
            tangential_acceleration = 0.0;
        } else {
            const double decel_elapsed_s = moving_elapsed_s - cruise_end_s;
            const double phase = 0.5 * TWO_PI * decel_elapsed_s / circle->accel_time_s;
            speed = 0.5 * circle->peak_velocity * (1.0 + cos(phase));
            distance = circle->accel_distance + circle->cruise_distance +
                0.5 * circle->peak_velocity * (
                    decel_elapsed_s + 2.0 * circle->accel_time_s * sin(phase) / TWO_PI
                );
            tangential_acceleration = -circle->max_acceleration * sin(phase);
        }
    }

    if (speed < 0.0 && speed > -1e-9) {
        speed = 0.0;
    }
    if (!(speed >= 0.0) || !isfinite(speed) || !isfinite(distance)) {
        fprintf(stderr, "Invalid circle reference speed=%.9f distance=%.9f\n", speed, distance);
        exit(EXIT_FAILURE);
    }

    const double theta = fmod(distance / circle->radius, TWO_PI);
    const double cos_theta = cos(theta);
    const double sin_theta = sin(theta);

    position[0] = start_position[0] + circle->radius * (1.0 - cos_theta);
    position[1] = start_position[1] + circle->radius * sin_theta;
    position[2] = start_position[2];
    velocity[0] = speed * sin_theta;
    velocity[1] = speed * cos_theta;
    velocity[2] = 0.0;
    const double centripetal_acceleration = (speed * speed) / circle->radius;
    acceleration[0] = tangential_acceleration * sin_theta + centripetal_acceleration * cos_theta;
    acceleration[1] = tangential_acceleration * cos_theta - centripetal_acceleration * sin_theta;
    acceleration[2] = 0.0;
}

static double vector3_norm(const double vector[3]) {
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}

static void cross3(const double *a, const double *b, double *out);

static void normalize_vector3_or_die(const char *name, double vector[3]) {
    const double norm = vector3_norm(vector);
    if (!(norm > 1e-9) || !isfinite(norm)) {
        fprintf(stderr, "Invalid %s norm %.9f\n", name, norm);
        exit(EXIT_FAILURE);
    }

    vector[0] /= norm;
    vector[1] /= norm;
    vector[2] /= norm;
}

static void rotmat_to_quat(const double rot[3][3], double quat[4]) {
    const double trace = rot[0][0] + rot[1][1] + rot[2][2];

    if (trace > 0.0) {
        const double scale = sqrt(trace + 1.0) * 2.0;
        quat[0] = 0.25 * scale;
        quat[1] = (rot[2][1] - rot[1][2]) / scale;
        quat[2] = (rot[0][2] - rot[2][0]) / scale;
        quat[3] = (rot[1][0] - rot[0][1]) / scale;
    } else if (rot[0][0] > rot[1][1] && rot[0][0] > rot[2][2]) {
        const double scale = sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0;
        quat[0] = (rot[2][1] - rot[1][2]) / scale;
        quat[1] = 0.25 * scale;
        quat[2] = (rot[0][1] + rot[1][0]) / scale;
        quat[3] = (rot[0][2] + rot[2][0]) / scale;
    } else if (rot[1][1] > rot[2][2]) {
        const double scale = sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0;
        quat[0] = (rot[0][2] - rot[2][0]) / scale;
        quat[1] = (rot[0][1] + rot[1][0]) / scale;
        quat[2] = 0.25 * scale;
        quat[3] = (rot[1][2] + rot[2][1]) / scale;
    } else {
        const double scale = sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0;
        quat[0] = (rot[1][0] - rot[0][1]) / scale;
        quat[1] = (rot[0][2] + rot[2][0]) / scale;
        quat[2] = (rot[1][2] + rot[2][1]) / scale;
        quat[3] = 0.25 * scale;
    }

    const double quat_norm = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    if (!(quat_norm > 1e-9) || !isfinite(quat_norm)) {
        fprintf(stderr, "Invalid quaternion norm %.9f from rotation matrix\n", quat_norm);
        exit(EXIT_FAILURE);
    }

    quat[0] /= quat_norm;
    quat[1] /= quat_norm;
    quat[2] /= quat_norm;
    quat[3] /= quat_norm;
}

static void compute_attitude_reference_from_acceleration(
    const double gravity[3],
    const double acceleration[3],
    double quat[4]
) {
    double body_z[3] = {
        acceleration[0] - gravity[0],
        acceleration[1] - gravity[1],
        acceleration[2] - gravity[2],
    };
    normalize_vector3_or_die("desired thrust direction", body_z);

    const double yaw_x[3] = {1.0, 0.0, 0.0};
    double body_y[3];
    cross3(body_z, yaw_x, body_y);
    normalize_vector3_or_die("desired body y", body_y);

    double body_x[3];
    cross3(body_y, body_z, body_x);
    normalize_vector3_or_die("desired body x", body_x);

    const double rot[3][3] = {
        {body_x[0], body_y[0], body_z[0]},
        {body_x[1], body_y[1], body_z[1]},
        {body_x[2], body_y[2], body_z[2]},
    };
    rotmat_to_quat(rot, quat);
}

static void apply_trajectory_reference_to_params(
    const trajectory_runtime_t *trajectory,
    double elapsed_s,
    double *params
) {
    if (trajectory->mode == TRAJECTORY_MODE_COLLISION) {
        return;
    }

    if (trajectory->mode != TRAJECTORY_MODE_CIRCLE) {
        fprintf(stderr, "Invalid trajectory mode: %d\n", trajectory->mode);
        exit(EXIT_FAILURE);
    }

    double position[3];
    double velocity[3];
    double acceleration[3];
    double quat_ref[4];
    if (!trajectory->initialized) {
        fprintf(stderr, "Trajectory runtime is not initialized\n");
        exit(EXIT_FAILURE);
    }

    compute_circle_reference(&trajectory->circle, trajectory->start_position, elapsed_s, position, velocity, acceleration);
    compute_attitude_reference_from_acceleration(&params[PARAM_GRAVITY], acceleration, quat_ref);
    memcpy(&params[PARAM_POS_REF], position, 3 * sizeof(double));
    memcpy(&params[PARAM_VEL_REF], velocity, 3 * sizeof(double));
    memcpy(&params[PARAM_ACC_REF], acceleration, 3 * sizeof(double));
    memcpy(&params[PARAM_QUAT_REF], quat_ref, 4 * sizeof(double));
}

static double clip_scalar(double value, double lower, double upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static void cross3(const double *a, const double *b, double *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void quat_to_rotmat(const double *quat, double rot[3][3]) {
    const double norm = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    if (!(norm > 1e-9) || !isfinite(norm)) {
        fprintf(stderr, "Invalid quaternion norm %.9f\n", norm);
        exit(EXIT_FAILURE);
    }

    const double qw = quat[0] / norm;
    const double qx = quat[1] / norm;
    const double qy = quat[2] / norm;
    const double qz = quat[3] / norm;

    rot[0][0] = 1.0 - 2.0 * (qy * qy + qz * qz);
    rot[0][1] = 2.0 * (qx * qy - qw * qz);
    rot[0][2] = 2.0 * (qx * qz + qw * qy);
    rot[1][0] = 2.0 * (qx * qy + qw * qz);
    rot[1][1] = 1.0 - 2.0 * (qx * qx + qz * qz);
    rot[1][2] = 2.0 * (qy * qz - qw * qx);
    rot[2][0] = 2.0 * (qx * qz - qw * qy);
    rot[2][1] = 2.0 * (qy * qz + qw * qx);
    rot[2][2] = 1.0 - 2.0 * (qx * qx + qy * qy);
}

static void quat_apply_world_yaw(double angle_rad, const double *q_in, double *q_out) {
    const double c = cos(0.5 * angle_rad);
    const double s = sin(0.5 * angle_rad);
    const double w = q_in[0];
    const double x = q_in[1];
    const double y = q_in[2];
    const double z = q_in[3];
    q_out[0] = c * w - s * z;
    q_out[1] = c * x - s * y;
    q_out[2] = c * y + s * x;
    q_out[3] = c * z + s * w;
}

static void apply_angle_offset_to_state(const trajectory_runtime_t *trajectory, double *x0) {
    const double cos_neg = trajectory->cos_neg;
    const double sin_neg = trajectory->sin_neg;
    const double cx = trajectory->gap_center[0];
    const double cy = trajectory->gap_center[1];

    const double dx = x0[0] - cx;
    const double dy = x0[1] - cy;
    x0[0] = cos_neg * dx - sin_neg * dy + cx;
    x0[1] = sin_neg * dx + cos_neg * dy + cy;

    const double vx = x0[3];
    const double vy = x0[4];
    x0[3] = cos_neg * vx - sin_neg * vy;
    x0[4] = sin_neg * vx + cos_neg * vy;

    double q_rot[4];
    quat_apply_world_yaw(-trajectory->angle_offset_rad, &x0[6], q_rot);
    memcpy(&x0[6], q_rot, 4 * sizeof(double));
}

static void mat3_vec_mul(const double mat[3][3], const double *vec, double *out) {
    for (int row = 0; row < 3; row++) {
        out[row] = mat[row][0] * vec[0] + mat[row][1] * vec[1] + mat[row][2] * vec[2];
    }
}

static void mat3_vec_mul_transpose(const double mat[3][3], const double *vec, double *out) {
    for (int row = 0; row < 3; row++) {
        out[row] = mat[0][row] * vec[0] + mat[1][row] * vec[1] + mat[2][row] * vec[2];
    }
}

static void build_inertia_matrix(const double *params, double inertia[3][3]) {
    inertia[0][0] = params[PARAM_IXX];
    inertia[0][1] = params[PARAM_IXY];
    inertia[0][2] = params[PARAM_IXZ];
    inertia[1][0] = params[PARAM_IXY];
    inertia[1][1] = params[PARAM_IYY];
    inertia[1][2] = params[PARAM_IYZ];
    inertia[2][0] = params[PARAM_IXZ];
    inertia[2][1] = params[PARAM_IYZ];
    inertia[2][2] = params[PARAM_IZZ];
}

static void set_disturbance_bounds(double *force_bound, double *torque_bound) {
    force_bound[0] = DIST_FORCE_BOUND_X;
    force_bound[1] = DIST_FORCE_BOUND_Y;
    force_bound[2] = DIST_FORCE_BOUND_Z;
    torque_bound[0] = DIST_TORQUE_BOUND_X;
    torque_bound[1] = DIST_TORQUE_BOUND_Y;
    torque_bound[2] = DIST_TORQUE_BOUND_Z;
}

static double get_hover_force(const double *params) {
    const double mass = params[PARAM_MASS];
    const double gx = params[PARAM_GRAVITY + 0];
    const double gy = params[PARAM_GRAVITY + 1];
    const double gz = params[PARAM_GRAVITY + 2];

    if (!(mass > 0.0)) {
        fprintf(stderr, "Invalid mass p[0]=%.9f\n", mass);
        exit(EXIT_FAILURE);
    }

    return mass * sqrt(gx * gx + gy * gy + gz * gz) / QUADROTOR_NU;
}

static void compute_hover_rps(const double *params, double *hover_rps) {
    const double hover_force = get_hover_force(params);

    for (int i = 0; i < QUADROTOR_NU; i++) {
        const double kf = params[PARAM_KF + i];

        if (!(kf > 0.0)) {
            fprintf(stderr, "Invalid thrust coefficient p[%d]=%.9f\n", PARAM_KF + i, kf);
            exit(EXIT_FAILURE);
        }

        hover_rps[i] = sqrt(hover_force / kf);
    }
}

static void convert_motor_forces_to_rps(const double *motor_forces, const double *params, double *motor_rps) {
    for (int i = 0; i < QUADROTOR_NU; i++) {
        const double kf = params[PARAM_KF + i];

        if (!(kf > 0.0)) {
            fprintf(stderr, "Invalid thrust coefficient p[%d]=%.9f\n", PARAM_KF + i, kf);
            exit(EXIT_FAILURE);
        }

        const double motor_force = motor_forces[i] > 0.0 ? motor_forces[i] : 0.0;
        motor_rps[i] = sqrt(motor_force / kf);
    }
}

static void compute_body_wrench_from_rps(const double *params, const double *motor_rps, double *force_body, double *torque_body) {
    double actual_forces[QUADROTOR_NU];

    for (int i = 0; i < QUADROTOR_NU; i++) {
        const double kf = params[PARAM_KF + i];

        if (!(kf > 0.0)) {
            fprintf(stderr, "Invalid thrust coefficient p[%d]=%.9f\n", PARAM_KF + i, kf);
            exit(EXIT_FAILURE);
        }

        actual_forces[i] = kf * motor_rps[i] * motor_rps[i];
    }

    for (int axis = 0; axis < 3; axis++) {
        force_body[axis] = 0.0;
        torque_body[axis] = 0.0;

        for (int motor = 0; motor < QUADROTOR_NU; motor++) {
            const int alloc_base = PARAM_ALLOC + 6 * motor;
            force_body[axis] += params[alloc_base + axis] * actual_forces[motor];
            torque_body[axis] += params[alloc_base + 3 + axis] * actual_forces[motor];
        }
    }
}

static void mat3_solve(const double mat[3][3], const double *vec, double *out) {
    const double det =
        mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
        mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
        mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

    if (!isfinite(det) || fabs(det) < 1e-12) {
        fprintf(stderr, "Singular 3x3 matrix in mat3_solve, det=%.12e\n", det);
        exit(EXIT_FAILURE);
    }

    const double inv[3][3] = {
        {
            (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) / det,
            (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) / det,
            (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) / det,
        },
        {
            (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) / det,
            (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) / det,
            (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / det,
        },
        {
            (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) / det,
            (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) / det,
            (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) / det,
        },
    };
    mat3_vec_mul(inv, vec, out);
}

// Continuous-time dynamics identical to the OCP model in nmpc_controller.py:
// x = [pos(3), vel(3), quat(4), omega(3), motor_rps(4)], with motor rps relaxing
// toward the commanded rps currently in effect.
static void initialize_motor_state_estimator(
    motor_state_estimator_t *estimator,
    const double *params,
    uint64_t sample_timestamp_us
) {
    double hover_rps[QUADROTOR_NU];
    compute_hover_rps(params, hover_rps);
    memcpy(estimator->rps_hat, hover_rps, sizeof(hover_rps));
    memcpy(estimator->commanded_rps_prev, hover_rps, sizeof(hover_rps));
    estimator->last_sample_timestamp_us = sample_timestamp_us;
    estimator->initialized = 1;
}

static void propagate_motor_state_estimator(
    motor_state_estimator_t *estimator,
    const double *params,
    uint64_t sample_timestamp_us
) {
    if (!estimator->initialized) {
        fprintf(stderr, "Motor state estimator is not initialized\n");
        exit(EXIT_FAILURE);
    }

    if (sample_timestamp_us < estimator->last_sample_timestamp_us) {
        fprintf(
            stderr,
            "State packet timestamp moved backwards: current=%llu previous=%llu\n",
            (unsigned long long)sample_timestamp_us,
            (unsigned long long)estimator->last_sample_timestamp_us
        );
        exit(EXIT_FAILURE);
    }

    const double dt = (double)(sample_timestamp_us - estimator->last_sample_timestamp_us) * 1e-6;

    for (int i = 0; i < QUADROTOR_NU; i++) {
        const double tau = params[PARAM_TAU + i];

        if (!(tau > 0.0)) {
            fprintf(stderr, "Invalid motor time constant p[%d]=%.9f\n", PARAM_TAU + i, tau);
            exit(EXIT_FAILURE);
        }

        const double desired_rps = estimator->commanded_rps_prev[i];
        const double current_rps = estimator->rps_hat[i];
        const double decay = exp(-dt / tau);
        estimator->rps_hat[i] = desired_rps + (current_rps - desired_rps) * decay;
    }

    estimator->last_sample_timestamp_us = sample_timestamp_us;
}

static int get_fresh_motor_rps_measurement(
    const state_packet_t *pkt,
    int motor_index,
    int64_t *measurement_age_us
) {
    *measurement_age_us = -1;

    const uint8_t valid_bit = (uint8_t)(1u << motor_index);
    if ((pkt->motor_rps_valid_mask & valid_bit) == 0u) {
        return 0;
    }

    const double measured_rps = pkt->motor_rps_meas[motor_index];
    const uint64_t measurement_timestamp_us = pkt->motor_rps_timestamp_us[motor_index];

    if (!isfinite(measured_rps) || measurement_timestamp_us == 0u) {
        return 0;
    }

    if (measurement_timestamp_us > pkt->sample_timestamp_us) {
        return 0;
    }

    const uint64_t age_us = pkt->sample_timestamp_us - measurement_timestamp_us;
    *measurement_age_us = (int64_t)age_us;
    return age_us <= MOTOR_RPS_MEAS_TIMEOUT_US;
}

static uint8_t correct_motor_state_estimator_with_measurements(
    motor_state_estimator_t *estimator,
    const state_packet_t *pkt,
    int prefer_measurement_seed,
    int64_t *measurement_age_us
) {
    uint8_t correction_mask = 0u;

    for (int motor_index = 0; motor_index < QUADROTOR_NU; motor_index++) {
        int64_t age_us = -1;
        measurement_age_us[motor_index] = -1;

        if (!get_fresh_motor_rps_measurement(pkt, motor_index, &age_us)) {
            continue;
        }

        measurement_age_us[motor_index] = age_us;

        if (prefer_measurement_seed) {
            estimator->rps_hat[motor_index] = pkt->motor_rps_meas[motor_index];
        } else {
            const double freshness = 1.0 - ((double)age_us / (double)MOTOR_RPS_MEAS_TIMEOUT_US);
            const double gain = clip_scalar(
                MOTOR_RPS_CORRECTION_GAIN_MAX * freshness,
                MOTOR_RPS_CORRECTION_GAIN_MIN,
                MOTOR_RPS_CORRECTION_GAIN_MAX
            );
            estimator->rps_hat[motor_index] += gain * (pkt->motor_rps_meas[motor_index] - estimator->rps_hat[motor_index]);
        }

        correction_mask |= (uint8_t)(1u << motor_index);
    }

    return correction_mask;
}

static void build_solver_state(
    const state_packet_t *pkt,
    const motor_state_estimator_t *estimator,
    double *x0
) {
    // The OCP is unchanged; the observer only improves the initial rotor-speed
    // estimate that populates x0[13:17].
    memcpy(x0, pkt->rigid_body_state, NRIGID * sizeof(double));
    memcpy(&x0[NRIGID], estimator->rps_hat, QUADROTOR_NU * sizeof(double));
}

static void store_commanded_rps(motor_state_estimator_t *estimator, const double *commanded_rps) {
    memcpy(estimator->commanded_rps_prev, commanded_rps, QUADROTOR_NU * sizeof(double));
}

static void initialize_disturbance_estimator(
    disturbance_estimator_t *estimator,
    const state_packet_t *pkt,
    const double *x0
) {
    memset(estimator, 0, sizeof(*estimator));
    estimator->last_force_timestamp_us = pkt->position_velocity_timestamp_us;
    estimator->last_torque_timestamp_us = pkt->angular_velocity_timestamp_us;
    memcpy(estimator->prev_velocity, &x0[3], 3 * sizeof(double));
    memcpy(estimator->prev_omega, &x0[10], 3 * sizeof(double));
    set_disturbance_bounds(estimator->estimated_force, estimator->estimated_torque);
    for (int axis = 0; axis < 3; axis++) {
        const double force_sigma = DIST_INITIAL_SIGMA_RATIO * estimator->estimated_force[axis];
        const double torque_sigma = DIST_INITIAL_SIGMA_RATIO * estimator->estimated_torque[axis];
        estimator->force_variance[axis] = force_sigma * force_sigma;
        estimator->torque_variance[axis] = torque_sigma * torque_sigma;
        estimator->estimated_force[axis] = 0.0;
        estimator->estimated_torque[axis] = 0.0;
        estimator->applied_force[axis] = 0.0;
        estimator->applied_torque[axis] = 0.0;
    }
    estimator->initialized = 1;
}

static void update_filter_state(
    double *estimated,
    double *applied,
    double *variance,
    const double *measurement,
    const double *bound,
    double dt_s,
    double process_sigma_ratio,
    double measurement_sigma_ratio,
    double abs_gate_ratio,
    int sample_count,
    double warmup_s
) {
    double innovation_nis = 0.0;
    int abs_gated = 0;
    int ready = 1;

    for (int axis = 0; axis < 3; axis++) {
        const double process_var = process_sigma_ratio * bound[axis];
        const double meas_sigma = measurement_sigma_ratio * bound[axis];
        const double min_sigma = DIST_VARIANCE_FLOOR_SIGMA_RATIO * bound[axis];
        const double min_var = min_sigma * min_sigma;
        const double max_var = bound[axis] * bound[axis];
        variance[axis] = clip_scalar(variance[axis] + process_var * process_var * dt_s, min_var, max_var);
        innovation_nis += (measurement[axis] - estimated[axis]) * (measurement[axis] - estimated[axis]) /
            (variance[axis] + meas_sigma * meas_sigma);
        if (fabs(measurement[axis]) > abs_gate_ratio * bound[axis]) {
            abs_gated = 1;
        }
    }

    const int nis_gated = innovation_nis > DIST_INNOVATION_GATE_NIS;
    const int gated = abs_gated || nis_gated;

    if (nis_gated && !abs_gated) {
        for (int axis = 0; axis < 3; axis++) {
            const double gated_process_var = DIST_GATED_PROCESS_SIGMA_RATIO * bound[axis];
            const double min_sigma = DIST_VARIANCE_FLOOR_SIGMA_RATIO * bound[axis];
            const double min_var = min_sigma * min_sigma;
            const double max_var = bound[axis] * bound[axis];
            variance[axis] = clip_scalar(
                variance[axis] + gated_process_var * gated_process_var * dt_s, min_var, max_var);
        }
    }

    if (!gated) {
        for (int axis = 0; axis < 3; axis++) {
            const double meas_sigma = measurement_sigma_ratio * bound[axis];
            const double min_sigma = DIST_VARIANCE_FLOOR_SIGMA_RATIO * bound[axis];
            const double min_var = min_sigma * min_sigma;
            const double max_var = bound[axis] * bound[axis];
            const double innovation = measurement[axis] - estimated[axis];
            const double innovation_cov = variance[axis] + meas_sigma * meas_sigma;
            const double gain = variance[axis] / innovation_cov;
            estimated[axis] += gain * innovation;
            variance[axis] = clip_scalar((1.0 - gain) * variance[axis], min_var, max_var);
        }
    }

    for (int axis = 0; axis < 3; axis++) {
        estimated[axis] = clip_scalar(estimated[axis], -bound[axis], bound[axis]);
        if (sqrt(variance[axis]) > DIST_READY_SIGMA_RATIO * bound[axis]) {
            ready = 0;
        }
    }

    if (!(warmup_s >= DIST_WARMUP_TIME_S && sample_count >= DIST_WARMUP_MIN_SAMPLES && ready)) {
        ready = 0;
    }

    if (ready) {
        for (int axis = 0; axis < 3; axis++) {
            const double step = DIST_SLEW_RATE_RATIO_PER_S * bound[axis] * dt_s;
            const double delta = clip_scalar(estimated[axis] - applied[axis], -step, step);
            applied[axis] = clip_scalar(applied[axis] + delta, -bound[axis], bound[axis]);
        }
    } else {
        for (int axis = 0; axis < 3; axis++) {
            applied[axis] = 0.0;
        }
    }
}

static void update_disturbance_estimator(
    disturbance_estimator_t *estimator,
    const state_packet_t *pkt,
    const double *x0,
    double *solver_params
) {
    double force_bound[3];
    double torque_bound[3];
    set_disturbance_bounds(force_bound, torque_bound);

    if (!estimator->initialized) {
        fprintf(stderr, "Disturbance estimator is not initialized\n");
        exit(EXIT_FAILURE);
    }

    if (pkt->position_velocity_timestamp_us < estimator->last_force_timestamp_us) {
        fprintf(stderr, "Position/velocity timestamp moved backwards: current=%llu previous=%llu\n",
            (unsigned long long)pkt->position_velocity_timestamp_us,
            (unsigned long long)estimator->last_force_timestamp_us);
        exit(EXIT_FAILURE);
    }

    if (pkt->angular_velocity_timestamp_us < estimator->last_torque_timestamp_us) {
        fprintf(stderr, "Angular-velocity timestamp moved backwards: current=%llu previous=%llu\n",
            (unsigned long long)pkt->angular_velocity_timestamp_us,
            (unsigned long long)estimator->last_torque_timestamp_us);
        exit(EXIT_FAILURE);
    }

    if (pkt->position_velocity_timestamp_us > estimator->last_force_timestamp_us) {
        const double dt_s = (double)(pkt->position_velocity_timestamp_us - estimator->last_force_timestamp_us) * 1e-6;
        const double mass = solver_params[PARAM_MASS];
        const double gravity[3] = {
            solver_params[PARAM_GRAVITY + 0],
            solver_params[PARAM_GRAVITY + 1],
            solver_params[PARAM_GRAVITY + 2],
        };
        double rotation[3][3];
        double force_body[3];
        double torque_body[3];
        double measured_world_acc[3];
        double measured_body_acc[3];
        double velocity_body[3];
        double drag_force_body[3];
        double gravity_body[3];
        double disturbance_force[3];
        quat_to_rotmat(&x0[6], rotation);
        compute_body_wrench_from_rps(solver_params, &x0[13], force_body, torque_body);

        for (int axis = 0; axis < 3; axis++) {
            measured_world_acc[axis] = (x0[3 + axis] - estimator->prev_velocity[axis]) / dt_s;
        }

        mat3_vec_mul_transpose(rotation, &x0[3], velocity_body);
        mat3_vec_mul_transpose(rotation, measured_world_acc, measured_body_acc);
        mat3_vec_mul_transpose(rotation, gravity, gravity_body);
        drag_force_body[0] = -BODY_DRAG_COEFF_BX * velocity_body[0];
        drag_force_body[1] = -BODY_DRAG_COEFF_BY * velocity_body[1];
        drag_force_body[2] = -BODY_DRAG_COEFF_BZ * velocity_body[2];

        for (int axis = 0; axis < 3; axis++) {
            disturbance_force[axis] =
                mass * (measured_body_acc[axis] - gravity_body[axis]) - force_body[axis] - drag_force_body[axis];
        }

        if (estimator->warmup_force_s + dt_s > DIST_STARTUP_BLANK_TIME_S) {
            update_filter_state(
                estimator->estimated_force,
                estimator->applied_force,
                estimator->force_variance,
                disturbance_force,
                force_bound,
                dt_s,
                DIST_FORCE_PROCESS_SIGMA_RATIO,
                DIST_FORCE_MEAS_SIGMA_RATIO,
                DIST_FORCE_ABS_GATE_RATIO,
                estimator->sample_count_force + 1,
                estimator->warmup_force_s + dt_s
            );
            estimator->sample_count_force++;
        }

        memcpy(estimator->prev_velocity, &x0[3], 3 * sizeof(double));
        estimator->last_force_timestamp_us = pkt->position_velocity_timestamp_us;
        estimator->warmup_force_s += dt_s;
    }

    if (pkt->angular_velocity_timestamp_us > estimator->last_torque_timestamp_us) {
        const double dt_s = (double)(pkt->angular_velocity_timestamp_us - estimator->last_torque_timestamp_us) * 1e-6;
        const double mass = solver_params[PARAM_MASS];
        const double gravity[3] = {
            solver_params[PARAM_GRAVITY + 0],
            solver_params[PARAM_GRAVITY + 1],
            solver_params[PARAM_GRAVITY + 2],
        };
        const double r_com[3] = {
            solver_params[PARAM_COM + 0],
            solver_params[PARAM_COM + 1],
            solver_params[PARAM_COM + 2],
        };
        double rotation[3][3];
        double inertia[3][3];
        double force_body[3];
        double torque_body[3];
        double omega[3] = {x0[10], x0[11], x0[12]};
        double omega_dot[3];
        double inertia_omega_dot[3];
        double inertia_omega[3];
        double omega_cross_inertia_omega[3];
        double gravity_force_world[3] = {mass * gravity[0], mass * gravity[1], mass * gravity[2]};
        double gravity_force_body[3];
        double torque_gravity[3];
        double disturbance_torque[3];

        quat_to_rotmat(&x0[6], rotation);
        build_inertia_matrix(solver_params, inertia);
        compute_body_wrench_from_rps(solver_params, &x0[13], force_body, torque_body);

        for (int axis = 0; axis < 3; axis++) {
            omega_dot[axis] = (omega[axis] - estimator->prev_omega[axis]) / dt_s;
        }

        mat3_vec_mul(inertia, omega_dot, inertia_omega_dot);
        mat3_vec_mul(inertia, omega, inertia_omega);
        cross3(omega, inertia_omega, omega_cross_inertia_omega);
        mat3_vec_mul_transpose(rotation, gravity_force_world, gravity_force_body);
        cross3(r_com, gravity_force_body, torque_gravity);

        for (int axis = 0; axis < 3; axis++) {
            disturbance_torque[axis] = inertia_omega_dot[axis] - torque_body[axis] - torque_gravity[axis] + omega_cross_inertia_omega[axis];
        }

        if (estimator->warmup_torque_s + dt_s > DIST_STARTUP_BLANK_TIME_S) {
            update_filter_state(
                estimator->estimated_torque,
                estimator->applied_torque,
                estimator->torque_variance,
                disturbance_torque,
                torque_bound,
                dt_s,
                DIST_TORQUE_PROCESS_SIGMA_RATIO,
                DIST_TORQUE_MEAS_SIGMA_RATIO,
                DIST_TORQUE_ABS_GATE_RATIO,
                estimator->sample_count_torque + 1,
                estimator->warmup_torque_s + dt_s
            );
            estimator->sample_count_torque++;
        }

        memcpy(estimator->prev_omega, omega, 3 * sizeof(double));
        estimator->last_torque_timestamp_us = pkt->angular_velocity_timestamp_us;
        estimator->warmup_torque_s += dt_s;
    }

    memcpy(&solver_params[PARAM_DIST_FORCE], estimator->applied_force, 3 * sizeof(double));
    memcpy(&solver_params[PARAM_DIST_TORQUE], estimator->applied_torque, 3 * sizeof(double));
    require_finite_array("solver_disturbance_force", &solver_params[PARAM_DIST_FORCE], 3);
    require_finite_array("solver_disturbance_torque", &solver_params[PARAM_DIST_TORQUE], 3);
}

static void write_solver_csv_header(FILE *csv) {
    fprintf(
        csv,
        "t_us,seq,packet_flags,status,solve_us,"
        "sample_timestamp_us,position_velocity_timestamp_us,attitude_timestamp_us,angular_velocity_timestamp_us,"
        "solver_u0,solver_u1,solver_u2,solver_u3,"
        "command_rps0,command_rps1,command_rps2,command_rps3,"
        "estimated_rps0,estimated_rps1,estimated_rps2,estimated_rps3,"
        "prev_command_rps0,prev_command_rps1,prev_command_rps2,prev_command_rps3,"
        "px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
        "sp_px,sp_py,sp_pz,sp_vx,sp_vy,sp_vz,sp_qw,sp_qx,sp_qy,sp_qz,"
        "fixed_dist_force_bx,fixed_dist_force_by,fixed_dist_force_bz,"
        "fixed_dist_torque_bx,fixed_dist_torque_by,fixed_dist_torque_bz,"
        "motor_rps_valid_mask,motor_rps_correction_mask,"
        "measured_rps0,measured_rps1,measured_rps2,measured_rps3,"
        "motor_rps_timestamp_us0,motor_rps_timestamp_us1,motor_rps_timestamp_us2,motor_rps_timestamp_us3,"
        "motor_rps_age_us0,motor_rps_age_us1,motor_rps_age_us2,motor_rps_age_us3,"
        "corrected_rps0,corrected_rps1,corrected_rps2,corrected_rps3"
    );

    for (int param_index = 0; param_index < QUADROTOR_NP; param_index++) {
        fprintf(csv, ",p%02d", param_index);
    }

    fprintf(csv, "\n");
}

static void write_solver_csv_row(
    FILE *csv,
    double solve_wall_time_us,
    const state_packet_t *pkt_in,
    const control_packet_t *pkt_out,
    const double *solver_u,
    const double *x0,
    const double *prev_command_rps,
    const double *solver_params,
    const int64_t *motor_rps_age_us,
    uint8_t motor_rps_correction_mask
) {
    fprintf(
        csv,
        "%.1f,%u,%u,%d,%.1f,"
        "%llu,%llu,%llu,%llu,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%u,%u,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%llu,%llu,%llu,%llu,"
        "%lld,%lld,%lld,%lld,"
        "%.6f,%.6f,%.6f,%.6f",
        solve_wall_time_us,
        pkt_in->seq,
        pkt_in->flags,
        pkt_out->status,
        pkt_out->solve_time_us,
        (unsigned long long)pkt_in->sample_timestamp_us,
        (unsigned long long)pkt_in->position_velocity_timestamp_us,
        (unsigned long long)pkt_in->attitude_timestamp_us,
        (unsigned long long)pkt_in->angular_velocity_timestamp_us,
        solver_u[0], solver_u[1], solver_u[2], solver_u[3],
        pkt_out->u[0], pkt_out->u[1], pkt_out->u[2], pkt_out->u[3],
        x0[13], x0[14], x0[15], x0[16],
        prev_command_rps[0], prev_command_rps[1], prev_command_rps[2], prev_command_rps[3],
        pkt_in->rigid_body_state[0], pkt_in->rigid_body_state[1], pkt_in->rigid_body_state[2],
        pkt_in->rigid_body_state[3], pkt_in->rigid_body_state[4], pkt_in->rigid_body_state[5],
        pkt_in->rigid_body_state[6], pkt_in->rigid_body_state[7], pkt_in->rigid_body_state[8], pkt_in->rigid_body_state[9],
        pkt_in->rigid_body_state[10], pkt_in->rigid_body_state[11], pkt_in->rigid_body_state[12],
        solver_params[PARAM_POS_REF + 0], solver_params[PARAM_POS_REF + 1], solver_params[PARAM_POS_REF + 2],
        solver_params[PARAM_POS_REF + 3], solver_params[PARAM_POS_REF + 4], solver_params[PARAM_POS_REF + 5],
        solver_params[PARAM_QUAT_REF + 0], solver_params[PARAM_QUAT_REF + 1], solver_params[PARAM_QUAT_REF + 2], solver_params[PARAM_QUAT_REF + 3],
        solver_params[PARAM_DIST_FORCE + 0], solver_params[PARAM_DIST_FORCE + 1], solver_params[PARAM_DIST_FORCE + 2],
        solver_params[PARAM_DIST_TORQUE + 0], solver_params[PARAM_DIST_TORQUE + 1], solver_params[PARAM_DIST_TORQUE + 2],
        pkt_in->motor_rps_valid_mask,
        motor_rps_correction_mask,
        pkt_in->motor_rps_meas[0], pkt_in->motor_rps_meas[1], pkt_in->motor_rps_meas[2], pkt_in->motor_rps_meas[3],
        (unsigned long long)pkt_in->motor_rps_timestamp_us[0],
        (unsigned long long)pkt_in->motor_rps_timestamp_us[1],
        (unsigned long long)pkt_in->motor_rps_timestamp_us[2],
        (unsigned long long)pkt_in->motor_rps_timestamp_us[3],
        (long long)motor_rps_age_us[0], (long long)motor_rps_age_us[1],
        (long long)motor_rps_age_us[2], (long long)motor_rps_age_us[3],
        x0[13], x0[14], x0[15], x0[16]
    );

    for (int param_index = 0; param_index < QUADROTOR_NP; param_index++) {
        fprintf(csv, ",%.9f", solver_params[param_index]);
    }

    fprintf(csv, "\n");
}

static void reinit_solver(nmpc_solver_runtime_t *solver, double *x0, double hover_force) {
    double u_init[QUADROTOR_NU];
    for (int i = 0; i < QUADROTOR_NU; i++) {
        u_init[i] = hover_force;
    }

    for (int k = 0; k < QUADROTOR_N; k++) {
        ocp_nlp_out_set(solver->config, solver->dims, solver->out, solver->in, k, "x", x0);
        ocp_nlp_out_set(solver->config, solver->dims, solver->out, solver->in, k, "u", u_init);
    }
    ocp_nlp_out_set(solver->config, solver->dims, solver->out, solver->in, QUADROTOR_N, "x", x0);
}

static void initialize_previous_horizon_controls(double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU], double hover_force) {
    for (int k = 0; k < QUADROTOR_N; k++) {
        for (int i = 0; i < QUADROTOR_NU; i++) {
            previous_horizon_controls[k][i] = hover_force;
        }
    }
}

static void update_previous_horizon_controls(
    nmpc_solver_runtime_t *solver,
    double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU]
) {
    for (int k = 0; k < QUADROTOR_N; k++) {
        ocp_nlp_out_get(solver->config, solver->dims, solver->out, k, "u", previous_horizon_controls[k]);
    }
}

static void set_stage_solver_params(
    nmpc_solver_runtime_t *solver,
    const double *solver_params,
    const double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU],
    const trajectory_runtime_t *trajectory,
    double trajectory_elapsed_s
) {
    double stage_params[QUADROTOR_NP];

    for (int k = 0; k <= QUADROTOR_N; k++) {
        memcpy(stage_params, solver_params, sizeof(stage_params));

        if (k > 0) {
            for (int i = 0; i < QUADROTOR_NU; i++) {
                stage_params[PARAM_U_PREV + i] = previous_horizon_controls[k - 1][i];
            }
        }

        apply_trajectory_reference_to_params(trajectory, trajectory_elapsed_s + (double)k * trajectory->nmpc_dt, stage_params);
        const double disturbance_decay = exp(-(double)k * trajectory->nmpc_dt / DIST_FORCE_HORIZON_DECAY_TAU_S);
        for (int axis = 0; axis < 3; axis++) {
            stage_params[PARAM_DIST_FORCE + axis] *= disturbance_decay;
        }
        solver->update_params(solver->capsule, k, stage_params, QUADROTOR_NP);
    }
}

static int run_benchmark(nmpc_solver_runtime_t *solver, int num_solves) {
    double x0[QUADROTOR_NX] = {0, 0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double p[QUADROTOR_NP];
    memset(p, 0, sizeof(p));
    p[PARAM_MASS] = 0.5;
    p[PARAM_IXX] = 0.001;
    p[PARAM_IYY] = 0.001;
    p[PARAM_IZZ] = 0.002;
    p[PARAM_GRAVITY + 2] = -9.81;
    p[PARAM_POS_REF + 2] = 1.0;
    p[PARAM_QUAT_REF] = 1.0;
    p[PARAM_ALLOC + 0] = 0.0; p[PARAM_ALLOC + 1] = 0.0; p[PARAM_ALLOC + 2] = 1.0;
    p[PARAM_ALLOC + 6] = 0.0; p[PARAM_ALLOC + 7] = 0.0; p[PARAM_ALLOC + 8] = 1.0;
    p[PARAM_ALLOC + 12] = 0.0; p[PARAM_ALLOC + 13] = 0.0; p[PARAM_ALLOC + 14] = 1.0;
    p[PARAM_ALLOC + 18] = 0.0; p[PARAM_ALLOC + 19] = 0.0; p[PARAM_ALLOC + 20] = 1.0;
    p[PARAM_KF + 0] = 1.0; p[PARAM_KF + 1] = 1.0; p[PARAM_KF + 2] = 1.0; p[PARAM_KF + 3] = 1.0;
    p[PARAM_TAU + 0] = 0.01; p[PARAM_TAU + 1] = 0.01; p[PARAM_TAU + 2] = 0.01; p[PARAM_TAU + 3] = 0.01;

    double hover_force = get_hover_force(p);
    double hover_rps[QUADROTOR_NU];
    compute_hover_rps(p, hover_rps);
    memcpy(&x0[NRIGID], hover_rps, sizeof(hover_rps));

    double lbx0[QUADROTOR_NX], ubx0[QUADROTOR_NX];
    memcpy(lbx0, x0, sizeof(x0));
    memcpy(ubx0, x0, sizeof(x0));
    ocp_nlp_constraints_model_set(solver->config, solver->dims, solver->in, solver->out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(solver->config, solver->dims, solver->in, solver->out, 0, "ubx", ubx0);

    for (int k = 0; k <= QUADROTOR_N; k++) {
        solver->update_params(solver->capsule, k, p, QUADROTOR_NP);
    }

    reinit_solver(solver, x0, hover_force);

    double *times = malloc(num_solves * sizeof(double));
    double total = 0;
    double min_t = 1e12, max_t = 0;

    for (int i = 0; i < num_solves; i++) {
        double t0 = get_time_us();
        solver->solve(solver->capsule);
        double t1 = get_time_us();
        times[i] = t1 - t0;
        total += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }

    double mean = total / num_solves;
    double var = 0;
    for (int i = 0; i < num_solves; i++) {
        var += (times[i] - mean) * (times[i] - mean);
    }
    var /= num_solves;

    printf("Benchmark: %d solves\n", num_solves);
    printf("  mean: %.1f us (%.3f ms)\n", mean, mean / 1000.0);
    printf("  min:  %.1f us (%.3f ms)\n", min_t, min_t / 1000.0);
    printf("  max:  %.1f us (%.3f ms)\n", max_t, max_t / 1000.0);
    printf("  std:  %.1f us\n", sqrt(var));
    printf("  100Hz feasible: %s\n", mean < 10000.0 ? "YES" : "NO");

    free(times);
    return 0;
}

static int read_exact(int fd, void *buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, (char *)buf + got, len - got);
        if (n <= 0) {
            return -1;
        }
        got += n;
    }
    return 0;
}

static int publish_run_manifest(const char *manifest_path, const char *tmp_path, const char *solver_csv_path) {
    FILE *manifest = fopen(tmp_path, "w");
    if (!manifest) {
        perror("fopen run manifest");
        return -1;
    }

    if (fprintf(manifest, "solver_csv=%s\n", solver_csv_path) < 0) {
        perror("fprintf run manifest");
        fclose(manifest);
        unlink(tmp_path);
        return -1;
    }

    if (fclose(manifest) != 0) {
        perror("fclose run manifest");
        unlink(tmp_path);
        return -1;
    }

    if (rename(tmp_path, manifest_path) != 0) {
        perror("rename run manifest");
        unlink(tmp_path);
        return -1;
    }

    return 0;
}

static void print_run_manifest(const char *manifest_path, const char *solver_csv_path) {
    printf("Active NMPC run manifest (%s):\n", manifest_path);
    printf("solver_csv=%s\n", solver_csv_path);
}

static int open_state_fifo(void) {
    mkdir(MPA_DIR, 0755);

    if (mkfifo(STATE_SINK_PATH, 0666) < 0 && errno != EEXIST) {
        perror("mkfifo nmpc_state");
        return -1;
    }

    printf("State FIFO ready at %s, waiting for bridge to connect...\n", STATE_SINK_PATH);
    int fd = open(STATE_SINK_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open nmpc_state");
        return -1;
    }
    printf("Bridge connected on state FIFO.\n");
    return fd;
}

static int open_control_fifo(void) {
    int fd = open(CONTROL_SINK_PATH, O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        if (errno != ENXIO && errno != ENOENT) {
            perror("open nmpc_control");
        }
        return -1;
    }
    printf("Control FIFO open at %s.\n", CONTROL_SINK_PATH);
    return fd;
}

static int run_pipe_server(
    nmpc_solver_runtime_t *nominal_solver,
    nmpc_solver_runtime_t *recovery_solver,
    trajectory_runtime_t *trajectory_runtime,
    double coast_motor_rps
) {
    int state_fd = -1;
    int control_fd = -1;
    int need_reinit = 1;
    int was_coasting = 0;
    int packet_logs_left = 5;
    int control_logs_left = 5;
    nmpc_solver_runtime_t *active_solver = NULL;
    motor_state_estimator_t motor_state_estimator;
    double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU];
    memset(&motor_state_estimator, 0, sizeof(motor_state_estimator));
    memset(previous_horizon_controls, 0, sizeof(previous_horizon_controls));
#if ENABLE_REALTIME_DISTURBANCE_ESTIMATION
    disturbance_estimator_t disturbance_estimator;
    memset(&disturbance_estimator, 0, sizeof(disturbance_estimator));
#endif

    unlink(RUN_MANIFEST_PATH);
    unlink(RUN_MANIFEST_TMP_PATH);

    char csv_path[128];
    {
        time_t now = time(NULL);
        strftime(csv_path, sizeof(csv_path), CSV_LOG_PATH_FMT, localtime(&now));
    }
    printf("Logging solver data to %s\n", csv_path);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("fopen solver csv");
        return 1;
    }
    write_solver_csv_header(csv);
    fflush(csv);

    if (publish_run_manifest(RUN_MANIFEST_PATH, RUN_MANIFEST_TMP_PATH, csv_path) != 0) {
        fclose(csv);
        return 1;
    }
    print_run_manifest(RUN_MANIFEST_PATH, csv_path);

    state_fd = open_state_fifo();
    if (state_fd < 0) {
        return 1;
    }

    printf("NMPC pipe solver running. Expecting state_packet_t: %lu bytes\n", sizeof(state_packet_t));
    printf("Sending control_packet_t: %lu bytes\n", sizeof(control_packet_t));

    state_packet_t pkt_in;
    control_packet_t pkt_out;
    int first_solve_logged = 0;

    while (1) {
        if (read_exact(state_fd, &pkt_in, sizeof(pkt_in)) < 0) {
            printf("State FIFO disconnected (EOF or error). Reopening...\n");
            close(state_fd);
            if (control_fd >= 0) {
                close(control_fd);
                control_fd = -1;
            }
            state_fd = open_state_fifo();
            if (state_fd < 0) {
                return 1;
            }
            need_reinit = 1;
            was_coasting = 0;
            active_solver = NULL;
            memset(&motor_state_estimator, 0, sizeof(motor_state_estimator));
            memset(previous_horizon_controls, 0, sizeof(previous_horizon_controls));
#if ENABLE_REALTIME_DISTURBANCE_ESTIMATION
            memset(&disturbance_estimator, 0, sizeof(disturbance_estimator));
#endif
            packet_logs_left = 5;
            continue;
        }

        if (pkt_in.sample_timestamp_us == 0) {
            fprintf(stderr, "State packet sample_timestamp_us is zero\n");
            exit(EXIT_FAILURE);
        }
        require_finite_array("packet_state", pkt_in.rigid_body_state, NRIGID);
        require_finite_array("packet_params", pkt_in.p, NP);

        double t0 = get_time_us();
        memset(&pkt_out, 0, sizeof(pkt_out));
        pkt_out.seq = pkt_in.seq;
        nmpc_solver_runtime_t *solver = (pkt_in.flags & FLAG_RECOVERY) ? recovery_solver : nominal_solver;
        const int coasting = (pkt_in.flags & FLAG_COAST) != 0;
        // Leaving coast counts as a mode change so the solver reinitializes from
        // the post-coast state instead of continuing a stale warm start.
        const int solver_mode_changed = !coasting && (active_solver != solver || was_coasting);

        if (coasting != was_coasting) {
            printf("NMPC coast %s: seq=%u flags=0x%02x command_rps=%.3f\n",
                   coasting ? "start" : "end", pkt_in.seq, pkt_in.flags, coast_motor_rps);
        }
        was_coasting = coasting;

        if (solver_mode_changed) {
            printf("NMPC mode switch: %s seq=%u flags=0x%02x\n", solver->name, pkt_in.seq, pkt_in.flags);
            active_solver = solver;
        }

        const int reset_motor_estimator = need_reinit || (pkt_in.flags & FLAG_REINIT);

        if (reset_motor_estimator) {
            initialize_motor_state_estimator(&motor_state_estimator, pkt_in.p, pkt_in.sample_timestamp_us);
        } else {
            propagate_motor_state_estimator(&motor_state_estimator, pkt_in.p, pkt_in.sample_timestamp_us);
        }
        if (reset_motor_estimator || solver_mode_changed) {
            reset_trajectory_runtime(trajectory_runtime, pkt_in.sample_timestamp_us, pkt_in.rigid_body_state);
        }
        const double trajectory_elapsed_s = get_trajectory_elapsed_s(trajectory_runtime, pkt_in.sample_timestamp_us);

        int64_t motor_rps_age_us[QUADROTOR_NU];
        const uint8_t motor_rps_correction_mask = correct_motor_state_estimator_with_measurements(
            &motor_state_estimator,
            &pkt_in,
            reset_motor_estimator,
            motor_rps_age_us
        );

        double x0[QUADROTOR_NX];
        build_solver_state(&pkt_in, &motor_state_estimator, x0);
        double solver_params[QUADROTOR_NP];
        memcpy(solver_params, pkt_in.p, sizeof(solver_params));

#if ENABLE_REALTIME_DISTURBANCE_ESTIMATION
        if (reset_motor_estimator) {
            initialize_disturbance_estimator(&disturbance_estimator, &pkt_in, x0);
            memcpy(&solver_params[PARAM_DIST_FORCE], disturbance_estimator.applied_force, 3 * sizeof(double));
            memcpy(&solver_params[PARAM_DIST_TORQUE], disturbance_estimator.applied_torque, 3 * sizeof(double));
        } else {
            update_disturbance_estimator(&disturbance_estimator, &pkt_in, x0, solver_params);
        }
#endif
        apply_trajectory_reference_to_params(trajectory_runtime, trajectory_elapsed_s, solver_params);

        if (trajectory_runtime->angle_offset_deg != 0.0) {
            apply_angle_offset_to_state(trajectory_runtime, x0);
        }

        if (packet_logs_left > 0) {
            double quat_norm = 0.0;
            for (int i = 6; i < 10; i++) {
                quat_norm += pkt_in.rigid_body_state[i] * pkt_in.rigid_body_state[i];
            }
            printf(
                "Packet %d seq=%u flags=0x%02x sample_timestamp_us=%llu quat_norm=%.6f "
                "pos=[%.6f %.6f %.6f] trajectory_start=[%.6f %.6f %.6f] sp_pos=[%.6f %.6f %.6f] "
                "body_q=[%.4f %.4f %.4f %.4f] sp_q=[%.4f %.4f %.4f %.4f] "
                "measured_rps=[%.6f %.6f %.6f %.6f] age_us=[%lld %lld %lld %lld] "
                "valid_mask=0x%02x corrected_mask=0x%02x corrected_rps=[%.6f %.6f %.6f %.6f] "
                "applied_dist_force=[%.4f %.4f %.4f] "
                "applied_dist_torque=[%.4f %.4f %.4f]\n",
                6 - packet_logs_left,
                pkt_in.seq,
                pkt_in.flags,
                (unsigned long long)pkt_in.sample_timestamp_us,
                sqrt(quat_norm),
                pkt_in.rigid_body_state[0], pkt_in.rigid_body_state[1], pkt_in.rigid_body_state[2],
                trajectory_runtime->start_position[0], trajectory_runtime->start_position[1], trajectory_runtime->start_position[2],
                solver_params[PARAM_POS_REF + 0], solver_params[PARAM_POS_REF + 1], solver_params[PARAM_POS_REF + 2],
                pkt_in.rigid_body_state[6], pkt_in.rigid_body_state[7], pkt_in.rigid_body_state[8], pkt_in.rigid_body_state[9],
                solver_params[PARAM_QUAT_REF + 0], solver_params[PARAM_QUAT_REF + 1], solver_params[PARAM_QUAT_REF + 2], solver_params[PARAM_QUAT_REF + 3],
                pkt_in.motor_rps_meas[0], pkt_in.motor_rps_meas[1], pkt_in.motor_rps_meas[2], pkt_in.motor_rps_meas[3],
                (long long)motor_rps_age_us[0], (long long)motor_rps_age_us[1],
                (long long)motor_rps_age_us[2], (long long)motor_rps_age_us[3],
                pkt_in.motor_rps_valid_mask,
                motor_rps_correction_mask,
                x0[13], x0[14], x0[15], x0[16],
                solver_params[PARAM_DIST_FORCE + 0], solver_params[PARAM_DIST_FORCE + 1], solver_params[PARAM_DIST_FORCE + 2],
                solver_params[PARAM_DIST_TORQUE + 0], solver_params[PARAM_DIST_TORQUE + 1], solver_params[PARAM_DIST_TORQUE + 2]
            );
            packet_logs_left--;
        }

        if (reset_motor_estimator || solver_mode_changed) {
            reinit_solver(solver, x0, get_hover_force(solver_params));
            initialize_previous_horizon_controls(previous_horizon_controls, get_hover_force(solver_params));
            need_reinit = 0;
        }

        double u_opt[QUADROTOR_NU] = {0.0};
        double x1[QUADROTOR_NX] = {0.0};
        double t1 = t0;

        if (coasting) {
            // Ballistic pass-through: no solve, all motors at the configured
            // minimum RPS. u_opt carries the equivalent motor forces so the
            // CSV log stays consistent with the applied commands.
            pkt_out.status = 0;
            for (int i = 0; i < QUADROTOR_NU; i++) {
                u_opt[i] = solver_params[PARAM_KF + i] * coast_motor_rps * coast_motor_rps;
            }
            memcpy(&x1[6], &x0[6], 4 * sizeof(double));
            t1 = get_time_us();
        } else {
            double lbx0[QUADROTOR_NX], ubx0[QUADROTOR_NX];
            memcpy(lbx0, x0, sizeof(lbx0));
            memcpy(ubx0, x0, sizeof(ubx0));
            ocp_nlp_constraints_model_set(solver->config, solver->dims, solver->in, solver->out, 0, "lbx", lbx0);
            ocp_nlp_constraints_model_set(solver->config, solver->dims, solver->in, solver->out, 0, "ubx", ubx0);

            set_stage_solver_params(solver, solver_params, previous_horizon_controls, trajectory_runtime, trajectory_elapsed_s);

            pkt_out.status = solver->solve(solver->capsule);
            t1 = get_time_us();

            ocp_nlp_out_get(solver->config, solver->dims, solver->out, 0, "u", u_opt);
            ocp_nlp_out_get(solver->config, solver->dims, solver->out, 1, "x", x1);
        }

        double prev_command_rps[QUADROTOR_NU];
        memcpy(prev_command_rps, motor_state_estimator.commanded_rps_prev, sizeof(prev_command_rps));
        convert_motor_forces_to_rps(u_opt, solver_params, pkt_out.u);
        store_commanded_rps(&motor_state_estimator, pkt_out.u);
        if (!coasting) {
            update_previous_horizon_controls(solver, previous_horizon_controls);
        }
        memcpy(pkt_out.quat_next, &x1[6], 4 * sizeof(double));
        pkt_out.solve_time_us = t1 - t0;

        if (control_logs_left > 0) {
            printf(
                "Control seq=%u solver_u=[%.6f %.6f %.6f %.6f] command_rps=[%.6f %.6f %.6f %.6f] "
                "measured_rps=[%.6f %.6f %.6f %.6f] age_us=[%lld %lld %lld %lld] "
                "valid_mask=0x%02x corrected_mask=0x%02x estimated_rps=[%.6f %.6f %.6f %.6f] applied_dist_force=[%.4f %.4f %.4f] "
                "applied_dist_torque=[%.4f %.4f %.4f]\n",
                pkt_out.seq,
                u_opt[0], u_opt[1], u_opt[2], u_opt[3],
                pkt_out.u[0], pkt_out.u[1], pkt_out.u[2], pkt_out.u[3],
                pkt_in.motor_rps_meas[0], pkt_in.motor_rps_meas[1], pkt_in.motor_rps_meas[2], pkt_in.motor_rps_meas[3],
                (long long)motor_rps_age_us[0], (long long)motor_rps_age_us[1],
                (long long)motor_rps_age_us[2], (long long)motor_rps_age_us[3],
                pkt_in.motor_rps_valid_mask,
                motor_rps_correction_mask,
                x0[13], x0[14], x0[15], x0[16],
                solver_params[PARAM_DIST_FORCE + 0], solver_params[PARAM_DIST_FORCE + 1], solver_params[PARAM_DIST_FORCE + 2],
                solver_params[PARAM_DIST_TORQUE + 0], solver_params[PARAM_DIST_TORQUE + 1], solver_params[PARAM_DIST_TORQUE + 2]
            );
            control_logs_left--;
        }

        write_solver_csv_row(
            csv,
            t1,
            &pkt_in,
            &pkt_out,
            u_opt,
            x0,
            prev_command_rps,
            solver_params,
            motor_rps_age_us,
            motor_rps_correction_mask
        );
        fflush(csv);

        if (control_fd < 0) {
            control_fd = open_control_fifo();
        }

        if (control_fd >= 0) {
            ssize_t n = write(control_fd, &pkt_out, sizeof(pkt_out));
            if (n < 0) {
                if (errno != EAGAIN) {
                    perror("write nmpc_control");
                }
                close(control_fd);
                control_fd = -1;
            } else if (!first_solve_logged) {
                printf(
                    "First solve seq=%u status=%d solve_time_us=%.1f solver_u=[%.6f %.6f %.6f %.6f] "
                    "command_rps=[%.6f %.6f %.6f %.6f] estimated_rps=[%.6f %.6f %.6f %.6f] "
                    "applied_dist_force=[%.4f %.4f %.4f] applied_dist_torque=[%.4f %.4f %.4f]\n",
                    pkt_out.seq, pkt_out.status, pkt_out.solve_time_us,
                    u_opt[0], u_opt[1], u_opt[2], u_opt[3],
                    pkt_out.u[0], pkt_out.u[1], pkt_out.u[2], pkt_out.u[3],
                    x0[13], x0[14], x0[15], x0[16],
                    solver_params[PARAM_DIST_FORCE + 0], solver_params[PARAM_DIST_FORCE + 1], solver_params[PARAM_DIST_FORCE + 2],
                    solver_params[PARAM_DIST_TORQUE + 0], solver_params[PARAM_DIST_TORQUE + 1], solver_params[PARAM_DIST_TORQUE + 2]
                );
                first_solve_logged = 1;
            }
        }
    }

    close(state_fd);
    if (control_fd >= 0) {
        close(control_fd);
    }
    if (csv) {
        fclose(csv);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    int num_solves = 1000;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
        } else if (strcmp(argv[i], "--num-solves") == 0 && i + 1 < argc) {
            num_solves = atoi(argv[++i]);
        }
    }

    quadrotor_nmpc_solver_capsule *nominal_capsule = quadrotor_nmpc_acados_create_capsule();
    int status = quadrotor_nmpc_acados_create_with_discretization(nominal_capsule, QUADROTOR_N, NULL);
    if (status) {
        fprintf(stderr, "nominal acados_create failed with status %d\n", status);
        return 1;
    }

    quadrotor_nmpc_recovery_solver_capsule *recovery_capsule = quadrotor_nmpc_recovery_acados_create_capsule();
    status = quadrotor_nmpc_recovery_acados_create_with_discretization(recovery_capsule, QUADROTOR_N, NULL);
    if (status) {
        fprintf(stderr, "recovery acados_create failed with status %d\n", status);
        quadrotor_nmpc_acados_free(nominal_capsule);
        quadrotor_nmpc_acados_free_capsule(nominal_capsule);
        return 1;
    }

    nmpc_solver_runtime_t nominal_solver = {
        .name = "nominal",
        .capsule = nominal_capsule,
        .nlp_solver = quadrotor_nmpc_acados_get_nlp_solver(nominal_capsule),
        .config = quadrotor_nmpc_acados_get_nlp_config(nominal_capsule),
        .dims = quadrotor_nmpc_acados_get_nlp_dims(nominal_capsule),
        .in = quadrotor_nmpc_acados_get_nlp_in(nominal_capsule),
        .out = quadrotor_nmpc_acados_get_nlp_out(nominal_capsule),
        .solve = solve_nominal,
        .update_params = update_params_nominal,
    };
    nmpc_solver_runtime_t recovery_solver = {
        .name = "recovery",
        .capsule = recovery_capsule,
        .nlp_solver = quadrotor_nmpc_recovery_acados_get_nlp_solver(recovery_capsule),
        .config = quadrotor_nmpc_recovery_acados_get_nlp_config(recovery_capsule),
        .dims = quadrotor_nmpc_recovery_acados_get_nlp_dims(recovery_capsule),
        .in = quadrotor_nmpc_recovery_acados_get_nlp_in(recovery_capsule),
        .out = quadrotor_nmpc_recovery_acados_get_nlp_out(recovery_capsule),
        .solve = solve_recovery,
        .update_params = update_params_recovery,
    };

    printf("Acados solvers created: NX=%d, NU=%d, NP=%d, N=%d\n",
           QUADROTOR_NX, QUADROTOR_NU, QUADROTOR_NP, QUADROTOR_N);

    nmpc_weight_config_t nominal_weights;
    nmpc_weight_config_t recovery_weights;
    trajectory_runtime_t trajectory_runtime;
    load_nmpc_weight_configs(NMPC_CONFIG_PATH, &nominal_weights, &recovery_weights);
    load_trajectory_runtime_config(NMPC_CONFIG_PATH, &trajectory_runtime);
    const double coast_motor_rps = load_motor_rps_min(NMPC_CONFIG_PATH);
    printf("Coast mode motor command: %.3f rps (nmpc.motor_rps_min)\n", coast_motor_rps);
    apply_nmpc_weights(&nominal_solver, &nominal_weights);
    apply_nmpc_weights(&recovery_solver, &recovery_weights);

    int ret;
    if (benchmark_mode) {
        ret = run_benchmark(&nominal_solver, num_solves);
    } else {
        ret = run_pipe_server(&nominal_solver, &recovery_solver, &trajectory_runtime, coast_motor_rps);
    }

    quadrotor_nmpc_recovery_acados_free(recovery_capsule);
    quadrotor_nmpc_recovery_acados_free_capsule(recovery_capsule);
    quadrotor_nmpc_acados_free(nominal_capsule);
    quadrotor_nmpc_acados_free_capsule(nominal_capsule);
    return ret;
}
