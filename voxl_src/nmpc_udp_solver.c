#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>

#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_quadrotor_nmpc.h"

#include <math.h>
#include "protocol.h"

#define QUADROTOR_NX QUADROTOR_NMPC_NX
#define QUADROTOR_NU QUADROTOR_NMPC_NU
#define QUADROTOR_NP QUADROTOR_NMPC_NP
#define QUADROTOR_N  QUADROTOR_NMPC_N
#define PARAM_KF 47
#define PARAM_TAU 51
#define PARAM_U_PREV 64

typedef struct {
    double rps_hat[QUADROTOR_NU];
    double commanded_rps_prev[QUADROTOR_NU];
    uint64_t last_sample_timestamp_us;
    int initialized;
} motor_state_estimator_t;

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double get_hover_force(const double *params) {
    const double mass = params[0];
    const double gx = params[7];
    const double gy = params[8];
    const double gz = params[9];

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

static void build_solver_state(
    const state_packet_t *pkt,
    const motor_state_estimator_t *estimator,
    double *x0
) {
    memcpy(x0, pkt->rigid_body_state, NRIGID * sizeof(double));
    memcpy(&x0[NRIGID], estimator->rps_hat, QUADROTOR_NU * sizeof(double));
}

static void store_commanded_rps(motor_state_estimator_t *estimator, const double *commanded_rps) {
    memcpy(estimator->commanded_rps_prev, commanded_rps, QUADROTOR_NU * sizeof(double));
}

static void reinit_solver(quadrotor_nmpc_solver_capsule *capsule, double *x0, double hover_force) {
    ocp_nlp_config *config = quadrotor_nmpc_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = quadrotor_nmpc_acados_get_nlp_dims(capsule);
    ocp_nlp_in *in = quadrotor_nmpc_acados_get_nlp_in(capsule);
    ocp_nlp_out *out = quadrotor_nmpc_acados_get_nlp_out(capsule);

    double u_init[QUADROTOR_NU];
    for (int i = 0; i < QUADROTOR_NU; i++) {
        u_init[i] = hover_force;
    }

    for (int k = 0; k < QUADROTOR_N; k++) {
        ocp_nlp_out_set(config, dims, out, in, k, "x", x0);
        ocp_nlp_out_set(config, dims, out, in, k, "u", u_init);
    }
    ocp_nlp_out_set(config, dims, out, in, QUADROTOR_N, "x", x0);
}

static void initialize_previous_horizon_controls(double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU], double hover_force) {
    for (int k = 0; k < QUADROTOR_N; k++) {
        for (int i = 0; i < QUADROTOR_NU; i++) {
            previous_horizon_controls[k][i] = hover_force;
        }
    }
}

static void update_previous_horizon_controls(
    quadrotor_nmpc_solver_capsule *capsule,
    double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU]
) {
    ocp_nlp_config *config = quadrotor_nmpc_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = quadrotor_nmpc_acados_get_nlp_dims(capsule);
    ocp_nlp_out *out = quadrotor_nmpc_acados_get_nlp_out(capsule);

    for (int k = 0; k < QUADROTOR_N; k++) {
        ocp_nlp_out_get(config, dims, out, k, "u", previous_horizon_controls[k]);
    }
}

static void set_stage_previous_control_params(
    quadrotor_nmpc_solver_capsule *capsule,
    const double *solver_params,
    const double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU]
) {
    double stage_params[QUADROTOR_NP];
    memcpy(stage_params, solver_params, sizeof(stage_params));
    quadrotor_nmpc_acados_update_params(capsule, 0, stage_params, QUADROTOR_NP);

    for (int k = 1; k <= QUADROTOR_N; k++) {
        for (int i = 0; i < QUADROTOR_NU; i++) {
            stage_params[PARAM_U_PREV + i] = previous_horizon_controls[k - 1][i];
        }
        quadrotor_nmpc_acados_update_params(capsule, k, stage_params, QUADROTOR_NP);
    }
}

static int run_benchmark(quadrotor_nmpc_solver_capsule *capsule, int num_solves) {
    ocp_nlp_config *config = quadrotor_nmpc_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = quadrotor_nmpc_acados_get_nlp_dims(capsule);
    ocp_nlp_in *in = quadrotor_nmpc_acados_get_nlp_in(capsule);
    ocp_nlp_out *out = quadrotor_nmpc_acados_get_nlp_out(capsule);

    double x0[QUADROTOR_NX] = {0, 0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double p[QUADROTOR_NP];
    memset(p, 0, sizeof(p));
    p[0] = 0.5;
    p[1] = 0.001; p[4] = 0.001; p[6] = 0.002;
    p[8] = -9.81;
    p[12] = 1.0;
    p[19] = 1.0;
    p[23] = 0; p[24] = 0; p[25] = 1;
    p[29] = 0; p[30] = 0; p[31] = 1;
    p[35] = 0; p[36] = 0; p[37] = 1;
    p[41] = 0; p[42] = 0; p[43] = 1;
    p[47] = 1.0; p[48] = 1.0; p[49] = 1.0; p[50] = 1.0;
    p[51] = 0.01; p[52] = 0.01; p[53] = 0.01; p[54] = 0.01;

    double hover_force = get_hover_force(p);
    double hover_rps[QUADROTOR_NU];
    compute_hover_rps(p, hover_rps);
    memcpy(&x0[NRIGID], hover_rps, sizeof(hover_rps));

    double lbx0[QUADROTOR_NX], ubx0[QUADROTOR_NX];
    memcpy(lbx0, x0, sizeof(x0));
    memcpy(ubx0, x0, sizeof(x0));
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);

    for (int k = 0; k <= QUADROTOR_N; k++) {
        quadrotor_nmpc_acados_update_params(capsule, k, p, QUADROTOR_NP);
    }

    reinit_solver(capsule, x0, hover_force);

    double *times = malloc(num_solves * sizeof(double));
    double total = 0;
    double min_t = 1e12, max_t = 0;

    for (int i = 0; i < num_solves; i++) {
        double t0 = get_time_us();
        quadrotor_nmpc_acados_solve(capsule);
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

static int recv_exact(int fd, void *buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, (char *)buf + got, len - got, 0);
        if (n <= 0) {
            return -1;
        }
        got += n;
    }
    return 0;
}

static int send_exact(int fd, const void *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, (const char *)buf + sent, len - sent, 0);
        if (n <= 0) {
            return -1;
        }
        sent += n;
    }
    return 0;
}

static int run_tcp_server(quadrotor_nmpc_solver_capsule *capsule, int port) {
    ocp_nlp_config *config = quadrotor_nmpc_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = quadrotor_nmpc_acados_get_nlp_dims(capsule);
    ocp_nlp_in *in = quadrotor_nmpc_acados_get_nlp_in(capsule);
    ocp_nlp_out *out = quadrotor_nmpc_acados_get_nlp_out(capsule);

    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(listenfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(listenfd);
        return 1;
    }

    if (listen(listenfd, 1) < 0) {
        perror("listen");
        close(listenfd);
        return 1;
    }

    printf("NMPC TCP solver listening on port %d\n", port);
    printf("Expecting state_packet_t: %lu bytes\n", sizeof(state_packet_t));
    printf("Sending control_packet_t: %lu bytes\n", sizeof(control_packet_t));

    while (1) {
        printf("Waiting for client connection...\n");
        int clientfd = accept(listenfd, NULL, NULL);
        if (clientfd < 0) {
            perror("accept");
            continue;
        }
        printf("Client connected.\n");

        int nodelay = 1;
        setsockopt(clientfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        state_packet_t pkt_in;
        control_packet_t pkt_out;
        motor_state_estimator_t motor_state_estimator;
        double previous_horizon_controls[QUADROTOR_N][QUADROTOR_NU];
        memset(&motor_state_estimator, 0, sizeof(motor_state_estimator));
        memset(previous_horizon_controls, 0, sizeof(previous_horizon_controls));
        int need_reinit = 1;

        while (1) {
            if (recv_exact(clientfd, &pkt_in, sizeof(pkt_in)) < 0) {
                printf("Client disconnected.\n");
                break;
            }

            double t0 = get_time_us();
            memset(&pkt_out, 0, sizeof(pkt_out));
            pkt_out.seq = pkt_in.seq;

            if (need_reinit || (pkt_in.flags & FLAG_REINIT)) {
                initialize_motor_state_estimator(&motor_state_estimator, pkt_in.p, pkt_in.sample_timestamp_us);
            } else {
                propagate_motor_state_estimator(&motor_state_estimator, pkt_in.p, pkt_in.sample_timestamp_us);
            }

            double x0[QUADROTOR_NX];
            build_solver_state(&pkt_in, &motor_state_estimator, x0);

            if (need_reinit || (pkt_in.flags & FLAG_REINIT)) {
                reinit_solver(capsule, x0, get_hover_force(pkt_in.p));
                initialize_previous_horizon_controls(previous_horizon_controls, get_hover_force(pkt_in.p));
                need_reinit = 0;
            }

            double u_opt[QUADROTOR_NU] = {0.0};
            double x1[QUADROTOR_NX] = {0.0};
            double t1 = t0;

            double lbx0[QUADROTOR_NX], ubx0[QUADROTOR_NX];
            memcpy(lbx0, x0, sizeof(lbx0));
            memcpy(ubx0, x0, sizeof(ubx0));
            ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
            ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);

            set_stage_previous_control_params(capsule, pkt_in.p, previous_horizon_controls);

            pkt_out.status = quadrotor_nmpc_acados_solve(capsule);
            t1 = get_time_us();

            ocp_nlp_out_get(config, dims, out, 0, "u", u_opt);
            ocp_nlp_out_get(config, dims, out, 1, "x", x1);
            convert_motor_forces_to_rps(u_opt, pkt_in.p, pkt_out.u);
            store_commanded_rps(&motor_state_estimator, pkt_out.u);
            update_previous_horizon_controls(capsule, previous_horizon_controls);
            memcpy(pkt_out.quat_next, &x1[6], 4 * sizeof(double));
            pkt_out.solve_time_us = t1 - t0;

            if (send_exact(clientfd, &pkt_out, sizeof(pkt_out)) < 0) {
                printf("Client disconnected (send).\n");
                break;
            }
        }

        close(clientfd);
    }

    close(listenfd);
    return 0;
}

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    int num_solves = 1000;
    int port = 5555;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
        } else if (strcmp(argv[i], "--num-solves") == 0 && i + 1 < argc) {
            num_solves = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        }
    }

    quadrotor_nmpc_solver_capsule *capsule = quadrotor_nmpc_acados_create_capsule();
    int status = quadrotor_nmpc_acados_create_with_discretization(capsule, QUADROTOR_N, NULL);
    if (status) {
        fprintf(stderr, "acados_create failed with status %d\n", status);
        return 1;
    }
    printf("Acados solver created: NX=%d, NU=%d, NP=%d, N=%d\n",
           QUADROTOR_NX, QUADROTOR_NU, QUADROTOR_NP, QUADROTOR_N);

    int ret;
    if (benchmark_mode) {
        ret = run_benchmark(capsule, num_solves);
    } else {
        ret = run_tcp_server(capsule, port);
    }

    quadrotor_nmpc_acados_free(capsule);
    quadrotor_nmpc_acados_free_capsule(capsule);
    return ret;
}
