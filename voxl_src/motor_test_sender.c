#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include "protocol.h"

#define MPA_DIR       "/run/mpa"
#define STATE_PIPE   "/run/mpa/nmpc_state"
#define CONTROL_PIPE "/run/mpa/nmpc_control"

static int read_exact(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) return -1;
        done += n;
    }
    return 0;
}

static int open_state_fifo(void)
{
    if (mkdir(MPA_DIR, 0755) < 0 && errno != EEXIST) {
        perror("mkdir /run/mpa");
        return -1;
    }

    if (mkfifo(STATE_PIPE, 0666) < 0 && errno != EEXIST) {
        perror("mkfifo state pipe");
        return -1;
    }

    printf("State FIFO ready at %s, waiting for bridge to connect...\n", STATE_PIPE);

    int state_fd = open(STATE_PIPE, O_RDONLY);
    if (state_fd < 0) {
        perror("open state pipe");
        return -1;
    }

    printf("State pipe open.\n");
    return state_fd;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc != 5) {
        fprintf(stderr, "Usage: %s <test_rps> <seconds_per_motor> <gap_seconds> <initial_delay_seconds>\n", argv[0]);
        return 1;
    }

    double test_rps = atof(argv[1]);
    int seconds_per_motor = atoi(argv[2]);
    double gap_seconds = atof(argv[3]);
    double initial_delay_seconds = atof(argv[4]);

    printf("Motor mapping test: rps=%.1f, %ds per motor, %.1fs gap, %.1fs initial delay\n",
           test_rps, seconds_per_motor, gap_seconds, initial_delay_seconds);
    printf("Expecting state_packet_t: %lu bytes\n", sizeof(state_packet_t));
    printf("Sending control_packet_t: %lu bytes\n", sizeof(control_packet_t));

    int state_fd = open_state_fifo();
    if (state_fd < 0) {
        return 1;
    }

    int control_fd = -1;

    state_packet_t pkt_in;
    control_packet_t pkt_out;

    int current_motor = 0;
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    int phase_printed = 0;
    int in_gap = 0;
    int in_initial_delay = initial_delay_seconds > 0.0;

    if (in_initial_delay) {
        printf("\n=== Initial delay before NMPC motor %d (%.1fs zero command) ===\n",
               current_motor, initial_delay_seconds);
    } else {
        printf("\n=== Testing NMPC motor %d ===\n", current_motor);
    }

    while (current_motor < 4) {
        if (read_exact(state_fd, &pkt_in, sizeof(pkt_in)) < 0) {
            fprintf(stderr, "State pipe closed.\n");
            break;
        }

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - t_start.tv_sec) + (now.tv_nsec - t_start.tv_nsec) / 1e9;

        memset(&pkt_out, 0, sizeof(pkt_out));
        pkt_out.seq = pkt_in.seq;
        pkt_out.status = 0;

        if (in_initial_delay) {
            if (elapsed >= initial_delay_seconds) {
                clock_gettime(CLOCK_MONOTONIC, &t_start);
                phase_printed = 0;
                in_initial_delay = 0;
            }

        } else if (in_gap) {
            if (!phase_printed) {
                printf("=== Gap after motor %d (%.1fs zero command) ===\n", current_motor, gap_seconds);
                phase_printed = 1;
            }

            if (elapsed >= gap_seconds) {
                current_motor++;
                if (current_motor >= 4) break;
                clock_gettime(CLOCK_MONOTONIC, &t_start);
                phase_printed = 0;
                in_gap = 0;
            }
        } else {
            if (elapsed >= seconds_per_motor) {
                printf("=== Done motor %d ===\n\n", current_motor);
                if (gap_seconds > 0.0 && current_motor < 3) {
                    clock_gettime(CLOCK_MONOTONIC, &t_start);
                    phase_printed = 0;
                    in_gap = 1;
                } else {
                    current_motor++;
                    if (current_motor >= 4) break;
                    clock_gettime(CLOCK_MONOTONIC, &t_start);
                    phase_printed = 0;
                }
            } else {
                if (!phase_printed) {
                    printf("=== Testing NMPC motor %d (u[%d] = %.1f rps) ===\n", current_motor, current_motor, test_rps);
                    phase_printed = 1;
                }
                pkt_out.u[current_motor] = test_rps;
            }
        }

        if (control_fd < 0) {
            control_fd = open(CONTROL_PIPE, O_WRONLY | O_NONBLOCK);
            if (control_fd >= 0) printf("Control pipe open.\n");
        }

        if (control_fd >= 0) {
            ssize_t n = write(control_fd, &pkt_out, sizeof(pkt_out));
            if (n < 0 && errno != EAGAIN) {
                perror("write control");
                close(control_fd);
                control_fd = -1;
            }
        }
    }

    printf("\nMotor test complete.\n");
    if (control_fd >= 0) close(control_fd);
    close(state_fd);
    return 0;
}
