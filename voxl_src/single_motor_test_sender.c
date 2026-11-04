#define _POSIX_C_SOURCE 199309L
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "protocol.h"

#define MPA_DIR "/run/mpa"
#define STATE_PIPE "/run/mpa/nmpc_state"
#define CONTROL_PIPE "/run/mpa/nmpc_control"

static int read_exact(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) {
            return -1;
        }
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
    state_packet_t pkt_in;
    control_packet_t pkt_out;
    int control_fd = -1;
    struct timespec t_start;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <test_rps> <duration_s>\n", argv[0]);
        return 1;
    }

    double test_rps = atof(argv[1]);
    int duration_s = atoi(argv[2]);
    if (duration_s <= 0) {
        fprintf(stderr, "ERROR: duration_s must be positive.\n");
        return 1;
    }

    printf("Single-motor test: motor=0, rps=%.1f, duration=%ds\n", test_rps, duration_s);
    printf("Expecting state_packet_t: %lu bytes\n", sizeof(state_packet_t));
    printf("Sending control_packet_t: %lu bytes\n", sizeof(control_packet_t));

    int state_fd = open_state_fifo();
    if (state_fd < 0) {
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    printf("\n=== Testing NMPC motor 0 (u[0] = %.1f rps) ===\n", test_rps);

    while (1) {
        struct timespec now;
        double elapsed;

        if (read_exact(state_fd, &pkt_in, sizeof(pkt_in)) < 0) {
            fprintf(stderr, "State pipe closed.\n");
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed = (now.tv_sec - t_start.tv_sec) + (now.tv_nsec - t_start.tv_nsec) / 1e9;
        if (elapsed >= duration_s) {
            break;
        }

        memset(&pkt_out, 0, sizeof(pkt_out));
        pkt_out.seq = pkt_in.seq;
        pkt_out.status = 0;
        pkt_out.u[0] = test_rps;

        if (control_fd < 0) {
            control_fd = open(CONTROL_PIPE, O_WRONLY | O_NONBLOCK);
            if (control_fd >= 0) {
                printf("Control pipe open.\n");
            }
        }

        if (control_fd >= 0) {
            ssize_t n = write(control_fd, &pkt_out, sizeof(pkt_out));
            if (n < 0) {
                if (errno == EAGAIN) {
                    continue;
                }

                perror("write control");
                close(control_fd);
                control_fd = -1;
            }
        }
    }

    printf("\nSingle-motor test complete.\n");
    if (control_fd >= 0) {
        close(control_fd);
    }
    close(state_fd);
    return 0;
}
