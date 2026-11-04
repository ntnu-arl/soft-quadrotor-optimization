#!/usr/bin/python3

# Copyright (c) 2025 ModalAI Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 4. The Software is used solely in conjunction with devices provided by
#    ModalAI Inc.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# For a license to use on non-ModalAI hardware, please contact license@modalai.com


from voxl_esc_setup_paths import *
voxl_esc_setup_paths()

from libesc import *
from esc_scanner import EscScanner
import time
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='ESC Test Spin Script')
parser.add_argument('--device',                 required=False, default=None)
parser.add_argument('--baud-rate',              required=False, default=None)
parser.add_argument('--id',          type=int,  required=True,  default=0)
parser.add_argument('--power',       type=int,  required=False, default=10)
parser.add_argument('--rpm',         type=int,  required=False, default=None)
parser.add_argument('--timeout',     type=float,  required=False, default=10000)
parser.add_argument('--skip-prompt', type=str,  required=False, default='False')
parser.add_argument('--led-red',     type=int,  required=False, default=0)
parser.add_argument('--led-green',   type=int,  required=False, default=0)
parser.add_argument('--led-blue',    type=int,  required=False, default=0)
parser.add_argument('--cmd-rate',    type=int,  required=False, default=100)
parser.add_argument('--ramp-time',   type=float,required=False, default=0.0)
parser.add_argument('--step-amplitude',   type=float,required=False, default=0.0)
parser.add_argument('--step-amplitudes',  type=str,  required=False, default=None,
                    help='Comma-separated list of step amplitudes to cycle through, e.g. "2000,5000,10000,20000". Overrides --step-amplitude.')
parser.add_argument('--step-frequency',   type=float,required=False, default=0.25)
parser.add_argument('--slew-rate-up', type=float,  required=False, default=0.0)
parser.add_argument('--slew-rate-down', type=float,  required=False, default=0.0)
parser.add_argument('--enable-plot', type=int,  required=False, default=0)
parser.add_argument('--waveform-type', type=str,  required=False, default='square')
parser.add_argument('--enable-logging', type=int,  required=False, default=0)
parser.add_argument('--step-delay',   type=float,required=False, default=1.0)
parser.add_argument('--temperature-limit', type=float, required=False, default=100.0)  #degrees C
parser.add_argument('--save-csv',     type=str,  required=False, default=None,
                    help='Path to save recorded data as CSV, e.g. motor_step_response.csv')
args = parser.parse_args()

devpath  = args.device
baudrate = args.baud_rate
esc_id   = args.id
spin_pwr = args.power #0-100
spin_rpm = args.rpm #0-30000 .. limited to 30K for safety
timeout  = args.timeout
led_red  = int(args.led_red > 0)
led_green= int(args.led_green > 0)
led_blue = int(args.led_blue > 0)
cmd_rate = args.cmd_rate
ramp_time= args.ramp_time
step_amplitude  = args.step_amplitude
step_amplitudes = [float(a) for a in args.step_amplitudes.split(',')] if args.step_amplitudes else None
step_frequency  = args.step_frequency
step_delay      = args.step_delay
enable_plot     = args.enable_plot
save_csv        = args.save_csv
slew_rate_up   = args.slew_rate_up
slew_rate_down = args.slew_rate_down
waveform_type  = args.waveform_type #'sinusoidal'
enable_logging = bool(args.enable_logging)
temperature_limit = args.temperature_limit

#optionally skip the safety prompt that asks to enter "yes" before spinning
skip_prompt = 'True' in args.skip_prompt or 'true' in args.skip_prompt

if devpath is not None and baudrate is None:
    print('ERROR: Please provide baud rate with --baud-rate option')
    sys.exit(1)

if spin_pwr < 0 or spin_pwr > 100:
    print("ERROR: Spin power must be between 0 and 100")
    sys.exit(1)

if spin_rpm is not None and (spin_rpm < 0 or spin_rpm > 30000):
    print("ERROR: Spin rpm must be between 0 and 30000")
    sys.exit(1)

if timeout < 0:
    print("ERROR: Timeout should be non-negative value of seconds")
    sys.exit(1)

# quick scan for ESCs to detect the port
scanner = EscScanner()
(devpath, baudrate) = scanner.scan(devpath, baudrate)

if devpath is not None and baudrate is not None:
    print('INFO: ESC(s) detected on port: ' + devpath + ', baud rate: ' + str(baudrate))
else:
    print('ERROR: No ESC(s) detected, exiting.')
    sys.exit(1)


# create ESC manager and search for ESCs
try:
    esc_manager = EscManager()
    esc_manager.open(devpath, baudrate)
except Exception as e:
    print('ERROR: Unable to connect to ESCs :')
    print(e)
    sys.exit(1)

# wait a little to let manager find all ESCs
time.sleep(0.25)
num_escs = len(esc_manager.get_escs())
if num_escs < 1:
    print('ERROR: No ESCs detected--exiting.')
    sys.exit(1)

escs = []
if esc_id != 255:
    esc = esc_manager.get_esc_by_id(esc_id)
    if esc is None:
        print('ERROR: Specified ESC ID not found--exiting.')
        sys.exit(1)
    escs.append(esc)
else:
    escs = esc_manager.get_escs()

# warn user
if not skip_prompt:
    print('WARNING: ')
    print('This test requires motors to spin at high speeds with')
    print('propellers attached. Please ensure that appropriate')
    print('protective equipment is being worn at all times and')
    print('that the motor and propeller are adequately isolated')
    print('from all persons.')
    print('')
    print('For best results, please perform this test at the')
    print('nominal voltage for the battery used.')
    print('')
    response = input('Type "Yes" to continue: ')
    if response not in ['yes', 'Yes', 'YES']:
        print('Test canceled by user')
        sys.exit(1)

if enable_logging:
    esc_manager.enable_protocol_logging()
#reduce sleep time of feedback reading thread to 1ms
#esc_manager.set_rx_sleep_time(0.001)
esc_manager.set_rx_sleep_time(1.0/cmd_rate)
esc_manager.enable_debug_msgs()


# spin up
if esc_id != 255:
    esc_manager.set_highspeed_fb(esc_id)  #tell ESC manager to only request feedback from this ID (so we get feedback 4x more often)

for esc in escs:
    esc.set_leds([led_red, led_green, led_blue])  #0 or 1 for R G and B values.. binary for now


rpm_ts          = [[],[],[],[]];
rpm_log         = [[],[],[],[]];
rpm_target_log  = [[],[],[],[]];
rpm_target_ts   = [[],[],[],[]];
volt_log        = [[],[],[],[]];
curr_log        = [[],[],[],[]];
cmd_log         = [[],[],[],[]];
temp_log        = [[],[],[],[]];
board_curr_log = []
board_curr_ts_log = []
csv_rows = []  # flat list of dicts for CSV export

pwr_last = spin_pwr;

update_cntr = 0
t_start     = time.time()
t_last      = t_start
dt_desired  = 1.0/cmd_rate
t_next      = t_start + dt_desired

while timeout == 0 or time.time() - t_start < timeout:
    update_cntr   += 1
    t_now          = time.time()
    t_sleep        = (t_next - t_now)
    t_next        += dt_desired

    if (t_sleep > 0):
        time.sleep(t_sleep)
    t_after_sleep  = time.time()
    dt_after_sleep = t_after_sleep - t_now

    if (ramp_time == 0.0):
        ramp_percentage = 1.0
    else:
        ramp_percentage = (t_now-t_start)/ramp_time
        #ramp_percentage = float(update_cntr) / (cmd_rate * ramp_time)
        if ramp_percentage > 1.0:
            ramp_percentage = 1.0
        #ramp_percentage = 1.0

    dt = time.time() - t_start
    dt_delayed = dt - step_delay

    if dt_delayed < 0:
        dt_delayed = 0

    sin_val      = np.sin(2*np.pi*step_frequency*dt_delayed)  #-1...1

    if waveform_type != 'sinusoidal':
        if sin_val > 0.0:
            sin_val = 1.0
        else:
            sin_val = 0.0


    if spin_rpm is not None:
        if step_amplitudes is not None:
            # cycle through amplitudes: each occupies one full square-wave period (1/step_frequency)
            period      = 1.0 / step_frequency
            elapsed     = max(0.0, dt - step_delay)
            amp_idx     = int(elapsed / period) % len(step_amplitudes)
            phase_in_period = (elapsed % period) / period  # 0..1
            cur_amplitude = step_amplitudes[amp_idx]
            sq = 1.0 if phase_in_period < 0.5 else 0.0
            spin_rpm_now = spin_rpm + cur_amplitude * sq
        else:
            spin_rpm_now = spin_rpm + step_amplitude * sin_val
        spin_rpm_now *= ramp_percentage
        for esc in escs:
            esc.set_target_rpm(spin_rpm_now)
            rpm_target_log[esc.get_id()].append(spin_rpm_now)
            rpm_target_ts[esc.get_id()].append(t_now - t_start)
        esc_manager.send_rpm_targets()
    else:
        spin_pwr_now  = spin_pwr + step_amplitude * sin_val
        spin_pwr_now *= ramp_percentage
        #spin_pwr_now = spin_pwr * sin_val
        #if spin_pwr_now < 10:
        #    spin_pwr_now = 10

        #spin_pwr_now = -spin_pwr
        #if sin_val > 0.5:
        #   spin_pwr_now = spin_pwr


        dt_cmd = t_now - t_last
        dp_dt = (spin_pwr_now - pwr_last) / (dt_cmd)
        if slew_rate_up != 0.0 and dp_dt > slew_rate_up:       #apply slew rate in increasing direction (if needed)
            spin_pwr_now = pwr_last + slew_rate_up * dt_cmd
        if slew_rate_down != 0.0 and dp_dt < -slew_rate_down:  #apply slew rate in decreasing direction (if needed)
            spin_pwr_now = pwr_last - slew_rate_down * dt_cmd

        pwr_last = spin_pwr_now
        t_last   = t_now

        #print(spin_pwr_now)
        for esc in escs:
            esc.set_target_power(spin_pwr_now)
        esc_manager.send_pwm_targets()

    board_voltage = esc_manager.get_board_voltage()
    board_current = esc_manager.get_board_current()

    for esc in escs:
        esc_id = esc.get_id()
        #if update_cntr > 15:  #skip a few initial iterations because the feedback does not come back right away and we end up saving zeros..
        if esc.get_last_time() != 0: #wait for feedback data to come in
            rpm_log[esc_id].append(esc.get_rpm())
            cmd_log[esc_id].append(esc.get_power())
            volt_log[esc_id].append(esc.get_voltage())
            curr_log[esc_id].append(esc.get_current())
            temp_log[esc_id].append(esc.get_temperature())
            rpm_ts[esc_id].append(t_now - t_start)
            if save_csv is not None:
                tgt = rpm_target_log[esc_id][-1] if rpm_target_log[esc_id] else float('nan')
                csv_rows.append({
                    'time_s':        round(t_now - t_start, 6),
                    'esc_id':        esc_id,
                    'target_rpm':    round(tgt, 1),
                    'actual_rpm':    round(esc.get_rpm(), 1),
                    'power_pct':     esc.get_power(),
                    'voltage_v':     round(esc.get_voltage(), 3),
                    'current_a':     round(esc.get_current(), 3),
                    'temperature_c': round(esc.get_temperature(), 2),
                })

            if (esc.get_temperature() > temperature_limit):
                print('ERROR: ESC %d overheated (%.2fC)!! Aborting.' % (esc_id,esc.get_temperature()) )
                timeout = 1.0
                #sys.exit(1)

            if board_current != None: #FIXME: should this be moved out of the esc loop?
                board_curr_log.append(board_current)
                board_curr_ts_log.append(t_now - t_start)

        if update_cntr % (np.floor(cmd_rate/50)) == 0:
            print('[%.3f] (%d) RPM: %.0f, PWR: %.0f, VBAT: %.2fV, TEMPERATURE: %.2fC, CURRENT: %.2fA' % (t_now - t_start,esc.get_id(), esc.get_rpm(), esc.get_power(), esc.get_voltage(), esc.get_temperature(), esc.get_current()))

    

print('Finished!')
print('[%f] TX=%d, RX=%d packets, RX CRC ERRORS=%d' % (t_now - t_start, esc_manager.tx_packet_count, esc_manager.rx_packet_count, esc_manager.protocol.crc_error_count))

if save_csv and csv_rows:
    import csv
    fieldnames = ['time_s', 'esc_id', 'target_rpm', 'actual_rpm', 'power_pct', 'voltage_v', 'current_a', 'temperature_c']
    with open(save_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print('INFO: Data saved to %s (%d rows)' % (save_csv, len(csv_rows)))

# plot results if possible
if enable_plot:
    try:
      import plotly.graph_objects as go
      from plotly.subplots import make_subplots
    except:
      print('WARNING: In order to plot the results, install the Python "plotly" module: pip3 install plotly --upgrade')
      sys.exit(0)


    try:
        from scipy.optimize import curve_fit

        def first_order_step(t, tau, rpm_init, rpm_final):
            return rpm_final + (rpm_init - rpm_final) * np.exp(-t / tau)

        print('\n--- Time Constant Estimation ---')
        for idx in range(4):
            if not rpm_target_log[idx]:
                continue
            tgt_t   = np.array(rpm_target_ts[idx])
            tgt_rpm = np.array(rpm_target_log[idx])
            act_t   = np.array(rpm_ts[idx])
            act_rpm = np.array(rpm_log[idx])

            diffs = np.abs(np.diff(tgt_rpm))
            thresh = 0.05 * (tgt_rpm.max() - tgt_rpm.min())
            step_idxs = np.where(diffs > thresh)[0]

            for i, si in enumerate(step_idxs):
                t_step    = tgt_t[si]
                rpm_before = tgt_rpm[si]
                rpm_after  = tgt_rpm[si + 1]
                t_end = tgt_t[step_idxs[i + 1]] if i + 1 < len(step_idxs) else tgt_t[-1]

                mask  = (act_t >= t_step) & (act_t <= t_end)
                t_win = act_t[mask] - t_step
                r_win = act_rpm[mask]
                if len(t_win) < 10:
                    continue
                try:
                    popt, _ = curve_fit(first_order_step, t_win, r_win,
                                        p0=[0.1, rpm_before, rpm_after],
                                        bounds=([0, -np.inf, -np.inf], [10, np.inf, np.inf]))
                    tau_ms = popt[0] * 1000
                    print('ESC #%d  Step %d  (%.0f -> %.0f RPM):  tau = %.1f ms' % (idx, i+1, rpm_before, rpm_after, tau_ms))
                    t_fit   = np.linspace(0, t_win[-1], 300)
                    rpm_fit = first_order_step(t_fit, *popt)
                    globals().setdefault('_fit_traces', []).append((idx, t_fit + t_step, rpm_fit, tau_ms, i+1))
                except Exception as e:
                    print('ESC #%d  Step %d: fitting failed (%s)' % (idx, i+1, e))
    except ImportError:
        print('WARNING: scipy not found — skipping tau fitting. Install with: pip3 install scipy')

    #xplot = np.arange(len(rpm_log[0]))
    fig = make_subplots(rows=5, cols=1, start_cell="top-left") #shared_xaxes=True
    for idx in range(4):
        fig.add_trace(go.Scatter(x=rpm_ts[idx], y=np.array(rpm_log[idx]), name='RPM #%d'%idx), row=1, col=1)
        if rpm_target_log[idx]:
            fig.add_trace(go.Scatter(x=rpm_target_ts[idx], y=np.array(rpm_target_log[idx]), name='RPM Target #%d'%idx, line=dict(dash='dash')), row=1, col=1)
        for (fidx, ft, fr, ftau, fstep) in globals().get('_fit_traces', []):
            if fidx == idx:
                fig.add_trace(go.Scatter(x=ft, y=fr, name='Fit #%d step%d tau=%.0fms'%(idx,fstep,ftau), line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=rpm_ts[idx], y=np.array(cmd_log[idx]), name='Command #%d'%idx), row=2, col=1)
        fig.add_trace(go.Scatter(x=rpm_ts[idx], y=np.array(volt_log[idx]), name='Voltage #%d (V)' % idx),row=3, col=1)
        fig.add_trace(go.Scatter(x=rpm_ts[idx], y=np.array(curr_log[idx]), name='Current #%d (A)' % idx),row=4, col=1)
        fig.add_trace(go.Scatter(x=rpm_ts[idx], y=np.array(temp_log[idx]), name='Temperature #%d (deg C)' % idx),row=5, col=1)

    if board_curr_log:
        fig.add_trace(go.Scatter(x=board_curr_ts_log, y=np.array(board_curr_log), name='Total Current (A)'),row=4, col=1)
    

    fig.update_layout(title_text='Voxl ESC Spin Test')
    fig.update_xaxes(title_text="Time (s)",row=1, col=1)
    fig.update_yaxes(title_text="Reported RPM",row=1, col=1)
    fig.update_xaxes(title_text="Time (s)",row=2, col=1)
    fig.update_yaxes(title_text="ESC Command (%)",row=2, col=1)
    fig.update_xaxes(title_text="Time (s)",row=3, col=1)
    fig.update_yaxes(title_text="Voltage (V)",row=3, col=1)
    fig.update_xaxes(title_text="Time (s)",row=4, col=1)
    fig.update_yaxes(title_text="Current (A)",row=4, col=1)
    fig.update_xaxes(title_text="Time (s)",row=5, col=1)
    fig.update_yaxes(title_text="Temperature (deg C)",row=5, col=1)
    fig.update_xaxes(matches='x')

    fig.write_html('voxl_esc_spin_step_results.html',include_plotlyjs='cdn')
    fig.show()
