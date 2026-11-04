# ESC Calibration — 4S Setup

## Motor / Hardware
- Battery: 4S LiPo (~15.2V nominal, ~16.4V full)
- `num_cycles_per_rev`: 6 (pole pairs)
- Min RPM: ~4800 (at 10% power, full charge)
- Max RPM: ~24900 (at 94% power)

## Calibrated Parameters (`default_parameter_file_for_esc_calibration_4s.xml`)
Feed-forward curve (voltage vs RPM quadratic fit):
- `pwm_vs_rpm_curve_a0` = 933.654
- `pwm_vs_rpm_curve_a1` = 0.14848
- `pwm_vs_rpm_curve_a2` = 1.538e-05

RPM controller gains (PI):
- `kp` = 12, `ki` = 7
- `max_kpe` = 200, `max_kie` = 200

## Workflow

### 1. Run calibration (feed-forward curve)
```
./voxl-esc-calibrate.py --id 0 --pwm-min 10 --pwm-max 95
```
Update `pwm_vs_rpm_curve_a0/a1/a2`, `min_rpm`, `max_rpm` in the XML, then upload:
```
./voxl-esc-upload-params.py --params-file ../default_parameter_file_for_esc_calibration_4s.xml
```

### 2. Measure min/max RPM
```
./voxl-esc-spin.py --id 0 --power 10                              # min RPM
./voxl-esc-spin.py --id 0 --power 100 --ramp-time 3.0 --timeout 3.5  # max RPM
```

### 3. Collect step-response data for motor model identification
```
./voxl-esc-spin-step.py --id 0 --rpm 4800 \
  --step-amplitudes "2000,5000,10000,20100" \
  --step-frequency 0.25 --step-delay 2.0 \
  --timeout 25 --cmd-rate 250 \
  --save-csv motor_step_response.csv
```

### 4. Fit asymmetric second-order motor model
```
python3 analyze_motor_step_response.py --csv motor_step_response.csv --esc-id 0
```
Outputs `natural_freq_up`, `natural_freq_down`, `damping_ratio_up`, `damping_ratio_down` ready to paste into `rotor_dynamics.py`.

## Motor Model Notes
- Spin-down follows a standard first-order model closely.
- Spin-up is **torque-saturated** for large steps: RPM increases linearly at `max_rate_up`
  before transitioning to exponential approach near the target.
- A rate-limited first-order model captures both phases.
- Measured time constants (approximate, at 4S full charge):
  - `motor_time_constant_up`  ≈ 0.100 s
  - `motor_time_constant_down` ≈ 0.064 s
  - `max_rps_rate_up`   ≈ 1860 RPS/s  (≈ 111600 RPM/s)
  - `max_rps_rate_down` ≈ measured from step test
