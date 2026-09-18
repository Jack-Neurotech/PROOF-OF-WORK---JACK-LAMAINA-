
# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis
import mne
from pathlib import Path
import subprocess
import base64


# ============================================================
# SELECT EEG FILE
# ============================================================

downloads = Path.home() / "Downloads"

script = f'''
tell application "Finder"
    activate
    set selectedFile to choose file ¬
        with prompt "Select an EEG file" ¬
        default location POSIX file "{downloads}"
    return POSIX path of selectedFile
end tell
'''

result = subprocess.run(
    ["osascript", "-e", script],
    capture_output=True,
    text=True
)


# ============================================================
# CHECK FILE SELECTION
# ============================================================

if result.returncode != 0:
    raise SystemExit("No EEG file selected.")

eeg_file = result.stdout.strip()

print()
print("Selected EEG file:")
print(eeg_file)


# ============================================================
# LOAD EEG
# ============================================================

raw = mne.io.read_raw_eeglab(
    eeg_file,
    preload=True
)


# ============================================================
# BASIC EEG INFORMATION
# ============================================================

sampling_rate = raw.info["sfreq"]

channel_names = raw.ch_names

number_of_channels = len(channel_names)

number_of_samples = raw.n_times

duration = (
    number_of_samples
    / sampling_rate
)


# ============================================================
# EXTRACT EEG DATA
# ============================================================

eeg_data = raw.get_data()


# ============================================================
# SIGNAL QUALITY
# ============================================================

def assess_quality(signal_data):

    finite_fraction = np.mean(
        np.isfinite(signal_data)
    )

    mean = np.mean(
        signal_data
    )

    standard_deviation = np.std(
        signal_data
    )

    variance = np.var(
        signal_data
    )

    rms = np.sqrt(
        np.mean(
            signal_data ** 2
        )
    )

    peak_to_peak = (
        np.max(signal_data)
        -
        np.min(signal_data)
    )

    minimum = np.min(
        signal_data
    )

    maximum = np.max(
        signal_data
    )

    median = np.median(
        signal_data
    )

    mad = np.median(
        np.abs(
            signal_data - median
        )
    )

    skewness = skew(
        signal_data
    )

    kurtosis_value = kurtosis(
        signal_data
    )

    zero_crossings = np.sum(
        np.diff(
            np.sign(signal_data)
        ) != 0
    )

    zero_crossing_rate = (
        zero_crossings
        /
        len(signal_data)
    )

    if rms > 0:

        crest_factor = (
            np.max(
                np.abs(signal_data)
            )
            /
            rms
        )

    else:

        crest_factor = 0


    # ========================================================
    # HIGH-AMPLITUDE DETECTION
    # ========================================================

    high_threshold = (
        mean
        +
        5 * standard_deviation
    )

    low_threshold = (
        mean
        -
        5 * standard_deviation
    )

    high_amplitude_samples = np.sum(
        (
            signal_data
            >
            high_threshold
        )
        |
        (
            signal_data
            <
            low_threshold
        )
    )

    high_amplitude_fraction = (
        high_amplitude_samples
        /
        len(signal_data)
    )


    return {

        "finite_fraction":
            finite_fraction,

        "mean":
            mean,

        "standard_deviation":
            standard_deviation,

        "variance":
            variance,

        "rms":
            rms,

        "peak_to_peak":
            peak_to_peak,

        "min":
            minimum,

        "max":
            maximum,

        "median":
            median,

        "mad":
            mad,

        "skewness":
            skewness,

        "kurtosis":
            kurtosis_value,

        "zero_crossings":
            zero_crossings,

        "zero_crossing_rate":
            zero_crossing_rate,

        "crest_factor":
            crest_factor,

        "high_amplitude_fraction":
            high_amplitude_fraction
    }


# ============================================================
# BAND POWER
# ============================================================

def calculate_band_power(
    frequencies,
    power,
    low,
    high
):

    band = (
        (frequencies >= low)
        &
        (frequencies < high)
    )

    return np.sum(
        power[band]
    )


# ============================================================
# SPECTRAL ENTROPY
# ============================================================

def calculate_spectral_entropy(power):

    total_power = np.sum(
        power
    )

    if total_power <= 0:

        return 0

    probability = (
        power
        /
        total_power
    )

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability
        *
        np.log2(probability)
    )

    return entropy


# ============================================================
# DOMINANT FREQUENCY
# ============================================================

def calculate_dominant_frequency(
    frequencies,
    power
):

    if len(power) == 0:

        return 0

    index = np.argmax(
        power
    )

    return frequencies[index]


# ============================================================
# PEAK ALPHA FREQUENCY
# ============================================================

def calculate_peak_alpha_frequency(
    frequencies,
    power
):

    alpha_band = (
        (frequencies >= 8)
        &
        (frequencies < 13)
    )

    alpha_frequencies = (
        frequencies[alpha_band]
    )

    alpha_power = (
        power[alpha_band]
    )

    if len(alpha_power) == 0:

        return 0

    index = np.argmax(
        alpha_power
    )

    return alpha_frequencies[index]


# ============================================================
# SPECTRAL EDGE FREQUENCY
# ============================================================

def calculate_spectral_edge(
    frequencies,
    power,
    percentage=0.95
):

    cumulative_power = np.cumsum(
        power
    )

    total_power = (
        cumulative_power[-1]
    )

    if total_power <= 0:

        return 0

    target_power = (
        total_power
        *
        percentage
    )

    index = np.searchsorted(
        cumulative_power,
        target_power
    )

    if index >= len(frequencies):

        index = (
            len(frequencies)
            - 1
        )

    return frequencies[index]


# ============================================================
# BAND-POWER RATIOS
# ============================================================

def calculate_band_power_ratios(
    delta_power,
    theta_power,
    alpha_power,
    beta_power
):

    return {

        "theta_delta_ratio":
            theta_power / delta_power
            if delta_power > 0
            else 0,

        "alpha_theta_ratio":
            alpha_power / theta_power
            if theta_power > 0
            else 0,

        "beta_alpha_ratio":
            beta_power / alpha_power
            if alpha_power > 0
            else 0,

        "beta_theta_ratio":
            beta_power / theta_power
            if theta_power > 0
            else 0
    }


# ============================================================
# ANALYZE ALL CHANNELS
# ============================================================

results = []


for channel_index in range(
    number_of_channels
):

    channel_name = (
        channel_names[
            channel_index
        ]
    )

    channel_data = (
        eeg_data[
            channel_index
        ]
    )


    # ========================================================
    # QUALITY
    # ========================================================

    quality = assess_quality(
        channel_data
    )


    # ========================================================
    # FREQUENCY ANALYSIS
    # ========================================================

    frequencies, power = (
        signal.periodogram(
            channel_data,
            fs=sampling_rate
        )
    )


    # ========================================================
    # SPECTRAL METRICS
    # ========================================================

    total_spectral_power = (
        np.sum(power)
    )

    dominant_frequency = (
        calculate_dominant_frequency(
            frequencies,
            power
        )
    )

    peak_alpha_frequency = (
        calculate_peak_alpha_frequency(
            frequencies,
            power
        )
    )

    spectral_entropy = (
        calculate_spectral_entropy(
            power
        )
    )

    spectral_edge = (
        calculate_spectral_edge(
            frequencies,
            power
        )
    )


    # ========================================================
    # BAND POWER
    # ========================================================

    delta_power = (
        calculate_band_power(
            frequencies,
            power,
            0.5,
            4
        )
    )

    theta_power = (
        calculate_band_power(
            frequencies,
            power,
            4,
            8
        )
    )

    alpha_power = (
        calculate_band_power(
            frequencies,
            power,
            8,
            13
        )
    )

    beta_power = (
        calculate_band_power(
            frequencies,
            power,
            13,
            30
        )
    )


    # ========================================================
    # TOTAL BAND POWER
    # ========================================================

    total_band_power = (
        calculate_band_power(
            frequencies,
            power,
            0.5,
            30
        )
    )


    # ========================================================
    # RELATIVE BAND POWER
    # ========================================================

    if total_band_power > 0:

        delta_relative = (
            delta_power
            /
            total_band_power
        )

        theta_relative = (
            theta_power
            /
            total_band_power
        )

        alpha_relative = (
            alpha_power
            /
            total_band_power
        )

        beta_relative = (
            beta_power
            /
            total_band_power
        )

    else:

        delta_relative = 0
        theta_relative = 0
        alpha_relative = 0
        beta_relative = 0


    # ========================================================
    # BAND-POWER RATIOS
    # ========================================================

    ratios = (
        calculate_band_power_ratios(
            delta_power,
            theta_power,
            alpha_power,
            beta_power
        )
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "channel":
            channel_name,

        "finite_fraction":
            quality["finite_fraction"],

        "mean":
            quality["mean"],

        "standard_deviation":
            quality["standard_deviation"],

        "variance":
            quality["variance"],

        "rms":
            quality["rms"],

        "peak_to_peak":
            quality["peak_to_peak"],

        "min":
            quality["min"],

        "max":
            quality["max"],

        "median":
            quality["median"],

        "mad":
            quality["mad"],

        "skewness":
            quality["skewness"],

        "kurtosis":
            quality["kurtosis"],

        "zero_crossings":
            quality["zero_crossings"],

        "zero_crossing_rate":
            quality["zero_crossing_rate"],

        "crest_factor":
            quality["crest_factor"],

        "high_amplitude_fraction":
            quality[
                "high_amplitude_fraction"
            ],

        "total_spectral_power":
            total_spectral_power,

        "dominant_frequency":
            dominant_frequency,

        "peak_alpha_frequency":
            peak_alpha_frequency,

        "spectral_entropy":
            spectral_entropy,

        "spectral_edge":
            spectral_edge,

        "delta_power":
            delta_power,

        "theta_power":
            theta_power,

        "alpha_power":
            alpha_power,

        "beta_power":
            beta_power,

        "delta_relative":
            delta_relative,

        "theta_relative":
            theta_relative,

        "alpha_relative":
            alpha_relative,

        "beta_relative":
            beta_relative,

        "theta_delta_ratio":
            ratios[
                "theta_delta_ratio"
            ],

        "alpha_theta_ratio":
            ratios[
                "alpha_theta_ratio"
            ],

        "beta_alpha_ratio":
            ratios[
                "beta_alpha_ratio"
            ],

        "beta_theta_ratio":
            ratios[
                "beta_theta_ratio"
            ]
    })


# ============================================================
# CREATE EEG WAVEFORM GRAPH
# ============================================================

display_seconds = min(
    10,
    duration
)

display_samples = int(
    display_seconds
    *
    sampling_rate
)

display_times = (
    np.arange(
        display_samples
    )
    /
    sampling_rate
)


waveform_figure = plt.figure(
    figsize=(12, 5)
)

plt.plot(
    display_times,
    eeg_data[
        0,
        :display_samples
    ]
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    f"EEG Signal — {channel_names[0]}"
)

plt.tight_layout()


# ============================================================
# CREATE POWER SPECTRUM GRAPH
# ============================================================

first_channel = eeg_data[0]

frequencies, power = (
    signal.periodogram(
        first_channel,
        fs=sampling_rate
    )
)


spectrum_figure = plt.figure(
    figsize=(12, 5)
)

plt.plot(
    frequencies,
    power
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power"
)

plt.title(
    f"Power Spectrum — {channel_names[0]}"
)

plt.xlim(
    0,
    50
)

plt.tight_layout()


# ============================================================
# CREATE RELATIVE BAND POWER GRAPH
# ============================================================

channel_labels = [
    result["channel"]
    for result in results
]

delta_values = [
    result["delta_relative"] * 100
    for result in results
]

theta_values = [
    result["theta_relative"] * 100
    for result in results
]

alpha_values = [
    result["alpha_relative"] * 100
    for result in results
]

beta_values = [
    result["beta_relative"] * 100
    for result in results
]


band_figure = plt.figure(
    figsize=(12, 6)
)

x = np.arange(
    number_of_channels
)

width = 0.2


plt.bar(
    x - 1.5 * width,
    delta_values,
    width,
    label="Delta"
)

plt.bar(
    x - 0.5 * width,
    theta_values,
    width,
    label="Theta"
)

plt.bar(
    x + 0.5 * width,
    alpha_values,
    width,
    label="Alpha"
)

plt.bar(
    x + 1.5 * width,
    beta_values,
    width,
    label="Beta"
)


plt.xticks(
    x,
    channel_labels,
    rotation=90
)

plt.xlabel(
    "EEG Channel"
)

plt.ylabel(
    "Relative Power (%)"
)

plt.title(
    "Relative EEG Band Power by Channel"
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE GRAPHS TO TEMPORARY FILES
# ============================================================

waveform_path = (
    "/tmp/eeg_waveform.png"
)

spectrum_path = (
    "/tmp/eeg_spectrum.png"
)

band_path = (
    "/tmp/eeg_band_power.png"
)


waveform_figure.savefig(
    waveform_path,
    dpi=150
)

spectrum_figure.savefig(
    spectrum_path,
    dpi=150
)

band_figure.savefig(
    band_path,
    dpi=150
)


plt.close(
    waveform_figure
)

plt.close(
    spectrum_figure
)

plt.close(
    band_figure
)


# ============================================================
# CREATE REPORT HTML
# ============================================================

report_html = f'''
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>EEG Analysis Report</title>

<style>

body {{
    font-family: -apple-system, BlinkMacSystemFont,
                 "Helvetica Neue", Arial, sans-serif;
    margin: 30px;
    background: #f5f5f7;
    color: #1d1d1f;
}}

h1 {{
    font-size: 30px;
}}

h2 {{
    margin-top: 30px;
    font-size: 22px;
}}

.info {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}}

.table-container {{
    background: white;
    padding: 15px;
    border-radius: 12px;
    overflow-x: auto;
}}

table {{
    border-collapse: collapse;
    width: max-content;
    min-width: 100%;
}}

th {{
    background: #e8e8ed;
    padding: 10px;
    border: 1px solid #d2d2d7;
    white-space: nowrap;
}}

td {{
    padding: 8px;
    border: 1px solid #d2d2d7;
    text-align: center;
    white-space: nowrap;
}}

img {{
    width: 100%;
    max-width: 1100px;
    margin-top: 10px;
    border-radius: 10px;
}}

.section {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}}

</style>

</head>

<body>

<h1>EEG Analysis Report</h1>


<div class="info">

<h2>Dataset Information</h2>

<p>
<strong>File:</strong>
{Path(eeg_file).name}
</p>

<p>
<strong>Sampling Rate:</strong>
{sampling_rate:.2f} Hz
</p>

<p>
<strong>Channels:</strong>
{number_of_channels}
</p>

<p>
<strong>Samples:</strong>
{number_of_samples}
</p>

<p>
<strong>Duration:</strong>
{duration:.2f} seconds
</p>

</div>


<div class="section">

<h2>Channel Metrics</h2>

<div class="table-container">

<table>

<tr>

<th>Channel</th>
<th>Mean</th>
<th>Std</th>
<th>Variance</th>
<th>RMS</th>
<th>Peak-to-Peak</th>
<th>Min</th>
<th>Max</th>
<th>Median</th>
<th>MAD</th>
<th>Skewness</th>
<th>Kurtosis</th>
<th>Zero Crossings</th>
<th>Crest Factor</th>
<th>High Amplitude</th>
<th>Total Spectral Power</th>
<th>Dominant Frequency</th>
<th>Peak Alpha Frequency</th>
<th>Spectral Entropy</th>
<th>Spectral Edge</th>
<th>Delta Power</th>
<th>Theta Power</th>
<th>Alpha Power</th>
<th>Beta Power</th>
<th>Delta %</th>
<th>Theta %</th>
<th>Alpha %</th>
<th>Beta %</th>
<th>Theta/Delta</th>
<th>Alpha/Theta</th>
<th>Beta/Alpha</th>
<th>Beta/Theta</th>

</tr>
'''


# ============================================================
# ADD CHANNEL RESULTS TO REPORT
# ============================================================

for result in results:

    report_html += f'''

<tr>

<td>{result["channel"]}</td>

<td>{result["mean"]:.4g}</td>

<td>{result["standard_deviation"]:.4g}</td>

<td>{result["variance"]:.4g}</td>

<td>{result["rms"]:.4g}</td>

<td>{result["peak_to_peak"]:.4g}</td>

<td>{result["min"]:.4g}</td>

<td>{result["max"]:.4g}</td>

<td>{result["median"]:.4g}</td>

<td>{result["mad"]:.4g}</td>

<td>{result["skewness"]:.4g}</td>

<td>{result["kurtosis"]:.4g}</td>

<td>{result["zero_crossings"]}</td>

<td>{result["crest_factor"]:.4g}</td>

<td>{result["high_amplitude_fraction"] * 100:.4f}%</td>

<td>{result["total_spectral_power"]:.4g}</td>

<td>{result["dominant_frequency"]:.3f} Hz</td>

<td>{result["peak_alpha_frequency"]:.3f} Hz</td>

<td>{result["spectral_entropy"]:.4g}</td>

<td>{result["spectral_edge"]:.3f} Hz</td>

<td>{result["delta_power"]:.4g}</td>

<td>{result["theta_power"]:.4g}</td>

<td>{result["alpha_power"]:.4g}</td>

<td>{result["beta_power"]:.4g}</td>

<td>{result["delta_relative"] * 100:.2f}%</td>

<td>{result["theta_relative"] * 100:.2f}%</td>

<td>{result["alpha_relative"] * 100:.2f}%</td>

<td>{result["beta_relative"] * 100:.2f}%</td>

<td>{result["theta_delta_ratio"]:.4g}</td>

<td>{result["alpha_theta_ratio"]:.4g}</td>

<td>{result["beta_alpha_ratio"]:.4g}</td>

<td>{result["beta_theta_ratio"]:.4g}</td>

</tr>

'''


# ============================================================
# ADD GRAPHS TO REPORT
# ============================================================

report_html += f'''

</table>

</div>

</div>


<div class="section">

<h2>EEG Waveform</h2>

<p>
First {display_seconds:.2f} seconds of
{channel_names[0]}.
</p>

<img src="file://{waveform_path}">

</div>


<div class="section">

<h2>Power Spectrum</h2>

<p>
Frequency-domain representation of
{channel_names[0]}.
</p>

<img src="file://{spectrum_path}">

</div>


<div class="section">

<h2>Relative Band Power</h2>

<p>
Relative Delta, Theta, Alpha, and Beta
power for every EEG channel.
</p>

<img src="file://{band_path}">

</div>


</body>

</html>
'''


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    "/tmp/eeg_analysis_report.html"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        report_html
    )


# ============================================================
# OPEN REPORT IN MACOS
# ============================================================

subprocess.run(
    [
        "open",
        report_path
    ]
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 60)
print("EEG ANALYSIS COMPLETE")
print("=" * 60)

print()
print("Channels analyzed:",
      number_of_channels)

print("Report created:")
print(report_path)

print()
print("The consolidated EEG report")
print("has been opened in macOS.")



        # ============================================================
# EEG MEASUREMENT AND TEST EXPLANATIONS
# ============================================================
#
# This section explains what each measurement/test means,
# the mathematical idea behind it, and how the EEG pipeline
# obtains the information from the recorded EEG signal.
#
# The basic flow of the EEG pipeline is:
#
# RAW EEG
#   ↓
# Voltage samples over time
#   ↓
# Time-domain measurements
#   ↓
# Frequency analysis
#   ↓
# Power spectrum
#   ↓
# Frequency bands
#   ↓
# Band power and ratios
#
# EEG amplitude is measured in microvolts (µV).
# Frequency is measured in Hertz (Hz).
# Power is commonly represented as µV².
# Power spectral density is represented as µV²/Hz.
#
# ============================================================
# 1. MEAN
# ============================================================
#
# Equation:
#
#     μ = (1/N) Σ xᵢ
#
# What the test is:
#
# The mean is the average voltage of the EEG signal.
# It tells us where the signal is centered around zero.
#
# In EEG, the mean can help describe the baseline level
# of the recorded signal.
#
# How the EEG pipeline acquired the information:
#
# The pipeline first loads the EEG recording and obtains
# the voltage samples for a channel.
#
# The mean is then calculated from every voltage sample
# in that channel.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 2. STANDARD DEVIATION
# ============================================================
#
# Equation:
#
#     σ = √[(1/N) Σ(xᵢ - μ)²]
#
# What the test is:
#
# Standard deviation measures how much the EEG samples
# vary around their mean.
#
# A small standard deviation means the signal stays close
# to its average value.
#
# A large standard deviation means the signal has greater
# amplitude variation.
#
# How the EEG pipeline acquired the information:
#
# The pipeline takes all voltage samples from the EEG channel,
# calculates their mean, measures how far each sample is from
# that mean, and calculates the standard deviation.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 3. VARIANCE
# ============================================================
#
# Equation:
#
#     σ² = (1/N) Σ(xᵢ - μ)²
#
# What the test is:
#
# Variance measures the amount of variability in the EEG
# signal.
#
# Variance is the square of standard deviation.
#
# A larger variance means the EEG signal has greater
# amplitude variability.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the difference between each EEG
# sample and the mean, squares those differences, and
# calculates their average.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 4. RMS — ROOT MEAN SQUARE
# ============================================================
#
# Equation:
#
#     RMS = √[(1/N) Σxᵢ²]
#
# What the test is:
#
# RMS measures the overall magnitude of the EEG signal.
#
# Unlike the mean, positive and negative voltages do not
# cancel each other out because the samples are squared.
#
# RMS is therefore useful for describing the overall
# strength of the signal.
#
# How the EEG pipeline acquired the information:
#
# The pipeline squares every EEG voltage sample, calculates
# the mean of those squared values, and then takes the
# square root.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 5. PEAK-TO-PEAK AMPLITUDE
# ============================================================
#
# Equation:
#
#     Vpp = Vmax - Vmin
#
# What the test is:
#
# Peak-to-peak amplitude measures the complete voltage range
# of the EEG signal.
#
# It is the difference between the highest and lowest
# recorded voltage.
#
# How the EEG pipeline acquired the information:
#
# The pipeline finds the maximum voltage and minimum voltage
# in the EEG channel and subtracts the minimum from the maximum.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 6. MINIMUM AND MAXIMUM
# ============================================================
#
# Equations:
#
#     Vmin = min(xᵢ)
#
#     Vmax = max(xᵢ)
#
# What the test is:
#
# These measurements identify the lowest and highest voltage
# values recorded in the EEG signal.
#
# They describe the extreme amplitudes present in the recording.
#
# How the EEG pipeline acquired the information:
#
# The pipeline searches all EEG samples and identifies the
# smallest and largest voltage values.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 7. MEDIAN
# ============================================================
#
# What the test is:
#
# The median is the middle EEG voltage after all samples
# are sorted from smallest to largest.
#
# The median is less affected by extreme values than the mean.
#
# How the EEG pipeline acquired the information:
#
# The pipeline sorts the EEG voltage samples and selects
# the middle value.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 8. MAD — MEDIAN ABSOLUTE DEVIATION
# ============================================================
#
# Equation:
#
#     MAD = median(|xᵢ - median(x)|)
#
# What the test is:
#
# MAD is a robust measurement of how much the EEG samples
# vary around the median.
#
# It is less sensitive to extreme values than standard deviation.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the median EEG voltage, calculates
# each sample's distance from that median, and then finds
# the median of those distances.
#
# Unit:
#
#     µV
#
#
# ============================================================
# 9. SKEWNESS
# ============================================================
#
# Conceptual equation:
#
#     skewness = E[(X - μ)³] / σ³
#
# What the test is:
#
# Skewness measures whether the distribution of EEG amplitudes
# is symmetric or has a longer tail on one side.
#
# Positive skewness means the distribution has a longer
# positive-side tail.
#
# Negative skewness means the distribution has a longer
# negative-side tail.
#
# How the EEG pipeline acquired the information:
#
# The pipeline examines the distribution of all EEG voltage
# samples and calculates its skewness.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 10. KURTOSIS
# ============================================================
#
# Conceptual equation:
#
#     kurtosis = E[(X - μ)⁴] / σ⁴
#
# What the test is:
#
# Kurtosis describes the tails of the EEG amplitude
# distribution and therefore helps characterize unusually
# extreme values.
#
# High kurtosis can indicate heavier tails and more extreme
# amplitude values.
#
# How the EEG pipeline acquired the information:
#
# The pipeline examines the fourth-power deviations of the
# EEG samples from their mean.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 11. ZERO CROSSINGS
# ============================================================
#
# Condition:
#
#     xᵢ × xᵢ₊₁ < 0
#
# What the test is:
#
# A zero crossing occurs when the EEG signal changes from
# positive to negative or from negative to positive.
#
# It measures how often the signal crosses zero voltage.
#
# How the EEG pipeline acquired the information:
#
# The pipeline examines the sign of consecutive EEG samples
# and counts each time the sign changes.
#
# Unit:
#
#     Count
#
#
# ============================================================
# 12. ZERO-CROSSING RATE
# ============================================================
#
# Equation:
#
#     ZCR = number of zero crossings / N
#
# What the test is:
#
# Zero-crossing rate normalizes the number of zero crossings
# by the number of EEG samples.
#
# It describes how frequently the signal changes polarity
# relative to the length of the signal.
#
# How the EEG pipeline acquired the information:
#
# The pipeline counts zero crossings and divides that count
# by the total number of samples in the EEG signal.
#
# Unit:
#
#     Unitless fraction
#
#
# ============================================================
# 13. CREST FACTOR
# ============================================================
#
# Equation:
#
#     CF = |Vpeak| / RMS
#
# What the test is:
#
# Crest factor compares the largest absolute EEG amplitude
# with the overall RMS magnitude of the signal.
#
# A high crest factor means the signal contains relatively
# large peaks compared with its typical magnitude.
#
# How the EEG pipeline acquired the information:
#
# The pipeline finds the largest absolute EEG voltage and
# divides it by the RMS of the signal.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 14. HIGH-AMPLITUDE FRACTION
# ============================================================
#
# Thresholds:
#
#     V > μ + 5σ
#
#     OR
#
#     V < μ - 5σ
#
# What the test is:
#
# This test identifies EEG samples that are more than
# five standard deviations away from the mean.
#
# These unusually large values can indicate artifacts or
# unusual events in the recording.
#
# This measurement does not automatically prove that a sample
# is an artifact. It identifies unusually high amplitudes.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the mean and standard deviation,
# establishes the ±5 standard-deviation thresholds, counts
# samples outside those thresholds, and divides by the
# total number of samples.
#
# Unit:
#
#     Fraction or percentage
#
#
# ============================================================
# 15. TOTAL SPECTRAL POWER
# ============================================================
#
# Equation:
#
#     Ptotal = Σ P(f)
#
# What the test is:
#
# Total spectral power represents the overall amount of
# signal power across the analyzed frequency range.
#
# It changes the question from:
#
#     "How large is the voltage?"
#
# to:
#
#     "How much signal energy/power exists across frequency?"
#
# How the EEG pipeline acquired the information:
#
# The pipeline first converts the EEG signal from the
# time domain into the frequency domain using a periodogram.
#
# It then sums the power values across frequencies.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 16. DOMINANT FREQUENCY
# ============================================================
#
# Equation:
#
#     fdominant = argmax P(f)
#
# What the test is:
#
# Dominant frequency is the frequency containing the greatest
# amount of spectral power.
#
# It identifies the strongest frequency component in the EEG.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power spectrum, finds the
# frequency where power is greatest, and reports that
# frequency.
#
# Unit:
#
#     Hz
#
#
# ============================================================
# 17. PEAK ALPHA FREQUENCY
# ============================================================
#
# Equation:
#
#     fαpeak = argmax P(f), 8 ≤ f < 13 Hz
#
# What the test is:
#
# Peak alpha frequency identifies the strongest frequency
# specifically inside the alpha band.
#
# The alpha range used by this pipeline is approximately
# 8–13 Hz.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power spectrum, selects only
# frequencies inside the alpha range, and finds the frequency
# with the greatest power.
#
# Unit:
#
#     Hz
#
#
# ============================================================
# 18. SPECTRAL ENTROPY
# ============================================================
#
# Equations:
#
#     pᵢ = Pᵢ / ΣPᵢ
#
#     H = -Σ pᵢ log₂(pᵢ)
#
# What the test is:
#
# Spectral entropy measures how distributed the EEG power
# is across frequencies.
#
# If most power is concentrated in a small number of
# frequencies, entropy is lower.
#
# If power is distributed across many frequencies,
# entropy is higher.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power spectrum, converts the
# power values into normalized proportions, and calculates
# entropy from those proportions.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 19. SPECTRAL EDGE FREQUENCY
# ============================================================
#
# Equation:
#
#     Σ P(f ≤ fedge) ≥ 0.95 Ptotal
#
# What the test is:
#
# Spectral edge frequency identifies the frequency below
# which a specified percentage of total spectral power lies.
#
# In this pipeline, the edge is based on 95% of the power.
#
# In simple terms:
#
#     "At what frequency have we accumulated 95% of the
#      signal's spectral power?"
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power spectrum, creates a
# cumulative sum of power across frequency, finds the point
# where cumulative power reaches 95% of total power, and
# reports that frequency.
#
# Unit:
#
#     Hz
#
#
# ============================================================
# 20. DELTA POWER
# ============================================================
#
# Frequency range:
#
#     0.5–4 Hz
#
# Equation:
#
#     Pδ = ∫₀.₅⁴ P(f) df
#
# What the test is:
#
# Delta power measures how much EEG spectral power exists
# in the delta frequency range.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power spectrum, selects
# frequencies between 0.5 and 4 Hz, and calculates the
# power contained in that frequency range.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 21. THETA POWER
# ============================================================
#
# Frequency range:
#
#     4–8 Hz
#
# Equation:
#
#     Pθ = ∫₄⁸ P(f) df
#
# What the test is:
#
# Theta power measures the spectral power contained in
# the theta frequency range.
#
# How the EEG pipeline acquired the information:
#
# The pipeline selects the 4–8 Hz portion of the power
# spectrum and calculates the power in that band.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 22. ALPHA POWER
# ============================================================
#
# Frequency range:
#
#     8–13 Hz
#
# Equation:
#
#     Pα = ∫₈¹³ P(f) df
#
# What the test is:
#
# Alpha power measures the spectral power contained in
# the alpha frequency range.
#
# How the EEG pipeline acquired the information:
#
# The pipeline selects frequencies between 8 and 13 Hz
# from the power spectrum and calculates the power in
# that band.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 23. BETA POWER
# ============================================================
#
# Frequency range:
#
#     13–30 Hz
#
# Equation:
#
#     Pβ = ∫₁₃³⁰ P(f) df
#
# What the test is:
#
# Beta power measures the spectral power contained in
# the beta frequency range.
#
# How the EEG pipeline acquired the information:
#
# The pipeline selects frequencies between 13 and 30 Hz
# from the power spectrum and calculates the power in
# that band.
#
# Unit:
#
#     µV²
#
#
# ============================================================
# 24. RELATIVE BAND POWER
# ============================================================
#
# Example equation for alpha:
#
#     Relative Alpha Power =
#
#         Pα / P0.5–30
#
# What the test is:
#
# Relative band power measures how much of the total
# analyzed EEG power is contained within a particular
# frequency band.
#
# For example, relative alpha power asks:
#
#     "What fraction of the total 0.5–30 Hz power
#      comes from the alpha band?"
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates the power of a frequency band,
# calculates the total power across the reference frequency
# range, and divides the band power by the total power.
#
# The result can be multiplied by 100 to express it
# as a percentage.
#
# Unit:
#
#     Unitless fraction or %
#
#
# ============================================================
# 25. THETA / DELTA RATIO
# ============================================================
#
# Equation:
#
#     Theta/Delta = Pθ / Pδ
#
# What the test is:
#
# This ratio compares theta power with delta power.
#
# It does not measure an absolute amount of power.
# Instead, it describes the relationship between two
# frequency bands.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates theta power and delta power
# independently from the power spectrum and then divides
# theta power by delta power.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 26. ALPHA / THETA RATIO
# ============================================================
#
# Equation:
#
#     Alpha/Theta = Pα / Pθ
#
# What the test is:
#
# This ratio compares alpha power with theta power.
#
# It describes the relative amount of alpha activity
# compared with theta activity.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates alpha and theta power from
# the power spectrum and divides alpha power by theta power.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 27. BETA / ALPHA RATIO
# ============================================================
#
# Equation:
#
#     Beta/Alpha = Pβ / Pα
#
# What the test is:
#
# This ratio compares beta power with alpha power.
#
# It describes the relative amount of beta activity
# compared with alpha activity.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates beta and alpha power and divides
# beta power by alpha power.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# 28. BETA / THETA RATIO
# ============================================================
#
# Equation:
#
#     Beta/Theta = Pβ / Pθ
#
# What the test is:
#
# This ratio compares beta power with theta power.
#
# It describes the relative amount of beta activity
# compared with theta activity.
#
# How the EEG pipeline acquired the information:
#
# The pipeline calculates beta and theta power and divides
# beta power by theta power.
#
# Unit:
#
#     Unitless
#
#
# ============================================================
# THE BIG PICTURE
# ============================================================
#
# The EEG pipeline is essentially asking several different
# questions about the same recorded neural signal.
#
# TIME-DOMAIN TESTS ask:
#
#     "What does the voltage signal look like over time?"
#
# Examples:
#
#     Mean
#     Standard deviation
#     Variance
#     RMS
#     Peak-to-peak
#     Minimum
#     Maximum
#     Median
#     MAD
#     Skewness
#     Kurtosis
#     Zero crossings
#     Crest factor
#     High-amplitude fraction
#
#
# FREQUENCY-DOMAIN TESTS ask:
#
#     "How is the EEG signal's power distributed across
#      different frequencies?"
#
# Examples:
#
#     Total spectral power
#     Dominant frequency
#     Peak alpha frequency
#     Spectral entropy
#     Spectral edge frequency
#
#
# BAND-POWER TESTS ask:
#
#     "How much power exists inside specific frequency bands?"
#
# Examples:
#
#     Delta power
#     Theta power
#     Alpha power
#     Beta power
#
#
# RATIO TESTS ask:
#
#     "How does the power of one frequency band compare
#      with another?"
#
# Examples:
#
#     Theta / Delta
#     Alpha / Theta
#     Beta / Alpha
#     Beta / Theta
#
#
# ============================================================
# CORE EEG PIPELINE
# ============================================================
#
# The most important conceptual transformation is:
#
#     EEG voltage
#          ↓
#     voltage samples over time
#          ↓
#     time-domain measurements
#          ↓
#     frequency transformation
#          ↓
#     power spectrum
#          ↓
#     frequency-band power
#          ↓
#     ratios and higher-level features
#
# In other words, the pipeline takes a physical electrical
# signal recorded from the scalp and progressively converts
# it into mathematical measurements that describe:
#
#     1. Signal amplitude
#     2. Signal variability
#     3. Signal shape
#     4. Signal frequency content
#     5. Distribution of spectral power
#     6. Power in specific frequency bands
#     7. Relationships between frequency bands
#
# This is the foundation of computational EEG analysis.
# ============================================================
