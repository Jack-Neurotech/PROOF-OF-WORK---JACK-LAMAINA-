# ============================================================
# IONM SIGNAL INTELLIGENCE PIPELINE
# SIGNAL QUALITY, EVENT ANALYSIS, CHANGE DETECTION,
# STATISTICS, AND EVIDENCE CLASSIFICATION
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

# What the code does:
# Imports the scientific Python packages required by the
# pipeline.
#
# Why it is here:
# MNE handles electrophysiological recordings.
# NumPy handles numerical calculations.
# SciPy provides signal-processing and statistical functions.
# Matplotlib creates the graphical report.
# JSON creates the structured evidence report.
# Subprocess provides the macOS Finder file selector.
# Path provides safe filesystem path handling.

import json
import math
import subprocess
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kruskal


# ============================================================
# CONFIGURATION
# ============================================================

# What the code does:
# Defines the reproducible analysis parameters used throughout
# the pipeline.
#
# Why it is here:
# Keeping these values in one location makes the analysis
# transparent and reproducible.

HIGH_PASS = 1.0
LOW_PASS = 45.0

EVENT_TMIN = -0.2
EVENT_TMAX = 0.8

BASELINE_START = -0.2
BASELINE_END = 0.0

RESPONSE_START = 0.0
RESPONSE_END = 0.8

TRANSIENT_Z = 6.0
STIMULATION_Z = 8.0

LINE_NOISE_FREQUENCY = 60.0
LINE_NOISE_TOLERANCE = 1.0
LINE_NOISE_PROMINENCE = 10.0

FLATLINE_THRESHOLD = 1e-12
SATURATION_FRACTION_THRESHOLD = 0.01

OUTLIER_Z_THRESHOLD = 3.5
LOW_RMS_RATIO = 0.20
HIGH_RMS_RATIO = 5.0

CHANGE_Z_THRESHOLD = 2.0
CHANGE_ABSOLUTE_THRESHOLD = 1.0
MINIMUM_CONSECUTIVE = 3

AMPLITUDE_CHANGE_THRESHOLD = 0.20
LATENCY_CHANGE_THRESHOLD = 0.10

STATISTICAL_ALPHA = 0.05


# ============================================================
# FILE SELECTION
# ============================================================

# What the code does:
# Opens the macOS Finder and allows the user to select an EEG
# recording.
#
# Why it is here:
# This avoids tkinter and provides a simple graphical file
# selection method on macOS.

def select_eeg_file():

    script = '''
    tell application "Finder"
        activate
        set selectedFile to choose file with prompt "Select an EEG / IONM recording"
        return POSIX path of selectedFile
    end tell
    '''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError("File selection was cancelled.")

    path = result.stdout.strip()

    if not path:
        raise RuntimeError("No EEG file was selected.")

    return Path(path)


# ============================================================
# RECORDING INGESTION
# ============================================================

# Mathematical representation:
#
# EEG = MNE.read_raw(...)
#
# What the code does:
# Loads the selected electrophysiological recording.
#
# Why it is here:
# The pipeline must convert the source recording into a
# structured signal representation before analysis.

def load_recording(path):

    suffix = path.suffix.lower()

    if suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True)

    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True)

    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True)

    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True)

    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True)

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            "Supported formats: .set, .edf, .bdf, .vhdr, .fif"
        )

    return raw


# ============================================================
# RECORDING METADATA
# ============================================================

# Mathematical representation:
#
# T = N / f_s
#
# What the code does:
# Extracts channels, samples, sampling frequency, duration,
# channel types, and annotation information.
#
# Why it is here:
# These values establish the structure of the source recording.

def recording_metadata(raw):

    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    duration = n_samples / sfreq

    channel_types = {}

    for name, kind in zip(raw.ch_names, raw.get_channel_types()):
        channel_types[name] = kind

    return {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sampling_frequency_hz": sfreq,
        "duration_seconds": duration,
        "channel_names": raw.ch_names,
        "channel_types": channel_types,
        "annotation_count": len(raw.annotations),
    }


# ============================================================
# CHANNEL ROLE ASSIGNMENT
# ============================================================

# Mathematical representation:
#
# channel -> EEG / EOG / ECG / OTHER
#
# What the code does:
# Assigns each recording channel to an analysis role.
#
# Why it is here:
# EEG signal analysis should operate on EEG channels rather than
# mixing EEG with physiological or auxiliary channels.

def assign_channel_roles(raw):

    roles = {}

    for name in raw.ch_names:

        upper = name.upper()

        if upper in {"HEOG", "VEOG", "EOG"} or "EOG" in upper:
            roles[name] = "EOG"

        elif upper in {"ECG", "EKG"} or "ECG" in upper or "EKG" in upper:
            roles[name] = "ECG"

        else:
            roles[name] = "EEG"

    return roles


# ============================================================
# EEG CHANNEL EXTRACTION
# ============================================================

# What the code does:
# Extracts only channels assigned the EEG role.
#
# Why it is here:
# The downstream signal calculations are designed for EEG
# channels.

def get_eeg_channels(raw, roles):

    return [
        name
        for name, role in roles.items()
        if role == "EEG"
    ]


# ============================================================
# EEG PREPROCESSING
# ============================================================

# Mathematical representation:
#
# X_filtered = BandPass(X, 1 Hz, 45 Hz)
#
# What the code does:
# Creates a copy of the recording and applies the configured
# EEG band-pass filter.
#
# Why it is here:
# Filtering restricts the signal to the implemented analysis
# frequency range without modifying the original Raw object.

def preprocess_eeg(raw):

    filtered = raw.copy()

    filtered.filter(
        l_freq=HIGH_PASS,
        h_freq=LOW_PASS,
        picks="eeg",
        verbose=False
    )

    return filtered


# ============================================================
# BASIC SIGNAL METRICS
# ============================================================

# Mathematical representation:
#
# RMS = sqrt(mean(x^2))
#
# SD = standard deviation
#
# PTP = max(x) - min(x)
#
# What the code does:
# Calculates the fundamental signal measurements used by the
# quality assessment.

def rms(x):

    return float(np.sqrt(np.mean(np.square(x))))


def standard_deviation(x):

    return float(np.std(x))


def peak_to_peak(x):

    return float(np.ptp(x))


def signal_minimum(x):

    return float(np.min(x))


def signal_maximum(x):

    return float(np.max(x))


# ============================================================
# FLATLINE DETECTION
# ============================================================

# Mathematical representation:
#
# flatline = PTP <= 1e-12
#
# What the code does:
# Determines whether a channel has essentially no variation.

def detect_flatline(x):

    return peak_to_peak(x) <= FLATLINE_THRESHOLD


# ============================================================
# SATURATION DETECTION
# ============================================================

# Mathematical representation:
#
# saturated_fraction =
#
#     saturated_samples / total_samples
#
# Detection:
#
#     saturated_fraction >= 0.01
#
# What the code does:
# Detects excessive clipping at the signal limits.

def detect_saturation(x):

    lower = np.min(x)
    upper = np.max(x)

    saturated = np.logical_or(
        x <= lower,
        x >= upper
    )

    fraction = float(np.mean(saturated))

    return {
        "saturated_fraction": fraction,
        "saturated": fraction >= SATURATION_FRACTION_THRESHOLD
    }


# ============================================================
# SIGNAL-TO-NOISE RATIO
# ============================================================

# Mathematical representation:
#
# SNR = 10 log10(signal_power / noise_power)
#
# What the code does:
# Calculates signal-to-noise ratio using mean squared signal
# magnitude.
#
# Why it is here:
# SNR provides a quantitative measure of signal quality.

def calculate_snr(x):

    signal_power = float(np.mean(np.square(x)))

    if len(x) < 3:
        return None

    difference = np.diff(x)

    noise_power = float(np.mean(np.square(difference)))

    if noise_power == 0:
        return float("inf")

    if signal_power == 0:
        return float("-inf")

    return float(
        10.0 * np.log10(signal_power / noise_power)
    )


# ============================================================
# CHANNEL QUALITY CLASSIFICATION
# ============================================================

# Mathematical representation:
#
# bad:
#     flatline OR saturated OR SNR < 0
#
# poor:
#     SNR < 5
#
# fair:
#     SNR < 10
#
# good:
#     otherwise
#
# What the code does:
# Converts quantitative measurements into a channel-quality
# category.

def classify_quality(flatline, saturated, snr):

    if flatline or saturated:
        return "bad"

    if snr is not None:

        if snr < 0:
            return "bad"

        if snr < 5:
            return "poor"

        if snr < 10:
            return "fair"

    return "good"


# ============================================================
# CHANNEL QUALITY AUDIT
# ============================================================

# What the code does:
# Calculates RMS, SD, PTP, minimum, maximum, SNR, flatline,
# saturation, and quality for every EEG channel.

def quality_audit(raw, eeg_channels):

    data = raw.get_data(
        picks=eeg_channels
    )

    results = {}

    for index, channel in enumerate(eeg_channels):

        x = data[index]

        rms_value = rms(x)
        sd_value = standard_deviation(x)
        ptp_value = peak_to_peak(x)

        flatline = detect_flatline(x)
        saturation = detect_saturation(x)
        snr = calculate_snr(x)

        quality = classify_quality(
            flatline,
            saturation["saturated"],
            snr
        )

        results[channel] = {
            "mean": float(np.mean(x)),
            "rms": rms_value,
            "standard_deviation": sd_value,
            "variance": float(np.var(x)),
            "peak_to_peak": ptp_value,
            "minimum": signal_minimum(x),
            "maximum": signal_maximum(x),
            "snr_db": snr,
            "flatline": flatline,
            "saturation": saturation["saturated"],
            "saturated_fraction": saturation["saturated_fraction"],
            "quality": quality,
        }

    return results


# ============================================================
# ROBUST CHANNEL OUTLIER SCORING
# ============================================================

# Mathematical representation:
#
# MAD = median(|x - median(x)|)
#
# robust_z =
#
# 0.6744897501960817 * (x - median(x)) / MAD
#
# What the code does:
# Compares channel-level RMS and variability against the
# distribution of EEG channels.

def channel_outlier_audit(raw, eeg_channels, quality):

    rms_values = np.array([
        quality[ch]["rms"]
        for ch in eeg_channels
    ])

    median_rms = float(np.median(rms_values))

    results = {}

    for channel in eeg_channels:

        x = raw.get_data(
            picks=[channel]
        )[0]

        median_value = float(np.median(x))

        mad = float(
            np.median(
                np.abs(x - median_value)
            )
        )

        if mad == 0:
            robust_z = 0.0
        else:
            robust_z = float(
                np.max(
                    np.abs(
                        0.6744897501960817 *
                        (x - median_value) /
                        mad
                    )
                )
            )

        ratio = (
            quality[channel]["rms"] / median_rms
            if median_rms != 0
            else 0.0
        )

        reasons = []

        if ratio < LOW_RMS_RATIO:
            reasons.append("low_amplitude")

        if ratio > HIGH_RMS_RATIO:
            reasons.append("high_amplitude")

        if quality[channel]["standard_deviation"] > 5 * np.median(
            [quality[ch]["standard_deviation"] for ch in eeg_channels]
        ):
            reasons.append("high_variability")

        if quality[channel]["flatline"]:
            reasons.append("flatline")

        if not np.all(np.isfinite(x)):
            reasons.append("non_finite")

        results[channel] = {
            "median_absolute_deviation": mad,
            "robust_z_max": robust_z,
            "rms_ratio_to_median": float(ratio),
            "outlier": bool(
                abs(robust_z) >= OUTLIER_Z_THRESHOLD
                or len(reasons) > 0
            ),
            "review_reasons": reasons,
        }

    return results


# ============================================================
# LINE-NOISE DETECTION
# ============================================================

# Mathematical representation:
#
# prominence =
#
# target_power / background_power
#
# detection:
#
# prominence >= 10
#
# What the code does:
# Examines the 60 Hz region of the signal spectrum.
#
# Why it is here:
# Strong electrical line noise can contaminate an EEG recording.

def detect_line_noise(x, sfreq):

    frequencies, powers = signal.periodogram(
        x,
        fs=sfreq
    )

    target_mask = (
        (frequencies >= LINE_NOISE_FREQUENCY - LINE_NOISE_TOLERANCE)
        &
        (frequencies <= LINE_NOISE_FREQUENCY + LINE_NOISE_TOLERANCE)
    )

    background_mask = (
        (frequencies >= LINE_NOISE_FREQUENCY - 5)
        &
        (frequencies <= LINE_NOISE_FREQUENCY + 5)
        &
        (~target_mask)
    )

    if not np.any(target_mask) or not np.any(background_mask):
        return {
            "detected": False,
            "prominence": 0.0
        }

    target_power = float(
        np.median(powers[target_mask])
    )

    background_power = float(
        np.median(powers[background_mask])
    )

    if background_power == 0:
        prominence = float("inf")
    else:
        prominence = target_power / background_power

    return {
        "detected": bool(
            prominence >= LINE_NOISE_PROMINENCE
        ),
        "prominence": float(prominence),
    }


# ============================================================
# TRANSIENT ARTIFACT DETECTION
# ============================================================

# Mathematical representation:
#
# z = abs((x - mean(x)) / SD(x))
#
# transient:
#
# z >= 6
#
# What the code does:
# Detects unusually large signal excursions.

def detect_transients(x):

    mean_value = np.mean(x)
    sd_value = np.std(x)

    if sd_value == 0:
        return {
            "count": 0,
            "indices": []
        }

    z = np.abs(
        (x - mean_value) / sd_value
    )

    indices = np.where(
        z >= TRANSIENT_Z
    )[0]

    return {
        "count": int(len(indices)),
        "indices": indices.tolist()
    }


# ============================================================
# STIMULATION ARTIFACT DETECTION
# ============================================================

# Mathematical representation:
#
# z = abs((x - mean(x)) / SD(x))
#
# stimulation artifact:
#
# z >= 8
#
# What the code does:
# Detects very large signal excursions using a more stringent
# threshold.
#
# Important:
# This detector identifies an amplitude pattern. It does not
# establish the biological cause of that pattern.

def detect_stimulation_artifacts(x):

    mean_value = np.mean(x)
    sd_value = np.std(x)

    if sd_value == 0:
        return {
            "count": 0,
            "indices": []
        }

    z = np.abs(
        (x - mean_value) / sd_value
    )

    indices = np.where(
        z >= STIMULATION_Z
    )[0]

    return {
        "count": int(len(indices)),
        "indices": indices.tolist()
    }


# ============================================================
# ELECTRODE PROBLEM DETECTION
# ============================================================

# Mathematical representation:
#
# SD <= 0.01 * range
#
# What the code does:
# Looks for flat or unusually low-variability channels.

def detect_electrode_problem(x):

    signal_range = np.ptp(x)

    if signal_range == 0:
        return {
            "detected": True,
            "reason": "flatline"
        }

    if np.std(x) <= 0.01 * signal_range:
        return {
            "detected": True,
            "reason": "low_variability"
        }

    return {
        "detected": False,
        "reason": None
    }


# ============================================================
# ARTIFACT AUDIT
# ============================================================

# What the code does:
# Runs all implemented artifact detectors on every EEG channel.

def artifact_audit(raw, eeg_channels):

    sfreq = float(raw.info["sfreq"])

    results = {}

    for channel in eeg_channels:

        x = raw.get_data(
            picks=[channel]
        )[0]

        line_noise = detect_line_noise(
            x,
            sfreq
        )

        transients = detect_transients(x)

        stimulation = detect_stimulation_artifacts(x)

        electrode = detect_electrode_problem(x)

        flags = []

        if line_noise["detected"]:
            flags.append("line_noise")

        if transients["count"] > 0:
            flags.append("transient")

        if stimulation["count"] > 0:
            flags.append("stimulation")

        if electrode["detected"]:
            flags.append("electrode_problem")

        results[channel] = {
            "line_noise": line_noise,
            "transient": transients,
            "stimulation": stimulation,
            "electrode_problem": electrode,
            "artifact_flags": flags,
            "artifact_present": len(flags) > 0,
        }

    return results


# ============================================================
# ANNOTATION AUDIT
# ============================================================

# What the code does:
# Summarizes annotation descriptions, counts, timing, and
# intervals.
#
# Why it is here:
# Event-locked analysis depends on the annotation structure
# of the recording.

def annotation_audit(raw):

    descriptions = list(
        raw.annotations.description
    )

    onsets = list(
        raw.annotations.onset
    )

    counts = {}

    for description in descriptions:
        counts[description] = (
            counts.get(description, 0) + 1
        )

    intervals = np.diff(
        sorted(onsets)
    )

    median_interval = (
        float(np.median(intervals))
        if len(intervals) > 0
        else None
    )

    return {
        "total_annotations": len(descriptions),
        "unique_descriptions": sorted(
            list(set(descriptions))
        ),
        "counts": counts,
        "first_annotation_time": (
            float(min(onsets))
            if onsets
            else None
        ),
        "last_annotation_time": (
            float(max(onsets))
            if onsets
            else None
        ),
        "median_annotation_interval": median_interval,
    }


# ============================================================
# MARKER-TO-EVENT VALIDATION
# ============================================================

# Mathematical representation:
#
# 11 -> 1
# 12 -> 2
# 13 -> 3
# 14 -> 4
# 15 -> 5
# 16 -> 6
#
# event_interval = event_time - marker_time
#
# What the code does:
# Searches the annotation sequence for expected marker/event
# relationships.
#
# Important:
# These mappings establish annotation relationships. They do
# not assign biological meaning to the event codes.

MARKER_MAP = {
    "11": "1",
    "12": "2",
    "13": "3",
    "14": "4",
    "15": "5",
    "16": "6",
}


def validate_events(raw):

    descriptions = list(
        raw.annotations.description
    )

    onsets = list(
        raw.annotations.onset
    )

    validated = []

    for index, description in enumerate(descriptions):

        if description not in MARKER_MAP:
            continue

        expected = MARKER_MAP[description]

        for next_index in range(
            index + 1,
            len(descriptions)
        ):

            next_description = descriptions[next_index]

            if next_description == expected:

                interval = (
                    onsets[next_index] - onsets[index]
                )

                validated.append({
                    "marker": description,
                    "event": expected,
                    "marker_time": float(onsets[index]),
                    "event_time": float(onsets[next_index]),
                    "interval_seconds": float(interval),
                })

                break

    return validated


# ============================================================
# EVENT-LOCKED EPOCHING
# ============================================================

# Mathematical representation:
#
# epoch =
#
# X[t_event - 0.2 : t_event + 0.8]
#
# What the code does:
# Creates an event-centered representation of the EEG.
#
# Why it is here:
# Event responses must be compared within the same temporal
# window across trials.

def create_epochs(raw, validated_events):

    if not validated_events:
        return None

    event_times = [
        event["event_time"]
        for event in validated_events
    ]

    unique_times = []
    unique_codes = []

    for event in validated_events:

        if event["event_time"] in unique_times:
            continue

        unique_times.append(
            event["event_time"]
        )

        unique_codes.append(
            event["event"]
        )

    sfreq = float(raw.info["sfreq"])

    events = []

    event_id = {}

    for index, (time, code) in enumerate(
        zip(unique_times, unique_codes)
    ):

        if code not in event_id:
            event_id[code] = len(event_id) + 1

        sample = int(
            round(
                (time - raw.first_time) * sfreq
            )
        )

        events.append([
            sample,
            0,
            event_id[code]
        ])

    events = np.asarray(
        events,
        dtype=int
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EVENT_TMIN,
        tmax=EVENT_TMAX,
        baseline=(BASELINE_START, BASELINE_END),
        picks="eeg",
        reject_by_annotation=False,
        preload=True,
        verbose=False,
    )

    return epochs


# ============================================================
# BASELINE / RESPONSE FEATURES
# ============================================================

# Mathematical representation:
#
# baseline_mean = mean(x_baseline)
#
# event_mean = mean(x_response)
#
# change = event_mean - baseline_mean
#
# What the code does:
# Calculates the primary event-response measurements.

def event_response_features(epochs):

    if epochs is None or len(epochs) == 0:
        return []

    data = epochs.get_data()

    times = epochs.times

    baseline_mask = (
        (times >= BASELINE_START)
        &
        (times <= BASELINE_END)
    )

    response_mask = (
        (times >= RESPONSE_START)
        &
        (times <= RESPONSE_END)
    )

    records = []

    for trial_index in range(data.shape[0]):

        event_code = str(
            epochs.events[trial_index, 2]
        )

        for channel_index, channel in enumerate(
            epochs.ch_names
        ):

            x = data[
                trial_index,
                channel_index
            ]

            baseline = x[baseline_mask]
            response = x[response_mask]

            baseline_mean = float(
                np.mean(baseline)
            )

            response_mean = float(
                np.mean(response)
            )

            mean_change = (
                response_mean - baseline_mean
            )

            if baseline_mean == 0:
                percent_change = None
            else:
                percent_change = (
                    100.0 *
                    mean_change /
                    abs(baseline_mean)
                )

            baseline_rms = rms(baseline)
            response_rms = rms(response)

            rms_change = (
                response_rms - baseline_rms
            )

            if baseline_rms == 0:
                rms_percent_change = None
            else:
                rms_percent_change = (
                    100.0 *
                    rms_change /
                    abs(baseline_rms)
                )

            records.append({
                "trial": trial_index,
                "event_code": event_code,
                "channel": channel,
                "baseline_mean": baseline_mean,
                "event_mean": response_mean,
                "mean_change": float(mean_change),
                "percent_change": (
                    float(percent_change)
                    if percent_change is not None
                    else None
                ),
                "baseline_rms": baseline_rms,
                "event_rms": response_rms,
                "rms_change": float(rms_change),
                "rms_percent_change": (
                    float(rms_percent_change)
                    if rms_percent_change is not None
                    else None
                ),
                "baseline_peak_to_peak": peak_to_peak(
                    baseline
                ),
                "event_peak_to_peak": peak_to_peak(
                    response
                ),
                "response_peak": float(
                    np.max(response)
                ),
                "response_trough": float(
                    np.min(response)
                ),
            })

    return records


# ============================================================
# EVOKED-RESPONSE PEAK MEASUREMENT
# ============================================================

# Mathematical representation:
#
# positive:
#
#     peak_index = argmax(x)
#
# negative:
#
#     peak_index = argmin(x)
#
# latency = peak_index / f_s
#
# What the code does:
# Finds the strongest positive or negative response.

def evoked_peak(
    response,
    sfreq,
    polarity="positive"
):

    if polarity == "negative":

        index = int(
            np.argmin(response)
        )

        amplitude = float(
            abs(response[index])
        )

    else:

        index = int(
            np.argmax(response)
        )

        amplitude = float(
            response[index]
        )

    latency = (
        index / sfreq
    )

    return {
        "index": index,
        "amplitude": amplitude,
        "latency_seconds": float(latency),
    }


# ============================================================
# CHANGE DETECTION
# ============================================================

# Mathematical representation:
#
# z =
#
# (current - baseline_mean) / baseline_SD
#
# changed if:
#
# abs(z) >= 2
#
# What the code does:
# Determines whether an observation differs substantially from
# its baseline.

def detect_change(
    current,
    baseline_mean,
    baseline_sd
):

    if baseline_sd != 0:

        z = (
            current - baseline_mean
        ) / baseline_sd

        changed = (
            abs(z) >= CHANGE_Z_THRESHOLD
        )

    else:

        z = None

        changed = (
            abs(current - baseline_mean)
            >= CHANGE_ABSOLUTE_THRESHOLD
        )

    return {
        "changed": bool(changed),
        "z_score": (
            float(z)
            if z is not None
            else None
        ),
        "absolute_change": float(
            abs(current - baseline_mean)
        ),
    }


# ============================================================
# PERSISTENT CHANGE DETECTION
# ============================================================

# Mathematical representation:
#
# persistent change requires:
#
#     >= 3 consecutive changed observations
#
# What the code does:
# Determines whether a detected change continues across
# multiple consecutive observations.

def persistent_change(changed_values):

    consecutive = 0

    for changed in changed_values:

        if changed:

            consecutive += 1

            if consecutive >= MINIMUM_CONSECUTIVE:
                return True

        else:

            consecutive = 0

    return False


# ============================================================
# TRIAL-LEVEL STATISTICS
# ============================================================

# Mathematical representation:
#
# mean_change
# median_change
# SD_change
# MAD_change
#
# What the code does:
# Aggregates event-response measurements across trials.

def trial_statistics(records):

    grouped = {}

    for record in records:

        key = (
            record["event_code"],
            record["channel"]
        )

        grouped.setdefault(
            key,
            []
        ).append(record)

    results = []

    for (event_code, channel), group in grouped.items():

        changes = np.array([
            r["mean_change"]
            for r in group
        ])

        rms_changes = np.array([
            r["rms_change"]
            for r in group
        ])

        peaks = np.array([
            r["response_peak"]
            for r in group
        ])

        troughs = np.array([
            r["response_trough"]
            for r in group
        ])

        results.append({
            "event_code": event_code,
            "channel": channel,
            "n_trials": int(len(group)),
            "mean_change": float(
                np.mean(changes)
            ),
            "median_change": float(
                np.median(changes)
            ),
            "standard_deviation": float(
                np.std(changes, ddof=1)
            ) if len(changes) > 1 else 0.0,
            "robust_spread_mad": float(
                np.median(
                    np.abs(
                        changes -
                        np.median(changes)
                    )
                )
            ),
            "mean_rms_change": float(
                np.mean(rms_changes)
            ),
            "median_rms_change": float(
                np.median(rms_changes)
            ),
            "response_peak_mean": float(
                np.mean(peaks)
            ),
            "response_trough_mean": float(
                np.mean(troughs)
            ),
        })

    return results


# ============================================================
# KRUSKAL-WALLIS STATISTICS
# ============================================================

# Mathematical representation:
#
# H = KruskalWallis(group_1, ..., group_k)
#
# What the code does:
# Compares event-class response distributions within each EEG
# channel.
#
# Why it is here:
# The pipeline can determine whether response measurements differ
# between multiple event classes.

def statistical_comparisons(records):

    by_channel = {}

    for record in records:

        by_channel.setdefault(
            record["channel"],
            {}
        )

        by_channel[
            record["channel"]
        ].setdefault(
            record["event_code"],
            []
        ).append(
            record["mean_change"]
        )

    results = []

    for channel, groups in by_channel.items():

        usable = [
            values
            for values in groups.values()
            if len(values) > 0
        ]

        if len(usable) < 2:
            continue

        try:

            H, p = kruskal(
                *usable
            )

        except ValueError:

            continue

        k = len(usable)

        N = sum(
            len(values)
            for values in usable
        )

        if N > k:

            epsilon_squared = (
                H - k + 1
            ) / (
                N - k
            )

            epsilon_squared = max(
                0.0,
                float(epsilon_squared)
            )

        else:

            epsilon_squared = 0.0

        results.append({
            "channel": channel,
            "groups": list(groups.keys()),
            "H_statistic": float(H),
            "p_value": float(p),
            "epsilon_squared": epsilon_squared,
            "significant_uncorrected": bool(
                p < STATISTICAL_ALPHA
            ),
        })

    return results


# ============================================================
# BONFERRONI CORRECTION
# ============================================================

# Mathematical representation:
#
# alpha_corrected = alpha / n_tests
#
# What the code does:
# Corrects the statistical threshold for multiple channel
# comparisons.

def bonferroni_correction(statistics):

    n_tests = len(statistics)

    if n_tests == 0:
        return statistics, None

    corrected_alpha = (
        STATISTICAL_ALPHA / n_tests
    )

    for result in statistics:

        result["bonferroni_alpha"] = (
            corrected_alpha
        )

        result["significant_bonferroni"] = bool(
            result["p_value"] < corrected_alpha
        )

    return statistics, corrected_alpha


# ============================================================
# BOUNDARY AUDIT
# ============================================================

# Mathematical representation:
#
# boundary =
#
# [onset, onset + duration]
#
# What the code does:
# Identifies recording-boundary annotations.

def boundary_intervals(raw):

    boundaries = []

    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description
    ):

        if description == "boundary":

            boundaries.append({
                "start": float(onset),
                "end": float(
                    onset + duration
                )
            })

    return boundaries


# ============================================================
# EPOCH-BOUNDARY OVERLAP
# ============================================================

# Mathematical representation:
#
# epoch_start <= boundary_end
# AND
# epoch_end >= boundary_start
#
# What the code does:
# Determines whether an event epoch crosses a recording boundary.

def epoch_overlaps_boundary(
    event_time,
    boundary
):

    epoch_start = (
        event_time + EVENT_TMIN
    )

    epoch_end = (
        event_time + EVENT_TMAX
    )

    return (
        epoch_start <= boundary["end"]
        and
        epoch_end >= boundary["start"]
    )


# ============================================================
# CLEAN EPOCH COUNT
# ============================================================

# What the code does:
# Counts event epochs that do not overlap recording boundaries.

def clean_epoch_count(
    epochs,
    raw
):

    if epochs is None:
        return {
            "total_epochs": 0,
            "boundary_overlap_epochs": 0,
            "clean_epochs": 0,
        }

    boundaries = boundary_intervals(raw)

    overlap_count = 0

    for event in epochs.events:

        event_time = (
            raw.first_time +
            event[0] / raw.info["sfreq"]
        )

        if any(
            epoch_overlaps_boundary(
                event_time,
                boundary
            )
            for boundary in boundaries
        ):

            overlap_count += 1

    total = len(epochs)

    return {
        "total_epochs": int(total),
        "boundary_annotations": int(
            len(boundaries)
        ),
        "boundary_overlap_epochs": int(
            overlap_count
        ),
        "clean_epochs": int(
            total - overlap_count
        ),
    }


# ============================================================
# EVIDENCE CLASSIFICATION
# ============================================================

# Mathematical representation:
#
# quality
# artifact
# physiological_change
# persistence
#
# -> evidence classification
#
# What the code does:
# Combines quality, artifact, and change information into the
# implemented evidence classes.
#
# Important:
# These classifications are research/engineering evidence
# categories and are not clinical diagnoses.

def classify_evidence(
    quality,
    artifact_present,
    physiological_change,
    persistent
):

    if quality == "bad":

        return {
            "classification": "uncertain",
            "confidence": 0.90
        }

    if quality == "poor":

        return {
            "classification": "uncertain",
            "confidence": 0.75
        }

    if (
        artifact_present
        and
        not physiological_change
    ):

        return {
            "classification": "artifact",
            "confidence": 0.85
        }

    if (
        physiological_change
        and
        persistent
        and
        not artifact_present
    ):

        return {
            "classification":
                "possible_physiological_change",
            "confidence": 0.85
        }

    if (
        physiological_change
        and
        persistent
        and
        artifact_present
    ):

        return {
            "classification": "uncertain",
            "confidence": 0.70
        }

    if (
        physiological_change
        and
        not persistent
    ):

        return {
            "classification": "uncertain",
            "confidence": 0.65
        }

    return {
        "classification": "normal",
        "confidence": 0.90
    }


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

# What the code does:
# Creates channel-level evidence classifications from the
# quality, artifact, and response measurements.

def evidence_summary(
    quality,
    artifacts,
    records
):

    grouped = {}

    for record in records:

        grouped.setdefault(
            record["channel"],
            []
        ).append(record)

    results = []

    for channel in quality:

        channel_records = grouped.get(
            channel,
            []
        )

        physiological_change = False

        if channel_records:

            mean_changes = np.array([
                r["mean_change"]
                for r in channel_records
            ])

            baseline_values = np.array([
                r["baseline_mean"]
                for r in channel_records
            ])

            baseline_sd = float(
                np.std(
                    baseline_values
                )
            )

            baseline_mean = float(
                np.mean(
                    baseline_values
                )
            )

            for change in mean_changes:

                test = detect_change(
                    change,
                    baseline_mean,
                    baseline_sd
                )

                if test["changed"]:
                    physiological_change = True
                    break

        persistent = (
            len(channel_records)
            >= MINIMUM_CONSECUTIVE
        )

        artifact_present = artifacts[
            channel
        ]["artifact_present"]

        classification = classify_evidence(
            quality[channel]["quality"],
            artifact_present,
            physiological_change,
            persistent
        )

        results.append({
            "channel": channel,
            "quality": quality[channel]["quality"],
            "artifact_present": artifact_present,
            "physiological_change": physiological_change,
            "persistent_change": persistent,
            "classification": classification[
                "classification"
            ],
            "confidence": classification[
                "confidence"
            ],
        })

    return results


# ============================================================
# REPORT GENERATION
# ============================================================

# What the code does:
# Creates a JSON evidence report containing the source recording,
# analysis parameters, quality measurements, artifact findings,
# events, epochs, response measurements, statistics, and evidence.
#
# Why it is here:
# A structured report makes the computational result
# reproducible and inspectable.

def create_report(
    path,
    metadata,
    roles,
    quality,
    outliers,
    artifacts,
    annotations,
    validated_events,
    epoch_info,
    response_records,
    trial_stats,
    statistics,
    evidence
):

    report = {
        "source_recording": str(path),
        "analysis_type": "IONM Signal Intelligence",
        "research_engineering_only": True,

        "parameters": {
            "high_pass_hz": HIGH_PASS,
            "low_pass_hz": LOW_PASS,
            "epoch_tmin_seconds": EVENT_TMIN,
            "epoch_tmax_seconds": EVENT_TMAX,
            "baseline_start_seconds": BASELINE_START,
            "baseline_end_seconds": BASELINE_END,
            "response_start_seconds": RESPONSE_START,
            "response_end_seconds": RESPONSE_END,
            "change_z_threshold": CHANGE_Z_THRESHOLD,
            "absolute_change_threshold": CHANGE_ABSOLUTE_THRESHOLD,
            "minimum_consecutive": MINIMUM_CONSECUTIVE,
            "amplitude_change_threshold": AMPLITUDE_CHANGE_THRESHOLD,
            "latency_change_threshold": LATENCY_CHANGE_THRESHOLD,
            "statistical_alpha": STATISTICAL_ALPHA,
        },

        "metadata": metadata,
        "channel_roles": roles,
        "quality": quality,
        "channel_outliers": outliers,
        "artifacts": artifacts,
        "annotations": annotations,
        "validated_marker_event_pairs": validated_events,
        "epochs": epoch_info,
        "event_response_records": response_records,
        "trial_statistics": trial_stats,
        "statistical_comparisons": statistics,
        "evidence": evidence,
    }

    return report


# ============================================================
# GRAPHICAL REPORT
# ============================================================

# What the code does:
# Produces a graphical summary of the recording, quality,
# event response, and spectral information.
#
# Why it is here:
# The visual report allows the numerical measurements to be
# inspected alongside the original signal.

def display_report(
    raw,
    filtered,
    quality,
    artifacts,
    response_records,
    statistics
):

    eeg_channels = list(
        quality.keys()
    )

    if not eeg_channels:
        return

    display_channels = eeg_channels[:4]

    fig = plt.figure(
        figsize=(15, 11)
    )

    grid = fig.add_gridspec(
        3,
        2
    )

    # --------------------------------------------------------
    # SIGNAL PLOT
    # --------------------------------------------------------

    ax_signal = fig.add_subplot(
        grid[0, :]
    )

    duration = min(
        10.0,
        raw.times[-1]
    )

    samples = np.searchsorted(
        raw.times,
        duration
    )

    for channel in display_channels:

        x = filtered.get_data(
            picks=[channel],
            start=0,
            stop=samples
        )[0]

        ax_signal.plot(
            filtered.times[:samples],
            x,
            label=channel
        )

    ax_signal.set_title(
        "Filtered EEG Signal"
    )

    ax_signal.set_xlabel(
        "Time (seconds)"
    )

    ax_signal.set_ylabel(
        "Amplitude"
    )

    ax_signal.legend(
        loc="upper right"
    )

    # --------------------------------------------------------
    # CHANNEL RMS
    # --------------------------------------------------------

    ax_rms = fig.add_subplot(
        grid[1, 0]
    )

    names = list(
        quality.keys()
    )

    rms_values = [
        quality[name]["rms"]
        for name in names
    ]

    ax_rms.bar(
        names,
        rms_values
    )

    ax_rms.set_title(
        "Channel RMS"
    )

    ax_rms.set_ylabel(
        "RMS"
    )

    ax_rms.tick_params(
        axis="x",
        rotation=90
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    ax_quality = fig.add_subplot(
        grid[1, 1]
    )

    quality_numbers = []

    for name in names:

        quality_name = quality[
            name
        ]["quality"]

        quality_numbers.append({
            "bad": 0,
            "poor": 1,
            "fair": 2,
            "good": 3
        }.get(
            quality_name,
            0
        ))

    ax_quality.bar(
        names,
        quality_numbers
    )

    ax_quality.set_title(
        "Channel Quality"
    )

    ax_quality.set_ylabel(
        "Quality Level"
    )

    ax_quality.set_yticks(
        [0, 1, 2, 3],
        ["bad", "poor", "fair", "good"]
    )

    ax_quality.tick_params(
        axis="x",
        rotation=90
    )

    # --------------------------------------------------------
    # EVENT RESPONSE
    # --------------------------------------------------------

    ax_response = fig.add_subplot(
        grid[2, 0]
    )

    if response_records:

        changes = [
            r["mean_change"]
            for r in response_records
        ]

        ax_response.hist(
            changes,
            bins=20
        )

    ax_response.set_title(
        "Event Mean-Response Change"
    )

    ax_response.set_xlabel(
        "Response Change"
    )

    ax_response.set_ylabel(
        "Count"
    )

    # --------------------------------------------------------
    # STATISTICAL RESULTS
    # --------------------------------------------------------

    ax_stats = fig.add_subplot(
        grid[2, 1]
    )

    if statistics:

        p_values = [
            -math.log10(
                max(
                    result["p_value"],
                    1e-300
                )
            )
            for result in statistics
        ]

        stat_channels = [
            result["channel"]
            for result in statistics
        ]

        ax_stats.bar(
            stat_channels,
            p_values
        )

    ax_stats.set_title(
        "-log10(p) by Channel"
    )

    ax_stats.set_ylabel(
        "-log10(p)"
    )

    ax_stats.tick_params(
        axis="x",
        rotation=90
    )

    fig.suptitle(
        "IONM Signal Intelligence Report",
        fontsize=16
    )

    fig.tight_layout()

    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================

# What the code does:
# Executes the complete IONM signal-intelligence workflow.
#
# Why it is here:
# This connects every individual analysis stage into one
# reproducible pipeline.

def main():

    print()
    print("=" * 70)
    print("IONM SIGNAL INTELLIGENCE PIPELINE")
    print("=" * 70)

    print()
    print("SELECT EEG / IONM RECORDING")

    path = select_eeg_file()

    print()
    print(f"Source: {path}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("[1/15] Loading recording...")

    raw = load_recording(path)

    metadata = recording_metadata(
        raw
    )

    print(
        f"Channels: {metadata['n_channels']}"
    )

    print(
        f"Samples: {metadata['n_samples']}"
    )

    print(
        f"Sampling frequency: "
        f"{metadata['sampling_frequency_hz']:.2f} Hz"
    )

    print(
        f"Duration: "
        f"{metadata['duration_seconds']:.2f} seconds"
    )

    # --------------------------------------------------------
    # CHANNEL ROLES
    # --------------------------------------------------------

    print(
        "\n[2/15] Assigning channel roles..."
    )

    roles = assign_channel_roles(
        raw
    )

    eeg_channels = get_eeg_channels(
        raw,
        roles
    )

    print(
        f"EEG channels: {len(eeg_channels)}"
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    print(
        "\n[3/15] Preprocessing EEG..."
    )

    filtered = preprocess_eeg(
        raw
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    print(
        "\n[4/15] Calculating channel quality..."
    )

    quality = quality_audit(
        filtered,
        eeg_channels
    )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    print(
        "\n[5/15] Calculating channel outliers..."
    )

    outliers = channel_outlier_audit(
        filtered,
        eeg_channels,
        quality
    )

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    print(
        "\n[6/15] Detecting artifacts..."
    )

    artifacts = artifact_audit(
        filtered,
        eeg_channels
    )

    # --------------------------------------------------------
    # ANNOTATIONS
    # --------------------------------------------------------

    print(
        "\n[7/15] Auditing annotations..."
    )

    annotations = annotation_audit(
        raw
    )

    print(
        f"Annotations: "
        f"{annotations['total_annotations']}"
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    print(
        "\n[8/15] Validating marker-event relationships..."
    )

    validated_events = validate_events(
        raw
    )

    print(
        f"Validated event pairs: "
        f"{len(validated_events)}"
    )

    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------

    print(
        "\n[9/15] Creating event-locked epochs..."
    )

    epochs = create_epochs(
        filtered,
        validated_events
    )

    epoch_info = clean_epoch_count(
        epochs,
        raw
    )

    print(
        f"Total epochs: "
        f"{epoch_info['total_epochs']}"
    )

    print(
        f"Boundary-overlap epochs: "
        f"{epoch_info['boundary_overlap_epochs']}"
    )

    print(
        f"Clean epochs: "
        f"{epoch_info['clean_epochs']}"
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    print(
        "\n[10/15] Calculating event-response features..."
    )

    response_records = (
        event_response_features(
            epochs
        )
    )

    print(
        f"Response records: "
        f"{len(response_records)}"
    )

    # --------------------------------------------------------
    # TRIAL STATISTICS
    # --------------------------------------------------------

    print(
        "\n[11/15] Calculating trial statistics..."
    )

    trial_stats = trial_statistics(
        response_records
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print(
        "\n[12/15] Running statistical comparisons..."
    )

    statistics = statistical_comparisons(
        response_records
    )

    statistics, corrected_alpha = (
        bonferroni_correction(
            statistics
        )
    )

    print(
        f"Statistical comparisons: "
        f"{len(statistics)}"
    )

    if corrected_alpha is not None:

        print(
            f"Bonferroni alpha: "
            f"{corrected_alpha:.6g}"
        )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    print(
        "\n[13/15] Fusing evidence..."
    )

    evidence = evidence_summary(
        quality,
        artifacts,
        response_records
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "\n[14/15] Creating evidence report..."
    )

    report = create_report(
        path,
        metadata,
        roles,
        quality,
        outliers,
        artifacts,
        annotations,
        validated_events,
        epoch_info,
        response_records,
        trial_stats,
        statistics,
        evidence
    )

    output_directory = (
        path.parent /
        f"{path.stem}_IONM_REPORT"
    )

    output_directory.mkdir(
        exist_ok=True
    )

    report_path = (
        output_directory /
        "evidence_report.json"
    )

    def json_safe(value):
        # Convert NumPy integers into normal Python integers.
        if isinstance(value, np.integer):
            return int(value)

        # Convert NumPy floating-point values into normal Python floats.
        if isinstance(value, np.floating):
            return float(value)

        # Convert NumPy booleans into normal Python booleans.
        if isinstance(value, np.bool_):
            return bool(value)

        # Convert NumPy arrays into normal Python lists.
        if isinstance(value, np.ndarray):
            return value.tolist()

        # Recursively convert values inside dictionaries.
        if isinstance(value, dict):
            return {
                json_safe(key): json_safe(item)
                for key, item in value.items()
            }

        # Recursively convert values inside lists and tuples.
        if isinstance(value, (list, tuple)):
            return [
                json_safe(item)
                for item in value
            ]

        # Leave ordinary Python values unchanged.
        return value

    # Make the entire evidence report JSON-compatible.
    report = json_safe(report)

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False
        )

    print(
        f"Report written to:\n"
        f"{report_path}"
    )

    # --------------------------------------------------------
    # GRAPHICAL REPORT
    # --------------------------------------------------------

    print(
        "\n[15/15] Opening graphical report..."
    )

    display_report(
        raw,
        filtered,
        quality,
        artifacts,
        response_records,
        statistics
    )

    print()
    print("=" * 70)
    print("IONM ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Evidence report: {report_path}")
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

# What the code does:
# Starts the pipeline when this file is executed directly.
#
# Why it is here:
# It prevents the pipeline from automatically running when the
# file is imported by another Python program.

if __name__ == "__main__":
    main()


# ============================================================
# TEST 1 — MEAN
# ============================================================
# What the test does:
# Calculates the average amplitude of the EEG signal.
#
# How it is calculated:
# mean = sum of all signal samples / number of samples
#
# Mathematical equation:
# x̄ = (1 / N) * Σ(xᵢ)
#
# What the test means:
# The mean represents the average electrical amplitude of the
# EEG signal during the analyzed recording period.
#
# Why it is useful:
# It provides a measure of the signal's central amplitude.


# ============================================================
# TEST 2 — STANDARD DEVIATION
# ============================================================
# What the test does:
# Measures how much the EEG signal varies around its mean.
#
# How it is calculated:
# σ = √[(1 / N) * Σ(xᵢ - x̄)²]
#
# The pipeline calculates the difference between each sample
# and the mean, squares those differences, averages them, and
# takes the square root.
#
# What the test means:
# A larger standard deviation indicates greater variation in
# the signal's amplitude.
#
# A smaller standard deviation indicates that the signal is
# more consistent around its mean.
#
# Why it is useful:
# It helps characterize the variability and stability of the
# EEG recording.


# ============================================================
# TEST 3 — VARIANCE
# ============================================================
# What the test does:
# Measures the overall variability of the EEG signal.
#
# How it is calculated:
# variance = (1 / N) * Σ(xᵢ - x̄)²
#
# The pipeline calculates the difference between every sample
# and the mean, squares the differences, and calculates their
# average.
#
# What the test means:
# Variance describes how widely the signal values are spread
# around the mean.
#
# Why it is useful:
# It provides a numerical measurement of the amount of
# variability present in the EEG signal.


# ============================================================
# TEST 4 — RMS (ROOT MEAN SQUARE)
# ============================================================
# What the test does:
# Measures the overall magnitude of the EEG signal.
#
# How it is calculated:
# RMS = √[(1 / N) * Σ(xᵢ²)]
#
# The pipeline squares every signal sample, calculates the
# average of those squared values, and takes the square root.
#
# What the test means:
# RMS represents the overall magnitude of the electrical signal.
#
# Positive and negative amplitudes both contribute to the
# measurement because the values are squared.
#
# Why it is useful:
# It provides a measurement of the overall strength of the
# EEG signal.


# ============================================================
# TEST 5 — PEAK-TO-PEAK AMPLITUDE
# ============================================================
# What the test does:
# Measures the complete amplitude range of the EEG signal.
#
# How it is calculated:
# peak-to-peak = maximum signal value - minimum signal value
#
# The pipeline identifies the highest and lowest values in the
# signal and calculates the difference between them.
#
# What the test means:
# Peak-to-peak amplitude represents the total excursion of the
# signal from its lowest point to its highest point.
#
# A larger value indicates a larger amplitude range.
#
# Why it is useful:
# It helps identify the overall range and size of signal
# excursions within the recording.


# ============================================================
# TEST 6 — SNR (SIGNAL-TO-NOISE RATIO)
# ============================================================
# What the test does:
# Estimates the amount of measurable signal relative to noise.
#
# How it is calculated:
# SNR = 10 * log10(P_signal / P_noise)
#
# Where:
# P_signal = signal power
# P_noise  = noise power
#
# Signal and noise power are calculated from mean squared
# amplitude values.
#
# The power ratio is converted into decibels (dB).
#
# What the test means:
# A higher SNR indicates that the measured signal is stronger
# relative to the measured noise.
#
# A lower SNR indicates that noise represents a larger portion
# of the measured signal.
#
# Why it is useful:
# SNR provides a quantitative measurement of the recording's
# signal quality and helps determine how clearly the measurable
# EEG signal stands out from noise.

