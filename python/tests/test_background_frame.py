"""
Background frame subtraction and post-FFT frame correction tests (line-field OCT)

Covers: recording, static subtraction, EMA update, 2D NumPy profile round trip,
raw file save/load, reset, and post-FFT frame correction on the CPU backend
(the C++ test suite covers CUDA and cross-backend comparison in depth).
"""

import os
import sys
import time
import tempfile

import numpy as np

try:
    import octproengine as ope
except ImportError as e:
    print(f"Failed to import octproengine: {e}")
    sys.exit(1)

SIGNAL_LENGTH = 64
ASCANS_PER_BSCAN = 8
SAMPLES_PER_BSCAN = SIGNAL_LENGTH * ASCANS_PER_BSCAN


def make_processor(bscans_per_buffer):
    proc = ope.Processor(ope.Backend.CPU)
    proc.set_input_parameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, bscans_per_buffer, ope.DataType.UINT16)
    proc.enable_log_scaling(False)
    proc.set_grayscale_range(0.0, 1.0)
    proc.set_signal_multiplicator_and_addend(1.0, 0.0)
    return proc


def process_buffer(proc, data):
    """Process one buffer and return the output as a NumPy array"""
    result = {}

    def on_output(output_array, buffer_id):
        result["output"] = np.copy(output_array).reshape(-1)

    callback_id = proc.add_output_callback(on_output)
    buffer = proc.get_next_available_buffer()
    buffer[:] = np.asarray(data, dtype=np.uint16).reshape(buffer.shape)
    proc.process(buffer)

    deadline = time.time() + 10.0
    while "output" not in result:
        if time.time() > deadline:
            raise RuntimeError("Timed out waiting for output")
        time.sleep(0.001)

    proc.remove_output_callback(callback_id)
    return result["output"]


def test_recording_and_subtraction():
    print("Test: recording and static subtraction...")
    proc = make_processor(1)
    proc.set_background_frame_bscans_to_average(2)
    proc.initialize()

    # Record 2 constant B-scans of value 300
    background_data = np.full(SAMPLES_PER_BSCAN, 300, dtype=np.uint16)
    proc.request_background_frame_recording()
    process_buffer(proc, background_data)
    process_buffer(proc, background_data)

    assert proc.has_background_frame_profile(), "Profile must exist after recording"
    profile = proc.get_background_frame_profile()
    assert profile.shape == (ASCANS_PER_BSCAN, SIGNAL_LENGTH), f"Unexpected shape {profile.shape}"
    assert np.allclose(profile, 300.0, atol=0.01), "Recorded profile must be 300"

    # With subtraction enabled, processing the recorded background must yield ~zero output
    proc.enable_background_frame_subtraction(True)
    output = process_buffer(proc, background_data)
    assert np.allclose(output, 0.0, atol=0.01), "Subtracting the recorded background must null the output"
    print("  PASSED")


def test_profile_roundtrip_and_reset():
    print("Test: 2D profile set/get, raw file round trip, reset...")
    proc = make_processor(1)
    proc.initialize()

    frame = np.random.default_rng(42).uniform(0, 1000, (ASCANS_PER_BSCAN, SIGNAL_LENGTH)).astype(np.float32)
    proc.set_background_frame_profile(frame)
    returned = proc.get_background_frame_profile()
    assert np.array_equal(returned, frame), "Profile round trip through the backend must be exact"

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "background_frame.raw")
        proc.save_background_frame_profile_to_file(filepath)
        assert os.path.getsize(filepath) == SAMPLES_PER_BSCAN * 4, "Raw file must be float32 samples only"

        proc.reset_background_frame()
        assert not proc.has_background_frame_profile(), "Reset must clear the profile"

        proc.load_background_frame_profile_from_file(filepath)
        loaded = proc.get_background_frame_profile()
        assert np.array_equal(loaded, frame), "Raw file round trip must be exact"

    # Wrong shape and negative values must be rejected
    # (re-raise AssertionError so a missing rejection cannot be swallowed)
    try:
        proc.set_background_frame_profile(np.zeros((3, 3), dtype=np.float32))
        assert False, "Wrong shape must be rejected"
    except AssertionError:
        raise
    except Exception:
        pass
    try:
        proc.set_background_frame_profile(np.full((ASCANS_PER_BSCAN, SIGNAL_LENGTH), -1.0, dtype=np.float32))
        assert False, "Negative values must be rejected"
    except AssertionError:
        raise
    except Exception:
        pass
    print("  PASSED")


def test_continuous_ema():
    print("Test: continuous EMA update...")
    proc = make_processor(1)
    proc.set_background_frame_bscans_to_average(2)  # alpha = 1/2
    proc.enable_background_frame_subtraction(True)
    proc.enable_continuous_background_frame_update(True)
    proc.initialize()

    data = np.full(SAMPLES_PER_BSCAN, 100, dtype=np.uint16)
    process_buffer(proc, data)  # bg: 0 -> 50
    process_buffer(proc, data)  # bg: 50 -> 75

    profile = proc.get_background_frame_profile()
    assert np.allclose(profile, 75.0, atol=0.01), f"EMA background must be 75, got {profile.flat[0]}"
    print("  PASSED")


def test_frame_correction():
    print("Test: post-FFT frame correction...")
    proc = make_processor(1)
    proc.enable_post_fft_frame_correction(True)
    proc.initialize()

    # A-scan 1 has 4x the value of A-scan 0 -> corrected DC ratio must be 2 (divides by sqrt of gain)
    data = np.zeros(SAMPLES_PER_BSCAN, dtype=np.uint16)
    data[0:SIGNAL_LENGTH] = 100
    data[SIGNAL_LENGTH:2 * SIGNAL_LENGTH] = 400
    output = process_buffer(proc, data)

    output_ascan_length = SIGNAL_LENGTH // 2
    dc0 = output[0]
    dc1 = output[output_ascan_length]
    assert abs(dc1 / dc0 - 2.0) < 0.01, f"Corrected ratio must be 2, got {dc1 / dc0}"
    print("  PASSED")


def test_config_copy_and_type_change():
    print("Test: config property is a copy; type change via set_config resizes buffers...")
    proc = make_processor(1)
    proc.initialize()

    # The config property must return a copy: mutating it must not silently change
    # the processor (and must not corrupt set_config's change detection)
    cfg = proc.config
    cfg.dataParams.signalLength = 999
    assert proc.config.dataParams.signalLength == SIGNAL_LENGTH, "config property must be a copy"

    # Data type change through a mutated copy must reallocate the input buffers
    cfg = proc.config
    cfg.dataParams.inputDataType = ope.DataType.UINT8
    proc.set_config(cfg)
    buffer = proc.get_next_available_buffer()
    assert buffer.nbytes == SAMPLES_PER_BSCAN, \
        f"Buffer must be uint8-sized ({SAMPLES_PER_BSCAN} bytes), got {buffer.nbytes}"
    print("  PASSED")


def test_unsupported_backend():
    print("Test: enabling on OpenCL backend must throw...")
    try:
        proc = ope.Processor(ope.Backend.OPENCL)
    except Exception as e:
        print(f"  SKIPPED (OpenCL not available: {e})")
        return
    try:
        proc.enable_background_frame_subtraction(True)
        assert False, "Enabling on OpenCL must raise"
    except AssertionError:
        raise
    except Exception:
        pass
    print("  PASSED")


def main():
    print("=" * 60)
    print("Background Frame / Frame Correction Tests (Line-Field OCT)")
    print("=" * 60)
    test_recording_and_subtraction()
    test_profile_roundtrip_and_reset()
    test_continuous_ema()
    test_frame_correction()
    test_config_copy_and_type_change()
    test_unsupported_backend()
    print()
    print("All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
