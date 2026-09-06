#include "../include/processor.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <limits>

// Tests for background frame (B-scan) subtraction for line-field OCT.
// Uses linear intensity scaling with passthrough parameters (min=0, max=1, coeff=1, addend=0)
// so that output sample 0 of each A-scan is 2*|residual| for a constant-valued A-scan
// (DC magnitude = |residual| * signalLength, divided by outputAscanLength = signalLength/2).

namespace {

const int SIGNAL_LENGTH = 64;
const int ASCANS_PER_BSCAN = 8;
const int SAMPLES_PER_BSCAN = SIGNAL_LENGTH * ASCANS_PER_BSCAN;

void configurePassthrough(ope::Processor& processor, int bscansPerBuffer) {
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, bscansPerBuffer, ope::DataType::UINT16);
	processor.enableLogScaling(false);
	processor.setGrayscaleRange(0.0f, 1.0f);
	processor.setSignalMultiplicatorAndAddend(1.0f, 0.0f);
}

// Builds a buffer where every sample of B-scan b has the constant value bscanValues[b]
std::vector<uint16_t> makeConstantBscans(const std::vector<uint16_t>& bscanValues) {
	std::vector<uint16_t> data(bscanValues.size() * SAMPLES_PER_BSCAN);
	for (size_t b = 0; b < bscanValues.size(); ++b) {
		std::fill(data.begin() + b * SAMPLES_PER_BSCAN,
				  data.begin() + (b + 1) * SAMPLES_PER_BSCAN,
				  bscanValues[b]);
	}
	return data;
}

// Processes the given buffers in order and returns the outputs in delivery order
std::vector<std::vector<float>> processBuffers(ope::Processor& processor,
											   const std::vector<std::vector<uint16_t>>& inputs,
											   int bscansPerBuffer) {
	size_t outputSamples = (SAMPLES_PER_BSCAN * bscansPerBuffer) / 2;
	std::vector<std::vector<float>> outputs;
	std::mutex outputsMutex;
	std::atomic<int> received{0};

	int callbackId = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		const float* data = static_cast<const float*>(buf.getDataPointer());
		std::lock_guard<std::mutex> lock(outputsMutex);
		outputs.emplace_back(data, data + outputSamples);
		received++;
	});

	for (const auto& input : inputs) {
		auto& buffer = processor.getNextAvailableInputBuffer();
		memcpy(buffer.getDataPointer(), input.data(), input.size() * sizeof(uint16_t));
		processor.process(buffer);
	}

	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
	while (received < static_cast<int>(inputs.size())) {
		TEST_ASSERT(std::chrono::steady_clock::now() < deadline, "Timed out waiting for output delivery");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	processor.removeOutputCallback(callbackId);
	return outputs;
}

bool nearlyEqual(float a, float b, float tolerance) {
	return std::abs(a - b) <= tolerance;
}

} // namespace

// Recording: the recorded profile must be the exact average of the recorded B-scans,
// including a recording that spans buffers and ends partway through a buffer
void testRecording(ope::Backend backend) {
	std::cout << "  Recording (spans buffers, ends mid-buffer)..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 2);
	processor.setBackgroundFrameBscansToAverage(3);
	processor.initialize();

	processor.requestBackgroundFrameRecording();

	// Two buffers with 2 B-scans each; only the first 3 B-scans (100, 200, 600) may contribute
	processBuffers(processor, {makeConstantBscans({100, 200}), makeConstantBscans({600, 9999})}, 2);

	TEST_ASSERT(processor.hasBackgroundFrameProfile(), "Profile must exist after recording completes");
	std::vector<float> profile = processor.getBackgroundFrameProfile();
	TEST_ASSERT(profile.size() == static_cast<size_t>(SAMPLES_PER_BSCAN), "Profile size must be samplesPerBscan");
	for (float value : profile) {
		TEST_ASSERT(nearlyEqual(value, 300.0f, 0.01f), "Recorded profile must be the average of the first 3 B-scans (300)");
	}
}

// Static subtraction: processing (delta + background) with subtraction enabled must equal
// processing (delta) alone
void testStaticSubtraction(ope::Backend backend) {
	std::cout << "  Static subtraction vs delta-only reference..." << std::endl;

	std::vector<float> background(SAMPLES_PER_BSCAN, 500.0f);
	std::vector<uint16_t> deltaOnly(SAMPLES_PER_BSCAN);
	std::vector<uint16_t> deltaPlusBackground(SAMPLES_PER_BSCAN);
	for (int i = 0; i < SAMPLES_PER_BSCAN; ++i) {
		uint16_t delta = static_cast<uint16_t>(200 + 100 * std::sin(i * 0.2));
		deltaOnly[i] = delta;
		deltaPlusBackground[i] = static_cast<uint16_t>(delta + 500);
	}

	ope::Processor reference(backend);
	configurePassthrough(reference, 1);
	reference.initialize();
	auto referenceOutput = processBuffers(reference, {deltaOnly}, 1);

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	processor.enableBackgroundFrameSubtraction(true);
	auto output = processBuffers(processor, {deltaPlusBackground}, 1);

	for (size_t i = 0; i < output[0].size(); ++i) {
		TEST_ASSERT(nearlyEqual(output[0][i], referenceOutput[0][i], 0.001f),
			"Subtracting the background must reproduce the delta-only output");
	}
}

// Normalization: out = scale*(in-bg)/sqrt(bg) where sqrt(bg) > 1, plain subtraction elsewhere
void testNormalization(ope::Backend backend) {
	std::cout << "  Normalization formula and sqrt(bg) <= 1 passthrough..." << std::endl;

	const float backgroundValue = 400.0f;   // sqrt = 20 > 1 -> normalize branch
	const float inputValue = 1000.0f;
	std::vector<float> background(SAMPLES_PER_BSCAN, backgroundValue);

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	processor.enableBackgroundFrameSubtraction(true);
	processor.enableBackgroundFrameNormalization(true);

	auto output = processBuffers(processor, {makeConstantBscans({static_cast<uint16_t>(inputValue)})}, 1);

	// residual = scale*(in-bg)/sqrt(bg); DC output sample = 2*|residual|
	float scale = std::sqrt(std::pow(2.0f, 16.0f));
	float expectedResidual = scale * (inputValue - backgroundValue) / std::sqrt(backgroundValue);
	float expectedDc = 2.0f * std::abs(expectedResidual);
	for (int a = 0; a < ASCANS_PER_BSCAN; ++a) {
		float dc = output[0][a * (SIGNAL_LENGTH / 2)];
		TEST_ASSERT(nearlyEqual(dc, expectedDc, expectedDc * 0.001f),
			"Normalized subtraction must follow scale*(in-bg)/sqrt(bg)");
	}

	// Passthrough branch: bg values with sqrt(bg) <= 1 must subtract without normalization
	std::vector<float> tinyBackground(SAMPLES_PER_BSCAN, 0.5f);
	ope::Processor passthrough(backend);
	configurePassthrough(passthrough, 1);
	passthrough.initialize();
	passthrough.setBackgroundFrameProfile(tinyBackground.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	passthrough.enableBackgroundFrameSubtraction(true);
	passthrough.enableBackgroundFrameNormalization(true);

	auto passthroughOutput = processBuffers(passthrough, {makeConstantBscans({100})}, 1);
	float expectedPassthroughDc = 2.0f * std::abs(100.0f - 0.5f);
	float dc = passthroughOutput[0][0];
	TEST_ASSERT(nearlyEqual(dc, expectedPassthroughDc, expectedPassthroughDc * 0.001f),
		"sqrt(bg) <= 1 must fall back to plain subtraction");
}

// Smoothing: the applied background must match a host-computed edge-clamped rolling average
void testSmoothing(ope::Backend backend) {
	std::cout << "  Smoothing (edge-clamped rolling average)..." << std::endl;

	const int windowRadius = 4;
	std::vector<float> background(SAMPLES_PER_BSCAN);
	for (int i = 0; i < SAMPLES_PER_BSCAN; ++i) {
		background[i] = 500.0f + 100.0f * std::sin(i * 0.7f);
	}

	// Host reference: smooth each spectrum, then compute the expected DC of (in - smoothedBg)
	const float inputValue = 1000.0f;
	std::vector<float> smoothed(SAMPLES_PER_BSCAN);
	for (int index = 0; index < SAMPLES_PER_BSCAN; ++index) {
		int sampleIndex = index % SIGNAL_LENGTH;
		int firstIndexOfLine = index - sampleIndex;
		int startIdx = std::max(firstIndexOfLine, index - windowRadius);
		int endIdx = std::min(firstIndexOfLine + SIGNAL_LENGTH - 1, index + windowRadius);
		float sum = 0.0f;
		for (int i = startIdx; i <= endIdx; ++i) sum += background[i];
		smoothed[index] = sum / static_cast<float>(endIdx - startIdx + 1);
	}

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	processor.enableBackgroundFrameSubtraction(true);
	processor.setBackgroundFrameSmoothing(true, windowRadius);

	auto output = processBuffers(processor, {makeConstantBscans({static_cast<uint16_t>(inputValue)})}, 1);

	// Expected DC per A-scan: |sum over samples of (in - smoothedBg)| / (signalLength/2)
	for (int a = 0; a < ASCANS_PER_BSCAN; ++a) {
		float residualSum = 0.0f;
		for (int s = 0; s < SIGNAL_LENGTH; ++s) {
			residualSum += inputValue - smoothed[a * SIGNAL_LENGTH + s];
		}
		float expectedDc = std::abs(residualSum) / (SIGNAL_LENGTH / 2);
		float dc = output[0][a * (SIGNAL_LENGTH / 2)];
		TEST_ASSERT(nearlyEqual(dc, expectedDc, std::max(0.01f, expectedDc * 0.001f)),
			"Smoothed subtraction must use the edge-clamped rolling average of the background");
	}
}

// EMA buffer-level semantics: ALL B-scans of a buffer are folded into the background first,
// then the FINAL background is subtracted from the entire buffer.
// bg0=0, alpha=1/2, B-scans 10 and 20 -> bg=12.5, residuals [-2.5, 7.5] (per-scan update
// would give [5, 7.5] instead)
void testEmaBufferSemantics(ope::Backend backend) {
	std::cout << "  EMA buffer-level fold-then-subtract semantics..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 2);
	processor.setBackgroundFrameBscansToAverage(2);  // alpha = 1/2
	processor.enableBackgroundFrameSubtraction(true);
	processor.enableContinuousBackgroundFrameUpdate(true);
	processor.initialize();

	auto output = processBuffers(processor, {makeConstantBscans({10, 20})}, 2);

	int outputAscanLength = SIGNAL_LENGTH / 2;
	float dcBscan0 = output[0][0];
	float dcBscan1 = output[0][ASCANS_PER_BSCAN * outputAscanLength];
	TEST_ASSERT(nearlyEqual(dcBscan0, 2.0f * 2.5f, 0.01f),
		"B-scan 0 residual must be -2.5 (final background subtracted, not per-scan update)");
	TEST_ASSERT(nearlyEqual(dcBscan1, 2.0f * 7.5f, 0.01f),
		"B-scan 1 residual must be 7.5");

	// EMA convergence: after k more B-scans of constant value v the background error
	// decays as (1 - alpha)^k
	int extraBuffers = 4;  // 8 more B-scans
	std::vector<std::vector<uint16_t>> constantBuffers(extraBuffers, makeConstantBscans({100, 100}));
	processBuffers(processor, constantBuffers, 2);

	std::vector<float> profile = processor.getBackgroundFrameProfile();
	float bgAfterFirstBuffer = 12.5f;
	float expectedBg = 100.0f + (bgAfterFirstBuffer - 100.0f) * std::pow(0.5f, extraBuffers * 2);
	TEST_ASSERT(nearlyEqual(profile[0], expectedBg, 0.01f),
		"EMA background must converge with (1-alpha)^k");
}

// The recording target is latched at request time: shrinking the averaging setting
// mid-recording must not corrupt the count or the normalization
void testRecordingTargetLatched(ope::Backend backend) {
	std::cout << "  Recording target latched at request time..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 2);
	processor.setBackgroundFrameBscansToAverage(5);
	processor.initialize();

	processor.requestBackgroundFrameRecording();
	processBuffers(processor, {makeConstantBscans({100, 200})}, 2);

	// Shrinking the setting mid-recording must not finalize early
	processor.setBackgroundFrameBscansToAverage(1);
	TEST_ASSERT(!processor.hasBackgroundFrameProfile(), "Recording must continue to the latched target of 5");

	processBuffers(processor, {makeConstantBscans({300, 400}), makeConstantBscans({500, 9999})}, 2);

	TEST_ASSERT(processor.hasBackgroundFrameProfile(), "Recording must complete at the latched target");
	std::vector<float> profile = processor.getBackgroundFrameProfile();
	for (float value : profile) {
		TEST_ASSERT(nearlyEqual(value, 300.0f, 0.01f),
			"Recorded profile must average exactly the 5 B-scans of the latched target (300)");
	}
}

// Recording takes precedence over continuous update, identically on all backends:
// while recording, EMA is suppressed; on the next buffer it resumes seeded by the
// freshly recorded frame
void testRecordingSuppressesEma(ope::Backend backend) {
	std::cout << "  Recording suppresses EMA, which resumes from the recorded frame..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 2);
	processor.setBackgroundFrameBscansToAverage(2);  // recording target 2, EMA alpha = 1/2
	processor.enableBackgroundFrameSubtraction(true);
	processor.enableContinuousBackgroundFrameUpdate(true);
	processor.initialize();

	processor.requestBackgroundFrameRecording();
	processBuffers(processor, {makeConstantBscans({10, 20})}, 2);

	// Recording of B-scans 10, 20 finalizes to exactly 15; EMA must not have run on top
	std::vector<float> profile = processor.getBackgroundFrameProfile();
	TEST_ASSERT(nearlyEqual(profile[0], 15.0f, 0.01f),
		"While recording, EMA must be suppressed (background must be exactly the recorded 15)");

	// Next buffer: EMA resumes seeded by the recorded frame: 15 -> 25 -> 30
	processBuffers(processor, {makeConstantBscans({35, 35})}, 2);
	profile = processor.getBackgroundFrameProfile();
	TEST_ASSERT(nearlyEqual(profile[0], 30.0f, 0.01f),
		"EMA must resume from the recorded frame on the next buffer");
}

// Raw profile file round trip and reset
void testSaveLoadReset(ope::Backend backend) {
	std::cout << "  Raw profile save/load round trip and reset..." << std::endl;

	std::vector<float> background(SAMPLES_PER_BSCAN);
	for (int i = 0; i < SAMPLES_PER_BSCAN; ++i) {
		background[i] = static_cast<float>(i % 1000);
	}

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);

	const std::string filepath = "test_background_frame_profile.raw";
	processor.saveBackgroundFrameProfileToFile(filepath);

	processor.resetBackgroundFrame();
	TEST_ASSERT(processor.getBackgroundFrameProfile().empty(), "Reset must clear the profile");

	processor.loadBackgroundFrameProfileFromFile(filepath);
	std::vector<float> loaded = processor.getBackgroundFrameProfile();
	TEST_ASSERT(loaded.size() == background.size(), "Loaded profile size must match");
	for (size_t i = 0; i < background.size(); ++i) {
		TEST_ASSERT(loaded[i] == background[i], "Loaded profile values must match exactly");
	}
	std::remove(filepath.c_str());

	// Invalid profiles must be rejected without changing state
	std::vector<float> negativeProfile(SAMPLES_PER_BSCAN, -1.0f);
	bool threw = false;
	try {
		processor.setBackgroundFrameProfile(negativeProfile.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	TEST_ASSERT(threw, "Negative profile values must be rejected");
	TEST_ASSERT(processor.getBackgroundFrameProfile() == loaded, "Rejected profile must not change state");
}

// Output delivery must continue after a mid-session reinitialization: the backends'
// ordered callback delivery restarts at buffer ID 0, so the processor must restart
// its buffer IDs too
void testReinitializeKeepsDelivering(ope::Backend backend) {
	std::cout << "  Output delivery continues after reinitialization..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();

	std::atomic<int> received{0};
	processor.addOutputCallback([&](const ope::IOBuffer&) { received++; });

	auto waitFor = [&](int count) {
		auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
		while (received < count) {
			TEST_ASSERT(std::chrono::steady_clock::now() < deadline,
				"Timed out waiting for output delivery after reinitialization");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	};

	std::vector<uint16_t> data = makeConstantBscans({100});
	{
		auto& buffer = processor.getNextAvailableInputBuffer();
		memcpy(buffer.getDataPointer(), data.data(), data.size() * sizeof(uint16_t));
		processor.process(buffer);
	}
	waitFor(1);

	// Dimension change triggers a reinitialization mid-session
	ope::ProcessorConfiguration config = processor.getConfig();
	config.dataParams.bscansPerBuffer = 2;
	processor.setConfig(config);

	std::vector<uint16_t> data2 = makeConstantBscans({100, 100});
	{
		auto& buffer = processor.getNextAvailableInputBuffer();
		memcpy(buffer.getDataPointer(), data2.data(), data2.size() * sizeof(uint16_t));
		processor.process(buffer);
	}
	waitFor(2);
}

// setInputParameters() must preserve geometry-compatible live profiles across its
// eager reinitialization, like setConfig() does
void testSetInputParametersPreservesFrame(ope::Backend backend) {
	std::cout << "  setInputParameters() preserves a compatible live frame..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.setBackgroundFrameBscansToAverage(2);  // alpha = 1/2
	processor.enableBackgroundFrameSubtraction(true);
	processor.enableContinuousBackgroundFrameUpdate(true);
	processor.initialize();

	std::vector<uint16_t> data = makeConstantBscans({100});
	processBuffers(processor, {data}, 1);  // live background: 0 -> 50

	// Only bscansPerBuffer changes: the frame geometry stays valid
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, 2, ope::DataType::UINT16);

	std::vector<float> profile = processor.getBackgroundFrameProfile();
	TEST_ASSERT(!profile.empty() && nearlyEqual(profile[0], 50.0f, 0.01f),
		"A bscansPerBuffer-only change must preserve the live background");
}

// setConfig() profile semantics: an explicitly replaced profile must reach the backend,
// an unchanged profile must not overwrite a live EMA-advanced calibration, and a
// bscansPerBuffer-only change must preserve the live frame across reinitialization
void testSetConfigProfileHandling(ope::Backend backend) {
	std::cout << "  setConfig() profile replacement / preservation..." << std::endl;

	// Replacement: frame 100 live, config carries frame 300 -> constant 1000 gives DC 1400
	{
		ope::Processor processor(backend);
		configurePassthrough(processor, 1);
		processor.initialize();
		std::vector<float> oldFrame(SAMPLES_PER_BSCAN, 100.0f);
		processor.setBackgroundFrameProfile(oldFrame.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
		processor.enableBackgroundFrameSubtraction(true);

		ope::ProcessorConfiguration config = processor.getConfig();
		config.setBackgroundFrameProfile(std::vector<float>(SAMPLES_PER_BSCAN, 300.0f), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
		processor.setConfig(config);

		auto output = processBuffers(processor, {makeConstantBscans({1000})}, 1);
		TEST_ASSERT(nearlyEqual(output[0][0], 2.0f * (1000.0f - 300.0f), 0.01f),
			"A replaced profile in setConfig() must reach the backend");
	}

	// Preservation: an unrelated settings round trip must not reset a live EMA background
	{
		ope::Processor processor(backend);
		configurePassthrough(processor, 1);
		processor.setBackgroundFrameBscansToAverage(2);  // alpha = 1/2
		processor.enableBackgroundFrameSubtraction(true);
		processor.enableContinuousBackgroundFrameUpdate(true);
		processor.initialize();

		std::vector<uint16_t> data = makeConstantBscans({100});
		processBuffers(processor, {data, data}, 1);  // live background: 0 -> 50 -> 75

		ope::ProcessorConfiguration config = processor.getConfig();
		config.processingParams.dcRemoval.windowSize = 32;  // unrelated change
		processor.setConfig(config);
		processor.setConfig(config);  // applying the SAME object again must not reset it either

		std::vector<float> profile = processor.getBackgroundFrameProfile();
		TEST_ASSERT(nearlyEqual(profile[0], 75.0f, 0.01f),
			"Repeated unrelated setConfig() calls must preserve the live EMA background");

		processBuffers(processor, {data}, 1);  // EMA must continue from 75, not restart
		profile = processor.getBackgroundFrameProfile();
		TEST_ASSERT(nearlyEqual(profile[0], 87.5f, 0.01f),
			"EMA must continue from the preserved background after the round trip");
	}

	// Geometry-compatible reinitialization: bscansPerBuffer change keeps the frame
	{
		ope::Processor processor(backend);
		configurePassthrough(processor, 1);
		processor.initialize();
		std::vector<float> frame(SAMPLES_PER_BSCAN, 100.0f);
		processor.setBackgroundFrameProfile(frame.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);

		ope::ProcessorConfiguration config = processor.getConfig();
		config.dataParams.bscansPerBuffer = 2;
		processor.setConfig(config);

		std::vector<float> preserved = processor.getBackgroundFrameProfile();
		TEST_ASSERT(preserved.size() == static_cast<size_t>(SAMPLES_PER_BSCAN),
			"Frame must survive a bscansPerBuffer-only reinitialization");
		TEST_ASSERT(nearlyEqual(preserved[0], 100.0f, 0.0001f), "Preserved frame values must match");
	}
}

// Invalid configurations and profile files must be rejected without partial state
void testValidationRejection() {
	std::cout << "  Validation: invalid setConfig() and NaN profile file rejected..." << std::endl;

	ope::Processor processor(ope::Backend::CPU);
	configurePassthrough(processor, 1);
	processor.initialize();

	// bscansToAverage = 0 must be rejected before any state changes
	ope::ProcessorConfiguration config = processor.getConfig();
	config.processingParams.backgroundFrame.bscansToAverage = 0;
	bool threw = false;
	try {
		processor.setConfig(config);
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	TEST_ASSERT(threw, "setConfig() with bscansToAverage = 0 must throw");
	TEST_ASSERT(processor.getConfig().processingParams.backgroundFrame.bscansToAverage != 0,
		"Rejected setConfig() must leave the configuration unchanged");

	// A correctly sized raw file containing NaN must be rejected, keeping the old profile
	std::vector<float> validFrame(SAMPLES_PER_BSCAN, 42.0f);
	processor.setBackgroundFrameProfile(validFrame.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);

	const std::string filepath = "test_background_frame_nan.raw";
	{
		std::vector<float> nanFrame(SAMPLES_PER_BSCAN, std::numeric_limits<float>::quiet_NaN());
		std::ofstream file(filepath, std::ios::binary);
		file.write(reinterpret_cast<const char*>(nanFrame.data()), nanFrame.size() * sizeof(float));
	}

	bool loadThrew = false;
	try {
		processor.loadBackgroundFrameProfileFromFile(filepath);
	} catch (const std::exception&) {
		loadThrew = true;
	}
	std::remove(filepath.c_str());
	TEST_ASSERT(loadThrew, "Loading a NaN profile file must throw");
	std::vector<float> profile = processor.getBackgroundFrameProfile();
	TEST_ASSERT(!profile.empty() && profile[0] == 42.0f,
		"A rejected profile file must leave the previous profile untouched");
}

// A dimension change must invalidate the frame immediately (before the lazy reinit runs)
// and after it - including changes that keep the element count identical (equal
// samplesPerBscan, incompatible layout)
void testDimensionChangeInvalidatesFrame(ope::Backend backend) {
	std::cout << "  Dimension change invalidates the frame (equal element count, before/after reinit)..." << std::endl;

	ope::Processor processor(backend);
	configurePassthrough(processor, 1);
	processor.initialize();
	std::vector<float> background(SAMPLES_PER_BSCAN, 100.0f);
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
	TEST_ASSERT(processor.hasBackgroundFrameProfile(), "Profile must exist before the change");

	// Equal element count, different layout: 2*signalLength x ascans/2
	processor.setInputParameters(SIGNAL_LENGTH * 2, ASCANS_PER_BSCAN / 2, 1, ope::DataType::UINT16);
	TEST_ASSERT(!processor.hasBackgroundFrameProfile(), "Stale frame must not be visible before the lazy reinit");
	TEST_ASSERT(processor.getBackgroundFrameProfile().empty(), "Stale frame must not be returned before the lazy reinit");

	processor.initialize();  // runs the pending reinitialization
	TEST_ASSERT(!processor.hasBackgroundFrameProfile(), "Stale frame must not survive reinitialization");
}

// The profile must survive a backend switch (backend -> config -> new backend)
void testBackendSwitchTransfer() {
	std::cout << "  Profile transfer on backend switch (CPU -> CUDA)..." << std::endl;
	if (!ope::BackendUtils::isCudaAvailable()) {
		std::cout << "    [SKIPPED] no CUDA device available" << std::endl;
		return;
	}

	std::vector<float> background(SAMPLES_PER_BSCAN);
	for (int i = 0; i < SAMPLES_PER_BSCAN; ++i) {
		background[i] = static_cast<float>(100 + i);
	}

	ope::Processor processor(ope::Backend::CPU);
	configurePassthrough(processor, 1);
	processor.initialize();
	processor.setBackgroundFrameProfile(background.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);

	processor.setBackend(ope::Backend::CUDA);

	std::vector<float> transferred = processor.getBackgroundFrameProfile();
	TEST_ASSERT(transferred.size() == background.size(), "Transferred profile size must match");
	for (size_t i = 0; i < background.size(); ++i) {
		TEST_ASSERT(nearlyEqual(transferred[i], background[i], 0.0001f), "Transferred profile values must match");
	}
}

// Enabling on OpenCL/Vulkan must throw without changing configuration
void testUnsupportedBackendRejection() {
	std::cout << "  Unsupported backend rejection (OpenCL)..." << std::endl;
	if (!ope::BackendUtils::isOpenCLAvailable()) {
		std::cout << "    [SKIPPED] no OpenCL runtime available" << std::endl;
		return;
	}

	ope::Processor processor(ope::Backend::OPENCL);
	bool threw = false;
	try {
		processor.enableBackgroundFrameSubtraction(true);
	} catch (const std::runtime_error&) {
		threw = true;
	}
	TEST_ASSERT(threw, "Enabling background frame subtraction on OpenCL must throw");
	TEST_ASSERT(!processor.getConfig().processingParams.backgroundFrame.enabled,
		"Rejected enable must not change the configuration");

	bool threwCorrection = false;
	try {
		processor.enablePostFftFrameCorrection(true);
	} catch (const std::runtime_error&) {
		threwCorrection = true;
	}
	TEST_ASSERT(threwCorrection, "Enabling frame correction on OpenCL must throw");
}

// Reset must work on passive backends, and a rejected backend switch must leave
// backend and stored backend configuration consistent
void testResetAndRejectedSwitchConsistency() {
	std::cout << "  Reset on passive backends and rejected setBackendConfig()..." << std::endl;

	ope::Processor processor(ope::Backend::CPU);
	configurePassthrough(processor, 1);
	processor.enableBackgroundFrameSubtraction(true);
	bool threw = false;
	try {
		processor.setBackendConfig(ope::OpenCLConfig());
	} catch (const std::runtime_error&) {
		threw = true;
	}
	TEST_ASSERT(threw, "setBackendConfig() to an unsupported backend must throw");
	TEST_ASSERT(processor.getBackend() == ope::Backend::CPU, "Backend must remain CPU");
	auto backendConfig = processor.getBackendConfig();
	TEST_ASSERT(backendConfig && backendConfig->getBackendType() == ope::Backend::CPU,
		"Stored backend configuration must match the actual backend");

	// Passive profile storage and reset work without processing support
	if (ope::BackendUtils::isOpenCLAvailable()) {
		ope::Processor passive(ope::Backend::OPENCL);
		passive.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, 1, ope::DataType::UINT16);
		std::vector<float> frame(SAMPLES_PER_BSCAN, 5.0f);
		passive.setBackgroundFrameProfile(frame.data(), SIGNAL_LENGTH, ASCANS_PER_BSCAN);
		TEST_ASSERT(passive.hasBackgroundFrameProfile(), "Passive profile must be stored");
		passive.resetBackgroundFrame();
		TEST_ASSERT(!passive.hasBackgroundFrameProfile(), "Reset must clear the passive profile");
	} else {
		std::cout << "    [SKIPPED] passive reset: no OpenCL runtime available" << std::endl;
	}
}

// After a failed switch the stored backend configuration must describe the backend
// instance that actually exists - not an invented default
void testFailedSwitchKeepsAccurateMetadata() {
	std::cout << "  Failed device switch keeps accurate backend metadata..." << std::endl;
	if (!ope::BackendUtils::isCudaAvailable()) {
		std::cout << "    [SKIPPED] no CUDA device available" << std::endl;
		return;
	}

	ope::Processor processor(ope::Backend::CPU);
	configurePassthrough(processor, 1);
	processor.initialize();

	ope::CudaConfig cudaConfig;
	cudaConfig.deviceId = 999;
	bool threw = false;
	try {
		processor.setBackendConfig(cudaConfig);
	} catch (const std::exception&) {
		threw = true;
	}
	TEST_ASSERT(threw, "Switching to an invalid CUDA device must throw");

	auto backendConfig = processor.getBackendConfig();
	TEST_ASSERT(backendConfig != nullptr, "Backend configuration must exist after the failure");
	if (backendConfig->getBackendType() == ope::Backend::CUDA) {
		// The CUDA backend instance was created with device 999; the stored configuration
		// must report that, not an invented default
		TEST_ASSERT(static_cast<ope::CudaConfig*>(backendConfig.get())->deviceId == 999,
			"Stored backend configuration must describe the actual instance");
	} else {
		TEST_ASSERT(backendConfig->getBackendType() == processor.getBackend(),
			"Stored backend configuration must match the actual backend");
	}
}

// CUDA multi-stream determinism: with continuous EMA and mid-run transitions the CUDA
// output sequence (3 streams by default) must match the strictly serial CPU backend
void testCudaSequenceMatchesCpu() {
	std::cout << "  CUDA multi-stream sequence vs serial CPU (continuous EMA + transitions)..." << std::endl;

	const int numBuffers = 30;
	std::vector<std::vector<uint16_t>> inputs;
	for (int n = 0; n < numBuffers; ++n) {
		inputs.push_back(makeConstantBscans({static_cast<uint16_t>(100 + 40 * (n % 5)),
											 static_cast<uint16_t>(300 + 25 * (n % 7))}));
	}

	auto runSequence = [&](ope::Backend backend) {
		ope::Processor processor(backend);
		configurePassthrough(processor, 2);
		processor.setBackgroundFrameBscansToAverage(4);
		processor.enableBackgroundFrameSubtraction(true);
		processor.enableContinuousBackgroundFrameUpdate(true);
		processor.initialize();

		std::vector<std::vector<uint16_t>> firstPart(inputs.begin(), inputs.begin() + 10);
		std::vector<std::vector<uint16_t>> secondPart(inputs.begin() + 10, inputs.begin() + 20);
		std::vector<std::vector<uint16_t>> thirdPart(inputs.begin() + 20, inputs.end());

		auto outputs = processBuffers(processor, firstPart, 2);

		// Transition 1: enable smoothing mid-run
		processor.setBackgroundFrameSmoothing(true, 3);
		auto outputs2 = processBuffers(processor, secondPart, 2);

		// Transition 2: back to static subtraction mid-run
		processor.enableContinuousBackgroundFrameUpdate(false);
		auto outputs3 = processBuffers(processor, thirdPart, 2);

		outputs.insert(outputs.end(), outputs2.begin(), outputs2.end());
		outputs.insert(outputs.end(), outputs3.begin(), outputs3.end());
		return outputs;
	};

	if (!ope::BackendUtils::isCudaAvailable()) {
		std::cout << "    [SKIPPED] no CUDA device available" << std::endl;
		return;
	}

	std::vector<std::vector<float>> cpuOutputs = runSequence(ope::Backend::CPU);
	std::vector<std::vector<float>> cudaOutputs = runSequence(ope::Backend::CUDA);

	TEST_ASSERT(cpuOutputs.size() == cudaOutputs.size(), "Both backends must deliver all buffers");
	for (size_t n = 0; n < cpuOutputs.size(); ++n) {
		for (size_t i = 0; i < cpuOutputs[n].size(); ++i) {
			TEST_ASSERT(nearlyEqual(cpuOutputs[n][i], cudaOutputs[n][i], 0.05f),
				"CUDA multi-stream output must match the serial CPU reference (buffer " +
				std::to_string(n) + ", sample " + std::to_string(i) + ")");
		}
	}
}

void runBackendSuite(ope::Backend backend, const char* name) {
	std::cout << "\n=== Backend: " << name << " ===" << std::endl;
	testRecording(backend);
	testStaticSubtraction(backend);
	testNormalization(backend);
	testSmoothing(backend);
	testEmaBufferSemantics(backend);
	testRecordingTargetLatched(backend);
	testRecordingSuppressesEma(backend);
	testSaveLoadReset(backend);
	testDimensionChangeInvalidatesFrame(backend);
	testSetConfigProfileHandling(backend);
	testReinitializeKeepsDelivering(backend);
	testSetInputParametersPreservesFrame(backend);
}

int main() {
	std::cout << "=== Background Frame Subtraction Tests (Line-Field OCT) ===" << std::endl;
	try {
		runBackendSuite(ope::Backend::CPU, "CPU");

		// Availability is decided by BackendUtils, not by catching exceptions:
		// once a backend is available, every failure inside the suite fails the test
		if (ope::BackendUtils::isCudaAvailable()) {
			runBackendSuite(ope::Backend::CUDA, "CUDA");
		} else {
			std::cout << "  [SKIPPED] CUDA suite: no CUDA device available" << std::endl;
		}

		std::cout << "\n=== Cross-backend ===" << std::endl;
		testValidationRejection();
		testBackendSwitchTransfer();
		testUnsupportedBackendRejection();
		testResetAndRejectedSwitchConsistency();
		testFailedSwitchKeepsAccurateMetadata();
		testCudaSequenceMatchesCpu();

		std::cout << "\nAll background frame tests passed" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
