#include "../include/processor.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>

const ope::Backend TEST_BACKEND = ope::Backend::CPU;

// Regression test: setConfig() with unchanged dimensions must forward processing
// parameter changes to an already initialized backend (not just the curves).

namespace {

std::vector<float> processOneBuffer(ope::Processor& processor, const std::vector<uint16_t>& inputData, int outputSamples) {
	std::vector<float> output(outputSamples);
	std::atomic<bool> done{false};

	int callbackId = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		const float* data = static_cast<const float*>(buf.getDataPointer());
		std::copy(data, data + outputSamples, output.begin());
		done = true;
	});

	auto& inputBuffer = processor.getNextAvailableInputBuffer();
	memcpy(inputBuffer.getDataPointer(), inputData.data(), inputData.size() * sizeof(uint16_t));
	processor.process(inputBuffer);

	while (!done) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	processor.removeOutputCallback(callbackId);
	return output;
}

} // namespace

void testSetConfigPropagatesProcessingFlags() {
	std::cout << "Testing setConfig() propagation of processing flags to initialized backend..." << std::endl;

	const int signalLength = 1024;
	const int ascansPerBscan = 16;
	const int bscansPerBuffer = 1;
	const int samplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	const int outputSamples = samplesPerBuffer / 2;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	processor.enableLogScaling(true);
	processor.initialize();

	// Deterministic input signal
	std::vector<uint16_t> inputData(samplesPerBuffer);
	for (int i = 0; i < samplesPerBuffer; i++) {
		inputData[i] = static_cast<uint16_t>(1000 + 500 * std::sin(i * 0.05));
	}

	std::vector<float> outputLog = processOneBuffer(processor, inputData, outputSamples);

	// Toggle log scaling off via bulk setConfig() with unchanged dimensions
	ope::ProcessorConfiguration config = processor.getConfig();
	config.processingParams.intensity.logScale = false;
	processor.setConfig(config);

	std::vector<float> outputLin = processOneBuffer(processor, inputData, outputSamples);

	// If the flag reached the backend, the outputs must differ
	bool identical = true;
	for (int i = 0; i < outputSamples; i++) {
		if (outputLog[i] != outputLin[i]) {
			identical = false;
			break;
		}
	}

	TEST_ASSERT(!identical, "setConfig() flag change must reach the initialized backend and alter the output");
	std::cout << "  [OK] logScale change via setConfig() reached the backend" << std::endl;
}

int main() {
	std::cout << "=== setConfig() Propagation Test ===" << std::endl;
	try {
		testSetConfigPropagatesProcessingFlags();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
