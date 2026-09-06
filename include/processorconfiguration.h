#ifndef OPE_PROCESSORCONFIGURATION_H
#define OPE_PROCESSORCONFIGURATION_H

#include <string>
#include <vector>
#include <map>
#include "types.h"
#include "export.h"


namespace ope {

enum class Backend {
	CUDA,
	CPU,
	OPENCL,
	VULKAN
};

enum class InterpolationMethod {
	LINEAR = 0,
	CUBIC = 1,
	LANCZOS = 2
};

enum class WindowType {
	HANN = 0,
	GAUSS = 1,
	SINE = 2,
	LANCZOS = 3,
	RECTANGULAR = 4,
	FLAT_TOP = 5
};

class OPE_API ProcessorConfiguration {
public:
	// === DATA PARAMETERS ===
	struct OPE_API DataParameters {
		int signalLength = 1024;
		int ascansPerBscan = 512;
		int bscansPerBuffer = 1;
		int buffersPerVolume = 1;
		DataType inputDataType = DataType::UINT16;
		DataType outputDataType = DataType::FLOAT32;

		// Computed properties
		int samplesPerBuffer() const {
			return signalLength * ascansPerBscan * bscansPerBuffer;
		}
		int outputSignalLength() const {
			return signalLength / 2;
		}
		int getBitDepth() const {
			return getDataTypeBitDepth(inputDataType);
		}
		int getBytesPerSample() const {
			return getDataTypeByteSize(inputDataType);
		}
		int getOutputBytesPerSample() const {
			return getDataTypeByteSize(outputDataType);
		}
	} dataParams;

	// === PROCESSING PARAMETERS ===
	struct OPE_API ProcessingParameters {

		// Input preprocessing
		struct OPE_API Input {
			bool bitshift = false;
			// todo: maybe add zero-padding and/or upsampling options 
		} input;

		// DC removal (ues rolling average with specified window size)
		struct OPE_API DCRemoval {
			bool enabled = false;
			int windowSize = 64;
		} dcRemoval;

		// k-linearization / resampling
		struct OPE_API Resampling {
			bool enabled = false;
			InterpolationMethod method = InterpolationMethod::LINEAR;
			float coefficients[4] = {0.0f, 1.0f, 0.0f, 0.0f};
			bool useCustomLut = false;
		} resampling;

		// Windowing / apodization
		struct OPE_API Windowing {
			bool enabled = false;
			WindowType type = WindowType::HANN;
			float centerPosition = 0.5f;
			float fillFactor = 0.95f;
			bool useCustomFunction = false;
		} windowing;

		// Dispersion compensation
		struct OPE_API Dispersion {
			bool enabled = false;
			float coefficients[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float factor = 1.0f; //todo: remove? this was initially intended to easily switch dispersion correction between positive +1.0 and negative -1.0 side (complex conjugate artifact). but with the non-optional truncation step this seems unnecessary.
			bool useCustomPhase = false;
		} dispersion;

		// Fixed pattern noise removal
		struct OPE_API FixedPatternNoise {
			bool enabled = false;
			int bscanAverageCount = 1;
			bool continuous = false;
			bool useCustomProfile = false;
		} fixedPatternNoise;

		// Background subtraction (post-FFT)
		struct OPE_API Background {
			bool enabled = false;
			float weight = 1.0f;
			float offset = 0.0f;
			bool useCustomProfile = false;
		} background;

		// Background frame (B-scan) subtraction before FFT (line-field OCT)
		struct OPE_API BackgroundFrame {
			bool enabled = false;
			bool normalize = false;              // subtract only vs. subtract + divide by sqrt(background)
			int bscansToAverage = 10;            // recording average count; EMA alpha = 1/bscansToAverage
			bool continuousUpdate = false;       // EMA update from live data instead of static recorded frame
			bool smoothSpectra = false;          // rolling-average smoothing of background spectra before use
			int smoothingWindowRadius = 16;      // total window = 2*radius+1
			bool useCustomProfile = false;
		} backgroundFrame;

		// Post-FFT frame correction: divide each A-scan by sqrt of its pre-subtraction spectral average (line-field OCT)
		struct OPE_API FrameCorrection {
			bool enabled = false;
		} frameCorrection;

		// Intensity mapping
		struct OPE_API Intensity {
			bool logScale = true;
			float rangeMin = 30.0f;     // Min dB value
			float rangeMax = 100.0f;     // Max dB value
			float preScale = 1.0f;       // Applied before log //todo: remove? rangeMin and rangeMax should be enough
			float postOffset = 0.0f;     // Added after scaling //todo: remove? rangeMin and rangeMax should be enough
		} intensity;

		// Geometry corrections
		struct OPE_API Geometry {
			bool alternatingBscanFlip = false;  // Flip every second B-scan (for bidirectional scanning)
			bool sinusoidalCorrection = false;  // Correct for sinusoidal scan pattern
		} geometry;

	} processingParams;

	// Constructor and Rule of 5
	ProcessorConfiguration();
	~ProcessorConfiguration();
	ProcessorConfiguration(const ProcessorConfiguration& other);
	ProcessorConfiguration(ProcessorConfiguration&& other) noexcept;
	ProcessorConfiguration& operator=(const ProcessorConfiguration& other);
	ProcessorConfiguration& operator=(ProcessorConfiguration&& other) noexcept;

	// === CUSTOM DATA MANAGEMENT ===
	// Set custom curves (automatically adjusts to signalLength)
	void setResamplingLut(const std::vector<float>& data);
	void setWindowFunction(const std::vector<float>& data);
	void setDispersionPhase(const std::vector<float>& data);
	void setBackgroundProfile(const std::vector<float>& data);
	void setFixedPatternNoiseProfile(const std::vector<float>& complexPairs);

	// Background frame profile (signalLength x ascansPerBscan floats, row = one spectrum).
	// Both dimensions are stored because equal element counts do not imply equal layout.
	// The profile is cleared (not resampled) when dimensions no longer match dataParams.
	void setBackgroundFrameProfile(const std::vector<float>& data, int samplesPerLine, int ascansPerBscan);
	std::vector<float> getBackgroundFrameProfile() const;
	int getBackgroundFrameSamplesPerLine() const;
	int getBackgroundFrameAscansPerBscan() const;

	// Get custom curves (returns adjusted data, empty if not set)
	std::vector<float> getResamplingLut() const;
	std::vector<float> getWindowFunction() const;
	std::vector<float> getDispersionPhase() const;
	std::vector<float> getBackgroundProfile() const;
	std::vector<float> getFixedPatternNoiseProfile() const;

	// Generate curves from parameters
	std::vector<float> generateResamplingLut() const;
	std::vector<float> generateWindowFunction() const;
	std::vector<float> generateDispersionPhase() const;

	// Clear custom data
	void clearResamplingLut();
	void clearWindowFunction();
	void clearDispersionPhase();
	void clearBackgroundProfile();
	void clearFixedPatternNoiseProfile();
	void clearBackgroundFrameProfile();

	// === FILE I/O ===
	enum class LoadMode {
		OVERWRITE_ALL,       // Replace everything including custom data
		PARAMETERS_ONLY,     // Keep current custom data, load only parameters
		MERGE_IF_MISSING     // Load custom data only if current is empty
	};

	enum class SaveMode {
		PARAMETERS_ONLY,     // Save only parameters to INI
		COMPLETE            // Save parameters + custom data in INI
	};
	// Note: the background frame profile is never embedded in the INI (COMPLETE covers the
	// 1D curves/profiles only). Frames are persisted separately as raw float32 binary via
	// saveBackgroundFrameProfileToFile() / loadBackgroundFrameProfileFromFile()

	bool saveToFile(const std::string& filepath, SaveMode mode = SaveMode::COMPLETE) const;
	bool loadFromFile(const std::string& filepath, LoadMode mode = LoadMode::OVERWRITE_ALL);

	// CSV export/import for individual curves
	bool saveResamplingLutToFile(const std::string& filepath) const;
	bool loadResamplingLutFromFile(const std::string& filepath);
	bool saveWindowFunctionToFile(const std::string& filepath) const;
	bool loadWindowFunctionFromFile(const std::string& filepath);
	bool saveDispersionPhaseToFile(const std::string& filepath) const;
	bool loadDispersionPhaseFromFile(const std::string& filepath);
	bool saveBackgroundProfileToFile(const std::string& filepath) const;
	bool loadBackgroundProfileFromFile(const std::string& filepath);
	bool saveFixedPatternNoiseProfileToFile(const std::string& filepath) const;
	bool loadFixedPatternNoiseProfileFromFile(const std::string& filepath);
	bool saveBackgroundFrameProfileToFile(const std::string& filepath) const;
	bool loadBackgroundFrameProfileFromFile(const std::string& filepath);

	// Validation
	bool validate() const;

	// === UTILITY METHODS ===
	// Check if custom curves are set
	bool hasCustomResamplingCurve() const;
	bool hasCustomWindowCurve() const;
	bool hasCustomDispersionCurve() const;
	bool hasCustomPostProcessBackgroundProfile() const;
	bool hasCustomFixedPatternNoiseProfile() const;
	bool hasCustomBackgroundFrameProfile() const;

	void adjustAllCustomCurves();

private:
	void ioFields(std::map<std::string, std::string>& m, bool saving, SaveMode saveMode, LoadMode loadMode);

	struct Impl;
	Impl* impl;
};

} // namespace ope

#endif // OPE_PROCESSORCONFIGURATION_H