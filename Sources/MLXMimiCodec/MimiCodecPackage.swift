import Foundation
import AVFoundation
import MLXToolKit
import MLX
import Hub
import MimiCodecEncoder

/// Errors at the Mimi package boundary.
public enum MimiCodecError: Error, Equatable {
    case audioDecodeFailed(String)
}

/// An MLXEngine `audioCodec` package over **Kyutai's Mimi** neural audio codec encoder — encodes
/// 24 kHz mono audio into a `[16, T]` discrete RVQ token grid at 12.5 Hz. A thin conformance
/// wrapper over the standalone `MimiCodecEncoder` engine (mimi-encoder-mlx-swift); all model logic
/// (SEANet conv encoder → causal transformer → stride-2 downsample → split RVQ) lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `MimiCodecConfiguration`, pages
/// weights in with `load()` (downloads the HF snapshot and builds the encoder), drives `run(_:)`,
/// and reclaims with `unload()`. Returns the canonical `AudioCodecResponse` (token grid).
///
/// **Encode direction only.** Decoding tokens back to audio is a separate future capability.
@InferenceActor
public final class MimiCodecPackage: ModelPackage {
    public typealias Configuration = MimiCodecConfiguration

    private static let encoderConfig = MimiEncoderConfiguration.qwen3TTS12Hz

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Mimi weights (Kyutai) are CC-BY-4.0 (permissive, allowlisted). Port code: MIT.
            license: LicenseDeclaration(weightLicense: .ccBy4, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/mimi-encoder-mlx",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // ~191 MB fp32 encoder weights + conv/transformer activations over the clip.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 800_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                AudioCodecContract.descriptor(
                    name: "mimi-encode",
                    summary: "Mimi neural audio codec encoder: encodes audio into a [16, T] discrete RVQ token grid at 12.5 Hz."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var encoder: MimiEncoder?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard encoder == nil else { return }
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: Hub.Repo(id: configuration.repo),
                                         matching: ["encoder.safetensors"])
        let enc = MimiEncoder(config: Self.encoderConfig)
        try enc.loadWeights(from: dir.appendingPathComponent("encoder.safetensors"))
        encoder = enc
    }

    public func unload() async {
        encoder = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let encoder else { throw PackageError.notLoaded }
        guard request.capability == .audioCodec,
              let req = request as? AudioCodecRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Decode to 24 kHz mono float samples (Mimi's expected input).
        let mono = try Self.decodeMono24k(req.audio)
        let codes = encoder.encode(audio: mono).asType(.int32)  // [16, T]
        let numCodebooks = codes.shape[0]
        let rows: [[Int32]] = (0..<numCodebooks).map { codes[$0].asArray(Int32.self) }
        return AudioCodecResponse(codes: rows,
                                  numCodebooks: numCodebooks,
                                  frameRate: Double(Self.encoderConfig.frameRate))
    }

    // MARK: - Audio I/O

    /// Decode a canonical `Audio` (.wav) to a 1-D 24 kHz mono float MLXArray. The Mimi core ships
    /// no audio loader, so this resamples via AVFoundation. Reads from a URL → temp-file round-trip.
    nonisolated static func decodeMono24k(_ audio: Audio) throws -> MLXArray {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: tmp) }
        catch { throw MimiCodecError.audioDecodeFailed("open: \(error.localizedDescription)") }

        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000,
                                         channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: target),
              let srcBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                            frameCapacity: AVAudioFrameCount(file.length))
        else { throw MimiCodecError.audioDecodeFailed("format setup") }

        do { try file.read(into: srcBuf) }
        catch { throw MimiCodecError.audioDecodeFailed("read: \(error.localizedDescription)") }

        let ratio = target.sampleRate / file.processingFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * ratio) + 4096
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCapacity) else {
            throw MimiCodecError.audioDecodeFailed("output buffer")
        }

        var fed = false
        var convError: NSError?
        converter.convert(to: outBuf, error: &convError) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return srcBuf
        }
        if let convError { throw MimiCodecError.audioDecodeFailed("convert: \(convError.localizedDescription)") }

        let n = Int(outBuf.frameLength)
        guard let ch = outBuf.floatChannelData else {
            throw MimiCodecError.audioDecodeFailed("no channel data")
        }
        let samples = Array(UnsafeBufferPointer(start: ch[0], count: n))
        return MLXArray(samples)
    }
}

extension MimiCodecPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MimiCodecPackage.self)
    }
}
