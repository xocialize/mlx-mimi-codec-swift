import Foundation
import MLXToolKit

/// Init-time configuration for `MimiCodecPackage` (C9): which published encoder checkpoint to load.
public struct MimiCodecConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo holding `encoder.safetensors`.
    public var repo: String
    /// Explicit weights directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/mimi-encoder-mlx",
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension MimiCodecConfiguration: WeightSourcing {
    /// The single checkpoint `load()` reads into the `MimiEncoder`.
    static let weightsFile = "encoder.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, matching: [Self.weightsFile])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        // Explicit local directory first (dev escape hatch), then the ModelStore layout.
        if let dir = modelDirectory,
           fm.fileExists(atPath: dir.appending(path: Self.weightsFile).path) {
            return []
        }
        if let dir = ModelStore(root: storeRoot).directory(for: repo),
           fm.fileExists(atPath: dir.appending(path: Self.weightsFile).path) {
            return []
        }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> MimiCodecConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension MimiCodecConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved checkpoint path; the prewarmer skips it when absent (first launch).
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return [dir.appending(path: Self.weightsFile)]
    }
}
