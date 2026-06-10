import Foundation
import MLXToolKit

/// Init-time configuration for `MimiCodecPackage` (C9): which published encoder checkpoint to load.
public struct MimiCodecConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo holding `encoder.safetensors`.
    public var repo: String
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/mimi-encoder-mlx",
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}
