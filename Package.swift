// swift-tools-version: 6.2
import PackageDescription

// mlx-mimi-codec-swift — the MLXEngine `audioCodec` package over Kyutai's Mimi encoder.
// A thin conformance layer over the standalone inference engine mimi-encoder-mlx-swift (product
// `MimiCodecEncoder`), same pattern as the other MLXEngine audio packages. The engine contract
// (MLXToolKit) is a local-path dep for in-workspace dev; the Mimi core is pinned to a tagged
// release. Encode direction only. Module/product is `MLXMimiCodec`.
let package = Package(
    name: "mlx-mimi-codec-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXMimiCodec", targets: ["MLXMimiCodec"]),
    ],
    dependencies: [
        // Bumped to 0.23.0 for the WeightSourcing auto-materialization contract (types ≥0.19.0).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.23.0"),
        .package(url: "https://github.com/xocialize/mimi-encoder-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Native downloader for WeightSourcing auto-materialization.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "MLXMimiCodec",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // Package identity derives from the repo URL's last path component.
                .product(name: "MimiCodecEncoder", package: "mimi-encoder-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            // MimiEncoder (MLX Module) crosses into the @InferenceActor; v5 mode relaxes strict
            // region-isolation to a warning (the engine serializes lifecycle) — same as siblings.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXMimiCodecTests",
            dependencies: [
                "MLXMimiCodec",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                // The offline MAT-1..5 materialization gate.
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
