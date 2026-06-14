# mlx-mimi-codec-swift

The MLXEngine **`audioCodec`** package over Kyutai's [Mimi](https://huggingface.co/kyutai/mimi) neural audio codec **encoder** — encodes audio into discrete RVQ tokens on Apple Silicon.

A thin conformance layer that wraps the standalone inference engine
[`mimi-encoder-mlx-swift`](https://github.com/xocialize/mimi-encoder-mlx-swift) (product
`MimiCodecEncoder`) and exposes it to [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift)
as a `ModelPackage`. All model logic (SEANet conv encoder → causal transformer → stride-2
downsample → split RVQ) lives in the core; this package maps the canonical
`AudioCodecRequest → AudioCodecResponse` contract onto it and owns the engine lifecycle.

## Capability

| | |
|---|---|
| Capability | `audioCodec` (encode direction) |
| Input | `Audio` (.wav, any rate/channels — resampled to 24 kHz mono) |
| Output | `AudioCodecResponse` — `[16, T]` per-codebook `[[Int32]]` token grid at 12.5 Hz |

Encode only; decoding tokens back to audio is a separate future capability.

## Weights

`mlx-community/mimi-encoder-mlx` (`encoder.safetensors`, fp32), selected via
`MimiCodecConfiguration.repo`. Derived from `kyutai/mimi` (CC-BY-4.0).

## Usage

```swift
import MLXServeCore
import MLXMimiCodec

let engine = MLXServeEngine()
try await engine.register(MimiCodecPackage.registration, configuration: MimiCodecConfiguration())

let clip = Audio(format: .wav, data: wavBytes)
let response = try await engine.run(AudioCodecRequest(audio: clip)) as! AudioCodecResponse
print(response.numCodebooks, response.codes.first?.count ?? 0)   // 16, T frames
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-mimi-codec-swift", from: "0.1.0")`, then import `MLXMimiCodec` (the conformant `audioCodec` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port code MIT; weights CC-BY-4.0 (Kyutai).
