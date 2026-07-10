// CancellationTests.swift — Mimi codec encoder through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint fires before the
// notLoaded guard or weights, so a stub configuration suffices. CAN-3: MimiEncoder.encode(audio:)
// is one whole-clip MLX graph eval — SEANet conv encoder → causal transformer → stride-2
// downsample → split RVQ, built as a single graph with NO per-frame/per-chunk loop (the
// "streaming encoder" framing describes the model's causality, not the Swift core's execution) —
// so the sub-second exemption is the honest posture. No do/catch on the run() path can launder a
// CancellationError (the decode-audio catches wrap sync AVFoundation I/O only).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXMimiCodec

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MimiCodecPackage(configuration: MimiCodecConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: AudioCodecRequest(audio: Audio(format: .wav, data: Data(),
                                                    sampleRate: 24_000, channels: 1)))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // audioCodec is not a long-run capability and the manifest declares no multi-GB
        // activation peak (~0.8 GB resident, no peakActivationBytes) — short-run envelope.
        XCTAssertFalse(CancellationConformance.longRunImplied(by: MimiCodecPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MimiCodecPackage.manifest,
            posture: .subSecondRuns(
                reason: "one whole-clip encoder forward — a single MLX graph eval (SEANet → "
                    + "transformer → downsample → RVQ in MimiEncoder.encode); the core has no "
                    + "per-frame or per-chunk execution loop to checkpoint"))
        XCTAssertTrue(report.passed, report.summary)
    }
}
