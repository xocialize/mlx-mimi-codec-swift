// MaterializationTests.swift — Mimi through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. Single encoder checkpoint — one declaration covers the package.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXMimiCodec

final class MaterializationTests: XCTestCase {

    /// Temp dir holding a probe file that makes an explicit-dir config read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "mimi-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: dir.appending(path: MimiCodecConfiguration.weightsFile).path, contents: Data([0]))
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: MimiCodecConfiguration(),
            satisfiedConfiguration: MimiCodecConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresSingleMainSource() {
        let sources = MimiCodecConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/mimi-encoder-mlx")
        XCTAssertEqual(sources[0].matching, ["encoder.safetensors"])
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "mimi-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = MimiCodecConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected layout.
        let dir = root.appending(path: cfg.repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: dir.appending(path: MimiCodecConfiguration.weightsFile).path, contents: Data([0]))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).modelDirectory?.path, dir.path)
        let explicit = MimiCodecConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = MimiCodecConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(
            cfg.prewarmPaths.map(\.path),
            [root.appending(path: "mlx-community/mimi-encoder-mlx/encoder.safetensors").path])
    }

    func testCodableRoundTrip() throws {
        let cfg = MimiCodecConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(MimiCodecConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}
