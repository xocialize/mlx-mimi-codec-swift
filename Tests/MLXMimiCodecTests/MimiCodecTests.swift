import Testing
import Foundation
import MLXToolKit
@testable import MLXMimiCodec

/// Offline conformance checks — no weights, no Metal. Live encode + codebook parity is proven in
/// the `MLXEngine Testing` app (the model needs the GPU and the published checkpoint).
struct MimiCodecTests {

    @Test func manifestIsAudioCodecAndPermissive() {
        let m = MimiCodecPackage.manifest
        #expect(m.capabilities == [.audioCodec])
        #expect(m.license.weightLicense == .ccBy4)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.provenance.sourceRepo == "mlx-community/mimi-encoder-mlx")
    }

    @Test func licenseGateAdmitsCcBy4Weights() {
        #expect(LicensePolicy.permissiveOnly.evaluate(MimiCodecPackage.manifest.license) == .admitted)
    }

    @Test func manifestRequirementsAreFullPrecision() {
        let r = MimiCodecPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .fp32)
    }

    @Test func surfaceIsTheCanonicalAudioCodecDescriptor() {
        let surface = MimiCodecPackage.manifest.surfaces.first
        #expect(surface?.capability == .audioCodec)
        #expect(surface?.parameters.first?.kind == .audio)
    }

    @Test func registrationConstructs() throws {
        let reg = MimiCodecPackage.registration
        #expect(reg.manifest.capabilities == [.audioCodec])
        let pkg = try reg.makePackage(MimiCodecConfiguration())
        #expect(pkg is MimiCodecPackage)
    }

    @Test func configurationDefaultsToPublishedRepo() {
        #expect(MimiCodecConfiguration().repo == "mlx-community/mimi-encoder-mlx")
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = MimiCodecConfiguration()
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/should-not-persist")
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(MimiCodecConfiguration.self, from: data)
        #expect(back.repo == "mlx-community/mimi-encoder-mlx")
        #expect(back.modelsRootDirectory == nil)
    }
}
