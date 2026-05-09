import XCTest
@testable import MLXAudioTTS

final class KokoroFallbackTests: XCTestCase {
    func testKokoroModelConformsToTimedSpeechGenerationModel() {
        let _: any TimedSpeechGenerationModel.Type = KokoroModel.self
    }

    func testCleanStatsReturnsUseTokens() {
        let decision = KokoroModel.decideFallback(
            stats: .clean,
            modelInputUnitCount: 10,
            timingUnitCount: 10
        )
        XCTAssertEqual(decision, .useTokens)
    }

    func testAnyNaNTriggersFallback() {
        let stats = KokoroModel.PredDurDefenseStats(nanReplacements: 1, clippedDurations: 0)
        XCTAssertEqual(
            KokoroModel.decideFallback(
                stats: stats,
                modelInputUnitCount: 10,
                timingUnitCount: 10
            ),
            .empty(reason: .nan)
        )
    }

    func testClipRatioBelowThresholdUsesTokens() {
        let stats = KokoroModel.PredDurDefenseStats(nanReplacements: 0, clippedDurations: 2)
        XCTAssertEqual(
            KokoroModel.decideFallback(
                stats: stats,
                modelInputUnitCount: 10,
                timingUnitCount: 10
            ),
            .useTokens
        )
    }

    func testClipRatioAtBoundaryUsesTokens() {
        let stats = KokoroModel.PredDurDefenseStats(nanReplacements: 0, clippedDurations: 25)
        XCTAssertEqual(
            KokoroModel.decideFallback(
                stats: stats,
                modelInputUnitCount: 100,
                timingUnitCount: 100
            ),
            .useTokens
        )
    }

    func testClipRatioAboveThresholdTriggersFallback() {
        let stats = KokoroModel.PredDurDefenseStats(nanReplacements: 0, clippedDurations: 26)
        XCTAssertEqual(
            KokoroModel.decideFallback(
                stats: stats,
                modelInputUnitCount: 100,
                timingUnitCount: 100
            ),
            .empty(reason: .clip)
        )
    }

    func testUnitCountMismatchTriggersFallbackEvenWhenStatsAreClean() {
        XCTAssertEqual(
            KokoroModel.decideFallback(
                stats: .clean,
                modelInputUnitCount: 3,
                timingUnitCount: 4
            ),
            .empty(reason: .unitCountMismatch)
        )
    }

    func testModelUnitCountIncludesTokenWhitespace() {
        let original = "hi cat"
        let hi = MToken(text: "hi", tokenRange: original.range(of: "hi")!, whitespace: " ", phonemes: "hˈaɪ")
        let cat = MToken(text: "cat", tokenRange: original.range(of: "cat")!, whitespace: "", phonemes: "kˈæt")

        XCTAssertEqual(KokoroModel.modelUnitCount(in: [hi, cat]), 9)
    }

    func testTimingLanguageDispositionAcceptsResolvedEnglishVoiceAndLanguage() {
        XCTAssertEqual(
            KokoroModel.timingLanguageDisposition(voiceName: "bf_emma", language: nil),
            .english("en-gb")
        )
        XCTAssertEqual(
            KokoroModel.timingLanguageDisposition(voiceName: "af_heart", language: "en-us"),
            .english("en-us")
        )
    }

    func testTimingLanguageDispositionRejectsNonEnglishAndUnresolvedLanguageWithoutModelLoad() {
        XCTAssertEqual(
            KokoroModel.timingLanguageDisposition(voiceName: "ff_siwis", language: nil),
            .emptyTokens
        )
        XCTAssertEqual(
            KokoroModel.timingLanguageDisposition(voiceName: "xf_unknown", language: nil),
            .emptyTokens
        )
        XCTAssertEqual(
            KokoroModel.timingLanguageDisposition(voiceName: "bf_emma", language: "ja"),
            .emptyTokens
        )
    }

    func testSecondsPerFrameRejectsInvalidScale() {
        XCTAssertThrowsError(
            try KokoroModel.secondsPerFrame(sampleRate: 0, hopSize: 5, upsampleProduct: 60)
        ) { error in
            guard case KokoroTimingError.invalidTimingScale = error else {
                return XCTFail("expected invalidTimingScale, got \(error)")
            }
        }
    }
}
