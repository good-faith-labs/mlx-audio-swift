import MLX
import XCTest
@testable import MLXAudioTTS

final class KokoroTimingsSmokeTests: XCTestCase {
    private struct StubPhonemeProcessor: TextProcessor {
        func prepare() async throws {}
        func process(text: String, language: String?) throws -> String { "hˈaɪ" }
    }

    func testLoaderRouteExposesTimedSpeechGenerationModelAndProducesMonotonicTokens() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["KOKORO_REGRESSION"] != "1",
            "set KOKORO_REGRESSION=1 to run"
        )

        let model: SpeechGenerationModel = try await TTS.loadModel(
            modelRepo: "mlx-community/Kokoro-82M-bf16"
        )
        let timed = try XCTUnwrap(model as? TimedSpeechGenerationModel)

        let text = "Once upon a time, there was a quiet library."
        let result = try await timed.generateWithTimings(
            text: text,
            voice: "bf_emma",
            refAudio: nil,
            refText: nil,
            language: nil,
            generationParameters: model.defaultGenerationParameters
        )

        XCTAssertGreaterThan(result.tokens.count, 5)
        XCTAssertEqual(result.sampleRate, model.sampleRate)

        for i in 1..<result.tokens.count {
            XCTAssertLessThanOrEqual(
                result.tokens[i - 1].endSeconds,
                result.tokens[i].startSeconds + 1e-6
            )
        }

        for token in result.tokens {
            XCTAssertGreaterThanOrEqual(token.normalizedStart, 0)
            XCTAssertLessThan(token.normalizedStart, token.normalizedEnd)
            XCTAssertLessThanOrEqual(token.normalizedEnd, text.utf16.count)
        }
    }

    func testNonEnglishVoiceReturnsEmptyTokens() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["KOKORO_REGRESSION"] != "1",
            "set KOKORO_REGRESSION=1 to run"
        )

        let model: SpeechGenerationModel = try await TTS.loadModel(
            modelRepo: "mlx-community/Kokoro-82M-bf16"
        )
        let timed = try XCTUnwrap(model as? TimedSpeechGenerationModel)

        let result = try await timed.generateWithTimings(
            text: "Bonjour le monde.",
            voice: "ff_siwis",
            refAudio: nil,
            refText: nil,
            language: "fr",
            generationParameters: model.defaultGenerationParameters
        )

        XCTAssertGreaterThan(result.audio.shape.last ?? 0, 0)
        XCTAssertEqual(result.tokens, [])
    }

    func testNilTextProcessorReturnsAudioAndEmptyTokensWithoutSubstitution() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["KOKORO_REGRESSION"] != "1",
            "set KOKORO_REGRESSION=1 to run"
        )

        let model = try await TTS.loadModel(modelRepo: "mlx-community/Kokoro-82M-bf16") as! KokoroModel
        model.setTextProcessor(nil)

        let result = try await model.generateWithTimings(
            text: "hˈaɪ",
            voice: "bf_emma",
            refAudio: nil,
            refText: nil,
            language: "en-us",
            generationParameters: model.defaultGenerationParameters
        )

        XCTAssertGreaterThan(result.audio.shape.last ?? 0, 0)
        XCTAssertEqual(result.tokens, [])
    }

    func testUnsupportedCustomProcessorReturnsAudioAndEmptyTokensWithoutSubstitution() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["KOKORO_REGRESSION"] != "1",
            "set KOKORO_REGRESSION=1 to run"
        )

        let model = try await TTS.loadModel(modelRepo: "mlx-community/Kokoro-82M-bf16") as! KokoroModel
        model.setTextProcessor(StubPhonemeProcessor())

        let result = try await model.generateWithTimings(
            text: "hi",
            voice: "bf_emma",
            refAudio: nil,
            refText: nil,
            language: "en-us",
            generationParameters: model.defaultGenerationParameters
        )

        XCTAssertGreaterThan(result.audio.shape.last ?? 0, 0)
        XCTAssertEqual(result.tokens, [])
    }

    func testPreparedTimingSeamProducesStableOffsetsForRetokenizedText() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["KOKORO_REGRESSION"] != "1",
            "set KOKORO_REGRESSION=1 to run"
        )

        let model = try await TTS.loadModel(modelRepo: "mlx-community/Kokoro-82M-bf16") as! KokoroModel
        let text = "In 3.14 seconds, the well-known reader said don't stop."
        let result = try await model.generateWithTimings(
            text: text,
            voice: "bf_emma",
            refAudio: nil,
            refText: nil,
            language: "en-us",
            generationParameters: model.defaultGenerationParameters
        )

        XCTAssertGreaterThan(result.audio.shape.last ?? 0, 0)
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertTrue(result.tokens.contains { $0.text == "3.14" || text.utf16OffsetSlice($0) == "3.14" })
        XCTAssertTrue(result.tokens.contains { $0.text == "well-known" || text.utf16OffsetSlice($0) == "well-known" })
        XCTAssertTrue(result.tokens.contains { $0.text == "don't" || text.utf16OffsetSlice($0) == "don't" })

        for token in result.tokens {
            XCTAssertGreaterThanOrEqual(token.normalizedStart, 0)
            XCTAssertLessThan(token.normalizedStart, token.normalizedEnd)
            XCTAssertLessThanOrEqual(token.normalizedEnd, text.utf16.count)
            XCTAssertEqual(text.utf16OffsetSlice(token), token.text)
        }
    }
}

private extension String {
    func utf16OffsetSlice(_ token: KokoroTimingToken) -> String {
        let lower = String.Index(utf16Offset: token.normalizedStart, in: self)
        let upper = String.Index(utf16Offset: token.normalizedEnd, in: self)
        return String(self[lower..<upper])
    }
}
