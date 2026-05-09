import XCTest
@testable import MLXAudioTTS

final class KokoroJoinTimestampsTests: XCTestCase {
    private let secondsPerFrame: Double = 1.0 / 80.0

    func testSingleEnglishWordAssignsTimingFromPredDur() throws {
        let original = "hi"
        let token = MToken(
            text: "hi",
            tokenRange: original.startIndex..<original.endIndex,
            whitespace: "",
            phonemes: "hˈaɪ"
        )
        let predDur: [Int32] = [2, 4, 2, 6, 4, 2]

        let result = try KokoroJoinTimestamps.run(
            predDur: predDur,
            tokens: [token],
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].start_ts ?? -1, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[0].end_ts ?? -1, 0.400, accuracy: 1e-9)
        XCTAssertNil(token.start_ts)
        XCTAssertNil(token.end_ts)
    }

    func testBosAudioOffsetsFirstTokenWhenBOSExceedsUpstreamTrim() throws {
        let original = "hi"
        let token = MToken(
            text: "hi",
            tokenRange: original.startIndex..<original.endIndex,
            whitespace: "",
            phonemes: "hˈaɪ"
        )

        let result = try KokoroJoinTimestamps.run(
            predDur: [5, 4, 2, 6, 4, 2],
            tokens: [token],
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result[0].start_ts ?? -1, 0.050, accuracy: 1e-9)
        XCTAssertEqual(result[0].end_ts ?? -1, 0.450, accuracy: 1e-9)
    }

    func testMultiWordSequenceAssignsMonotonicTimings() throws {
        let original = "the cat"
        let theTok = MToken(
            text: "the",
            tokenRange: original.range(of: "the")!,
            whitespace: " ",
            phonemes: "ðə"
        )
        let catTok = MToken(
            text: "cat",
            tokenRange: original.range(of: "cat")!,
            whitespace: "",
            phonemes: "kˈæt"
        )
        let predDur: [Int32] = [2, 3, 2, 1, 4, 2, 5, 3, 2]

        let result = try KokoroJoinTimestamps.run(
            predDur: predDur,
            tokens: [theTok, catTok],
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result[0].start_ts!, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[0].end_ts!, 0.1375, accuracy: 1e-9)
        XCTAssertEqual(result[1].start_ts!, 0.1375, accuracy: 1e-9)
        XCTAssertEqual(result[1].end_ts!, 0.500, accuracy: 1e-9)
        assertTimingInvariants(result)
    }

    func testTrailingPunctuationGetsItsOwnTiming() throws {
        let original = "hi!"
        let hi = MToken(text: "hi", tokenRange: original.range(of: "hi")!, whitespace: "", phonemes: "hˈaɪ")
        let bang = MToken(text: "!", tokenRange: original.range(of: "!")!, whitespace: "", phonemes: "!")
        let predDur: [Int32] = [2, 4, 2, 6, 4, 2, 2]

        let result = try KokoroJoinTimestamps.run(
            predDur: predDur,
            tokens: [hi, bang],
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result[0].end_ts!, 0.400, accuracy: 1e-9)
        XCTAssertEqual(result[1].start_ts!, 0.400, accuracy: 1e-9)
        XCTAssertEqual(result[1].end_ts!, 0.450, accuracy: 1e-9)
    }

    func testMergedContractionGetsSingleSpan() throws {
        let original = "don't"
        let token = MToken(
            text: "don't",
            tokenRange: original.startIndex..<original.endIndex,
            whitespace: "",
            phonemes: "doʊnt"
        )
        let result = try KokoroJoinTimestamps.run(
            predDur: [2, 4, 5, 3, 4, 3, 2],
            tokens: [token],
            secondsPerFrame: secondsPerFrame
        )
        XCTAssertEqual(result[0].start_ts!, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[0].end_ts!, Double(2 * (4 + 5 + 3 + 4 + 3)) * secondsPerFrame, accuracy: 1e-9)
    }

    func testCursorOverrunThrows() {
        let original = "x"
        let token = MToken(
            text: "x",
            tokenRange: original.startIndex..<original.endIndex,
            whitespace: "",
            phonemes: "xxxx"
        )

        XCTAssertThrowsError(
            try KokoroJoinTimestamps.run(
                predDur: [2, 1, 2],
                tokens: [token],
                secondsPerFrame: secondsPerFrame
            )
        ) { error in
            guard case KokoroTimingError.phonemeCountMismatch = error else {
                return XCTFail("expected phonemeCountMismatch, got \(error)")
            }
        }
    }

    func testInvalidSecondsPerFrameThrows() {
        let original = "a"
        let token = MToken(text: "a", tokenRange: original.startIndex..<original.endIndex, whitespace: "", phonemes: "a")

        XCTAssertThrowsError(
            try KokoroJoinTimestamps.run(
                predDur: [2, 5, 2],
                tokens: [token],
                secondsPerFrame: .nan
            )
        ) { error in
            guard case KokoroTimingError.invalidSecondsPerFrame = error else {
                return XCTFail("expected invalidSecondsPerFrame, got \(error)")
            }
        }
    }

    func testEmptyPhonemeTokenIsZeroDurationAtCursor() throws {
        let original = "ab"
        let a = MToken(text: "a", tokenRange: original.range(of: "a")!, whitespace: "", phonemes: "a")
        let b = MToken(text: "b", tokenRange: original.range(of: "b")!, whitespace: "", phonemes: "")

        let result = try KokoroJoinTimestamps.run(
            predDur: [2, 5, 2],
            tokens: [a, b],
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result[0].end_ts!, 0.125, accuracy: 1e-9)
        XCTAssertEqual(result[1].start_ts!, 0.125, accuracy: 1e-9)
        XCTAssertEqual(result[1].end_ts!, 0.125, accuracy: 1e-9)
    }

    func testTokenCannotConsumeEOSPadding() {
        let original = "a"
        let token = MToken(text: "a", tokenRange: original.startIndex..<original.endIndex, whitespace: "", phonemes: "a")

        XCTAssertThrowsError(
            try KokoroJoinTimestamps.run(
                predDur: [2, 5],
                tokens: [token],
                secondsPerFrame: secondsPerFrame
            )
        ) { error in
            guard case KokoroTimingError.phonemeCountMismatch = error else {
                return XCTFail("expected phonemeCountMismatch, got \(error)")
            }
        }
    }

    func testEmptyInputWithOnlyBosAndEosReturnsEmptyTokens() throws {
        let result = try KokoroJoinTimestamps.run(
            predDur: [2, 2],
            tokens: [],
            secondsPerFrame: secondsPerFrame
        )
        XCTAssertEqual(result.count, 0)
    }

    func testLeadingAndMedialPunctuationRemainMonotonic() throws {
        let original = "\"hello, yes"
        let quote = MToken(text: "\"", tokenRange: original.range(of: "\"")!, whitespace: "", phonemes: "\"")
        let hello = MToken(text: "hello", tokenRange: original.range(of: "hello")!, whitespace: "", phonemes: "hello")
        let comma = MToken(text: ",", tokenRange: original.range(of: ",")!, whitespace: " ", phonemes: ",")
        let yes = MToken(text: "yes", tokenRange: original.range(of: "yes")!, whitespace: "", phonemes: "yes")
        let predDur = Array(repeating: Int32(1), count: 13)

        let result = try KokoroJoinTimestamps.run(
            predDur: predDur,
            tokens: [quote, hello, comma, yes],
            secondsPerFrame: secondsPerFrame
        )
        assertTimingInvariants(result)
    }

    func testAdditionalFixtureShapesPreserveCountAndMonotonicTiming() throws {
        let fixtures: [(String, [(String, String, String)])] = [
            ("1984", [("1984", "1984", "")]),
            ("3.14", [("3.14", "3.14", "")]),
            ("well-known", [("well-known", "wellknown", "")]),
            ("yes, however,", [("yes", "yes", ""), (",", ",", " "), ("however", "however", ""), (",", ",", "")]),
            ("x", [("x", "x", "")]),
            ("hello 世界", [("hello", "hello", " "), ("世界", "世界", "")]),
        ]

        for (original, specs) in fixtures {
            let tokens = specs.map { text, phonemes, whitespace in
                MToken(text: text, tokenRange: original.range(of: text)!, whitespace: whitespace, phonemes: phonemes)
            }
            let units = tokens.reduce(0) { $0 + (($1.phonemes ?? "") + $1.whitespace).utf16.count }
            let predDur = Array(repeating: Int32(1), count: units + 2)
            let result = try KokoroJoinTimestamps.run(
                predDur: predDur,
                tokens: tokens,
                secondsPerFrame: secondsPerFrame
            )
            XCTAssertEqual(result.count, tokens.count)
            assertTimingInvariants(result, file: #filePath, line: #line)
        }
    }

    func testMaximumLengthInputAround510ModelUnits() throws {
        let original = String(repeating: "a", count: 510)
        let token = MToken(
            text: original,
            tokenRange: original.startIndex..<original.endIndex,
            whitespace: "",
            phonemes: original
        )
        let predDur = Array(repeating: Int32(1), count: 512)
        let result = try KokoroJoinTimestamps.run(
            predDur: predDur,
            tokens: [token],
            secondsPerFrame: secondsPerFrame
        )
        XCTAssertEqual(result[0].start_ts!, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[0].end_ts!, 510 * 2 * secondsPerFrame, accuracy: 1e-9)
    }

    func testRunMaterializingReturnsTokensWithUtf16Offsets() throws {
        let original = "hi cat"
        let hi = MToken(text: "hi", tokenRange: original.range(of: "hi")!, whitespace: " ", phonemes: "hˈaɪ")
        let cat = MToken(text: "cat", tokenRange: original.range(of: "cat")!, whitespace: "", phonemes: "kˈæt")
        let timingSeeds = try KokoroTimingSeed.makeAll(from: [hi, cat], originalText: original)

        let result = try KokoroJoinTimestamps.materialize(
            predDur: [2, 4, 2, 6, 4, 1, 4, 2, 5, 3, 2],
            tokens: [hi, cat],
            timingSeeds: timingSeeds,
            originalUTF16Length: original.utf16.count,
            secondsPerFrame: secondsPerFrame
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].text, "hi")
        XCTAssertEqual(result[0].normalizedStart, 0)
        XCTAssertEqual(result[0].normalizedEnd, 2)
        XCTAssertEqual(result[1].text, "cat")
        XCTAssertEqual(result[1].normalizedStart, 3)
        XCTAssertEqual(result[1].normalizedEnd, 6)
        XCTAssertFalse(result[0].isPunctuation)
        XCTAssertFalse(result[1].isPunctuation)
        assertRangeInvariants(result, originalUTF16Length: original.utf16.count)
    }

    func testSeedRejectsTextRangeMismatch() {
        let original = "hi"
        let token = MToken(text: "bye", tokenRange: original.startIndex..<original.endIndex, whitespace: "", phonemes: "b")

        XCTAssertThrowsError(try KokoroTimingSeed.makeAll(from: [token], originalText: original)) { error in
            guard case KokoroTimingError.invalidTokenRange = error else {
                return XCTFail("expected invalidTokenRange, got \(error)")
            }
        }
    }

    func testMaterializeRejectsCorruptSeedOffsets() {
        let original = "hi"
        let token = MToken(text: "hi", tokenRange: original.startIndex..<original.endIndex, whitespace: "", phonemes: "h")
        let seed = KokoroTimingSeed(text: "hi", normalizedStart: 2, normalizedEnd: 1, phonemes: "h", whitespace: "")

        XCTAssertThrowsError(
            try KokoroJoinTimestamps.materialize(
                predDur: [2, 1, 2],
                tokens: [token],
                timingSeeds: [seed],
                originalUTF16Length: original.utf16.count,
                secondsPerFrame: secondsPerFrame
            )
        ) { error in
            guard case KokoroTimingError.invalidTokenRange = error else {
                return XCTFail("expected invalidTokenRange, got \(error)")
            }
        }
    }

    private func assertTimingInvariants(
        _ tokens: [MToken],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token.end_ts ?? -1, token.start_ts ?? -1, file: file, line: line)
        }
        for i in 1..<tokens.count {
            XCTAssertLessThanOrEqual(tokens[i - 1].end_ts ?? -1, tokens[i].start_ts ?? -1, file: file, line: line)
        }
    }

    private func assertRangeInvariants(
        _ tokens: [KokoroTimingToken],
        originalUTF16Length: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token.normalizedStart, 0, file: file, line: line)
            XCTAssertLessThan(token.normalizedStart, token.normalizedEnd, file: file, line: line)
            XCTAssertLessThanOrEqual(token.normalizedEnd, originalUTF16Length, file: file, line: line)
        }
        for i in 1..<tokens.count {
            XCTAssertLessThanOrEqual(tokens[i - 1].normalizedEnd, tokens[i].normalizedStart, file: file, line: line)
        }
    }
}
