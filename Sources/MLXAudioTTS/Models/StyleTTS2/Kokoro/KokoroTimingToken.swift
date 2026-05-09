import Foundation
@preconcurrency import MLX

public struct KokoroTimingToken: Sendable, Codable, Equatable {
    public let text: String
    public let normalizedStart: Int
    public let normalizedEnd: Int
    public let phonemes: String
    public let startSeconds: Double
    public let endSeconds: Double
    public let isPunctuation: Bool

    public init(
        text: String,
        normalizedStart: Int,
        normalizedEnd: Int,
        phonemes: String,
        startSeconds: Double,
        endSeconds: Double,
        isPunctuation: Bool
    ) {
        self.text = text
        self.normalizedStart = normalizedStart
        self.normalizedEnd = normalizedEnd
        self.phonemes = phonemes
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.isPunctuation = isPunctuation
    }
}

public struct KokoroTimedGeneration {
    public let audio: MLXArray
    public let sampleRate: Int
    public let tokens: [KokoroTimingToken]

    public init(audio: MLXArray, sampleRate: Int, tokens: [KokoroTimingToken]) {
        self.audio = audio
        self.sampleRate = sampleRate
        self.tokens = tokens
    }
}

struct KokoroTimingSeed: Sendable, Equatable {
    let text: String
    let normalizedStart: Int
    let normalizedEnd: Int
    let phonemes: String
    let whitespace: String
}

enum KokoroTimingError: Error, Equatable {
    case phonemeCountMismatch(cursor: Int, count: Int, predDurLength: Int)
    case invalidTimingScale(sampleRate: Int, hopSize: Int, upsampleProduct: Int)
    case invalidSecondsPerFrame(Double)
    case nonEnglishTimingRequested(String)
    case invalidTokenRange(token: String, lower: Int, upper: Int, utf16Length: Int)
}

extension KokoroTimingToken {
    private static let punctuationOnly: NSRegularExpression = {
        // Text consists entirely of punctuation, symbols, or whitespace.
        try! NSRegularExpression(pattern: "^[\\p{P}\\p{S}\\s]+$")
    }()

    static func isPunctuationOnly(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return punctuationOnly.firstMatch(in: text, range: range) != nil
    }
}

extension KokoroTimingSeed {
    static func makeAll(from tokens: [MToken], originalText: String) throws -> [KokoroTimingSeed] {
        let utf16Length = originalText.utf16.count
        return try tokens.map { token in
            let lower = token.tokenRange.lowerBound.utf16Offset(in: originalText)
            let upper = token.tokenRange.upperBound.utf16Offset(in: originalText)
            guard lower >= 0, upper > lower, upper <= utf16Length else {
                throw KokoroTimingError.invalidTokenRange(
                    token: token.text,
                    lower: lower,
                    upper: upper,
                    utf16Length: utf16Length
                )
            }

            let lowerIndex = String.Index(utf16Offset: lower, in: originalText)
            let upperIndex = String.Index(utf16Offset: upper, in: originalText)
            let slice = String(originalText[lowerIndex..<upperIndex])
            guard slice == token.text else {
                throw KokoroTimingError.invalidTokenRange(
                    token: token.text,
                    lower: lower,
                    upper: upper,
                    utf16Length: utf16Length
                )
            }

            return KokoroTimingSeed(
                text: token.text,
                normalizedStart: lower,
                normalizedEnd: upper,
                phonemes: token.phonemes ?? "",
                whitespace: token.whitespace
            )
        }
    }
}
