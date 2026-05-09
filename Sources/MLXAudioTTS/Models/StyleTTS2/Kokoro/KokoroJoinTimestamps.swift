import Foundation

enum KokoroJoinTimestamps {
    /// Returns copied `MToken` objects with `start_ts` and `end_ts` filled.
    /// Pure from the caller's perspective: input token objects are not mutated.
    @discardableResult
    static func run(
        predDur: [Int32],
        tokens: [MToken],
        secondsPerFrame: Double
    ) throws -> [MToken] {
        guard secondsPerFrame.isFinite, secondsPerFrame > 0 else {
            throw KokoroTimingError.invalidSecondsPerFrame(secondsPerFrame)
        }

        guard predDur.count >= 2 else {
            throw KokoroTimingError.phonemeCountMismatch(
                cursor: 0,
                count: 2,
                predDurLength: predDur.count
            )
        }

        var phonemeCursor = 1
        var leftHalfFrames = Int64(2 * max(0, Int(predDur[0]) - 3))
        var rightHalfFrames = leftHalfFrames
        let timedTokens = tokens.map { MToken(copying: $0) }

        for token in timedTokens {
            let phonemeCount = (token.phonemes ?? "").utf16.count
            let hasWhitespace = !token.whitespace.isEmpty
            if phonemeCount == 0 {
                if hasWhitespace {
                    guard phonemeCursor < predDur.count else {
                        throw KokoroTimingError.phonemeCountMismatch(
                            cursor: phonemeCursor,
                            count: 1,
                            predDurLength: predDur.count
                        )
                    }
                    let spaceHalfFrames = Int64(predDur[phonemeCursor])
                    leftHalfFrames = rightHalfFrames + spaceHalfFrames
                    token.start_ts = Double(leftHalfFrames) * secondsPerFrame
                    token.end_ts = token.start_ts
                    rightHalfFrames = leftHalfFrames + spaceHalfFrames
                    phonemeCursor += 1
                } else {
                    token.start_ts = Double(leftHalfFrames) * secondsPerFrame
                    token.end_ts = token.start_ts
                }
                continue
            }

            let phonemeUpper = phonemeCursor + phonemeCount
            let upper = phonemeUpper + (hasWhitespace ? 1 : 0)
            guard upper <= predDur.count else {
                throw KokoroTimingError.phonemeCountMismatch(
                    cursor: phonemeCursor,
                    count: phonemeCount + (hasWhitespace ? 1 : 0),
                    predDurLength: predDur.count
                )
            }

            token.start_ts = Double(leftHalfFrames) * secondsPerFrame

            var tokenDurationFrames: Int64 = 0
            for i in phonemeCursor..<phonemeUpper {
                tokenDurationFrames += Int64(predDur[i])
            }
            let spaceHalfFrames = hasWhitespace ? Int64(predDur[phonemeUpper]) : 0

            leftHalfFrames = rightHalfFrames + (2 * tokenDurationFrames) + spaceHalfFrames
            token.end_ts = Double(leftHalfFrames) * secondsPerFrame
            rightHalfFrames = leftHalfFrames + spaceHalfFrames

            phonemeCursor = upper
        }

        let eosPadding = 1
        guard phonemeCursor + eosPadding <= predDur.count else {
            throw KokoroTimingError.phonemeCountMismatch(
                cursor: phonemeCursor,
                count: eosPadding,
                predDurLength: predDur.count
            )
        }

        return timedTokens
    }

    static func materialize(
        predDur: [Int32],
        tokens: [MToken],
        timingSeeds: [KokoroTimingSeed],
        originalUTF16Length: Int,
        secondsPerFrame: Double
    ) throws -> [KokoroTimingToken] {
        let timedTokens = try run(
            predDur: predDur,
            tokens: tokens,
            secondsPerFrame: secondsPerFrame
        )

        guard timedTokens.count == timingSeeds.count else {
            throw KokoroTimingError.invalidTokenRange(
                token: "<timing-seed-count-mismatch>",
                lower: timingSeeds.count,
                upper: timedTokens.count,
                utf16Length: originalUTF16Length
            )
        }

        var out: [KokoroTimingToken] = []
        out.reserveCapacity(timedTokens.count)
        for (token, seed) in zip(timedTokens, timingSeeds) {
            guard
                seed.normalizedStart >= 0,
                seed.normalizedEnd > seed.normalizedStart,
                seed.normalizedEnd <= originalUTF16Length
            else {
                throw KokoroTimingError.invalidTokenRange(
                    token: seed.text,
                    lower: seed.normalizedStart,
                    upper: seed.normalizedEnd,
                    utf16Length: originalUTF16Length
                )
            }
            out.append(
                KokoroTimingToken(
                    text: seed.text,
                    normalizedStart: seed.normalizedStart,
                    normalizedEnd: seed.normalizedEnd,
                    phonemes: token.phonemes ?? "",
                    startSeconds: token.start_ts ?? 0,
                    endSeconds: token.end_ts ?? (token.start_ts ?? 0),
                    isPunctuation: KokoroTimingToken.isPunctuationOnly(seed.text)
                )
            )
        }
        return out
    }
}
