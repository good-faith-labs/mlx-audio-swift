import Foundation
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import os

extension KokoroModel: TimedSpeechGenerationModel {
    private static let timingsLog = Logger(subsystem: "mlx-audio-swift", category: "kokoro.timings")

    private static let timingMaxTokenCount = 510

    enum TimingLanguageDisposition: Equatable {
        case english(String)
        case emptyTokens
    }

    enum FallbackDecision: Equatable {
        case useTokens
        case empty(reason: FallbackReason)
    }

    enum FallbackReason: Equatable {
        case nan
        case clip
        case unitCountMismatch
    }

    public func generateWithTimings(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> KokoroTimedGeneration {
        let voiceName = voice ?? "af_heart"
        let languageDisposition = Self.timingLanguageDisposition(
            voiceName: voiceName,
            language: language
        )

        guard case .english(let inferredLang) = languageDisposition else {
            let audio = try await self.generate(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: language,
                generationParameters: generationParameters
            )
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }

        if let multilingual = textProcessor as? KokoroMultilingualProcessor {
            try await multilingual.prepare(for: inferredLang)
        }

        let timingInput: (phonemized: String, tokens: [MToken], timingSeeds: [KokoroTimingSeed])
        do {
            guard let prepared = try await phonemizeEnglishForTimings(
                text: text,
                language: inferredLang
            ) else {
                Self.timingsLog.warning(
                    "Kokoro timings unavailable for textProcessor=\(String(describing: self.textProcessor))"
                )
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
            }
            timingInput = prepared
        } catch KokoroTimingError.nonEnglishTimingRequested(_) {
            let audio = try await self.generate(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: language,
                generationParameters: generationParameters
            )
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        } catch let error as KokoroTimingError {
            Self.timingsLog.error(
                "Kokoro timing seam failed: \(String(describing: error)); returning empty tokens"
            )
            let audio = try await self.generate(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: language,
                generationParameters: generationParameters
            )
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }

        let (phonemized, mTokens, timingSeeds) = timingInput
        let tokenIds = tokenize(phonemized)
        let modelInputUnitCount = tokenIds.count
        guard modelInputUnitCount <= Self.timingMaxTokenCount else {
            throw AudioGenerationError.invalidInput(
                "Input too long: \(modelInputUnitCount) tokens exceeds max \(Self.timingMaxTokenCount)"
            )
        }

        var tokenArray = [Int32(0)]
        tokenArray.append(contentsOf: tokenIds.map { Int32($0) })
        tokenArray.append(0)
        let inputIds = MLXArray(tokenArray).reshaped([1, -1])

        let voiceEmb: MLXArray
        if let refAudio {
            voiceEmb = refAudio
        } else {
            voiceEmb = try loadVoice(named: voiceName)
        }
        let refIdx = min(modelInputUnitCount, voiceEmb.shape[0] - 1)
        let refS = voiceEmb[refIdx..<(refIdx + 1)]

        // `generate(...)` currently consumes Kokoro speech rate from `self.speed`;
        // keep the timed sibling on the same inference seam for audio parity.
        let (audio, predDurArr, stats) = self.callAsFunctionWithStats(
            inputIds: inputIds,
            refS: refS,
            speed: speed
        )
        let audioFlat = audio.reshaped([-1])
        let predDur: [Int32] = predDurArr.asArray(Int32.self)

        let secondsPerFrame: Double
        do {
            secondsPerFrame = try Self.secondsPerFrame(
                sampleRate: sampleRate,
                hopSize: config.istftnet.genIstftHopSize,
                upsampleProduct: config.istftnet.upsampleRates.reduce(1, *)
            )
        } catch {
            Self.timingsLog.error("KokoroModel invalid timing scale: \(String(describing: error)); returning empty tokens")
            return KokoroTimedGeneration(audio: audioFlat, sampleRate: sampleRate, tokens: [])
        }

        return Self.materializeTimedGeneration(
            audio: audioFlat,
            sampleRate: sampleRate,
            predDur: predDur,
            mTokens: mTokens,
            timingSeeds: timingSeeds,
            stats: stats,
            secondsPerFrame: secondsPerFrame,
            modelInputUnitCount: modelInputUnitCount,
            originalUTF16Length: text.utf16.count,
            logTextPrefix: String(text.prefix(40))
        )
    }

    static func timingLanguageDisposition(
        voiceName: String,
        language: String?
    ) -> TimingLanguageDisposition {
        guard let resolved = (language ?? KokoroMultilingualProcessor.languageForVoice(voiceName))?.lowercased() else {
            return .emptyTokens
        }
        if resolved == "en" || resolved.hasPrefix("en-") {
            return .english(resolved)
        }
        return .emptyTokens
    }

    static func decideFallback(
        stats: PredDurDefenseStats,
        modelInputUnitCount: Int,
        timingUnitCount: Int,
        clipRatio: Double = 0.25
    ) -> FallbackDecision {
        if modelInputUnitCount != timingUnitCount { return .empty(reason: .unitCountMismatch) }
        if stats.nanReplacements > 0 { return .empty(reason: .nan) }
        let denom = max(modelInputUnitCount, 1)
        let ratio = Double(stats.clippedDurations) / Double(denom)
        if ratio > clipRatio { return .empty(reason: .clip) }
        return .useTokens
    }

    static func materializeTimedGeneration(
        audio: MLXArray,
        sampleRate: Int,
        predDur: [Int32],
        mTokens: [MToken],
        timingSeeds: [KokoroTimingSeed],
        stats: PredDurDefenseStats,
        secondsPerFrame: Double,
        modelInputUnitCount: Int,
        originalUTF16Length: Int,
        logTextPrefix: String
    ) -> KokoroTimedGeneration {
        guard secondsPerFrame.isFinite, secondsPerFrame > 0 else {
            Self.timingsLog.error("Kokoro timings fallback (invalid scale) text=\"\(logTextPrefix)\"")
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }

        let timingUnitCount = Self.modelUnitCount(in: mTokens)
        switch Self.decideFallback(
            stats: stats,
            modelInputUnitCount: modelInputUnitCount,
            timingUnitCount: timingUnitCount
        ) {
        case .useTokens:
            break
        case .empty(let reason):
            switch reason {
            case .nan:
                Self.timingsLog.warning(
                    "Kokoro timings fallback (NaN): nan=\(stats.nanReplacements) clipped=\(stats.clippedDurations) text=\"\(logTextPrefix)\""
                )
            case .clip:
                Self.timingsLog.warning(
                    "Kokoro timings fallback (clip): clipped=\(stats.clippedDurations)/\(modelInputUnitCount) text=\"\(logTextPrefix)\""
                )
            case .unitCountMismatch:
                Self.timingsLog.error(
                    "Kokoro timings fallback (unit mismatch): model=\(modelInputUnitCount) timing=\(timingUnitCount) text=\"\(logTextPrefix)\""
                )
            }
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }

        guard mTokens.count == timingSeeds.count else {
            Self.timingsLog.error(
                "Kokoro timings seed mismatch: timedTokens=\(mTokens.count) timingSeeds=\(timingSeeds.count) text=\"\(logTextPrefix)\""
            )
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }

        do {
            let tokens = try KokoroJoinTimestamps.materialize(
                predDur: predDur,
                tokens: mTokens,
                timingSeeds: timingSeeds,
                originalUTF16Length: originalUTF16Length,
                secondsPerFrame: secondsPerFrame
            )
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: tokens)
        } catch {
            Self.timingsLog.error("Kokoro timings join failed: \(String(describing: error))")
            return KokoroTimedGeneration(audio: audio, sampleRate: sampleRate, tokens: [])
        }
    }

    static func modelUnitCount(in tokens: [MToken]) -> Int {
        tokens.reduce(0) { partial, token in
            partial + ((token.phonemes ?? "") + token.whitespace).utf16.count
        }
    }

    static func secondsPerFrame(sampleRate: Int, hopSize: Int, upsampleProduct: Int) throws -> Double {
        let value = Double(hopSize * upsampleProduct) / Double(sampleRate)
        guard value.isFinite, value > 0 else {
            throw KokoroTimingError.invalidTimingScale(
                sampleRate: sampleRate,
                hopSize: hopSize,
                upsampleProduct: upsampleProduct
            )
        }
        return value
    }

    private func phonemizeEnglishForTimings(
        text: String,
        language: String?
    ) async throws -> (phonemized: String, tokens: [MToken], timingSeeds: [KokoroTimingSeed])? {
        if let multilingual = textProcessor as? KokoroMultilingualProcessor {
            return try multilingual.phonemizeEnglishForTimings(text: text, language: language)
        }
        if let misaki = textProcessor as? MisakiTextProcessor {
            return try misaki.phonemizeForTimings(text: text, language: language)
        }
        return nil
    }
}
