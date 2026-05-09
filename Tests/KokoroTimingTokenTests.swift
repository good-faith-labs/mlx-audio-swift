import XCTest
@testable import MLXAudioTTS

final class KokoroTimingTokenTests: XCTestCase {
    func testIsPunctuationOnlyMatchesPunctuationSymbolsAndWhitespace() {
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly(","))
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly("."))
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly("!?"))
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly("—"))
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly(" "))
        XCTAssertTrue(KokoroTimingToken.isPunctuationOnly("\u{2014}"))
        XCTAssertFalse(KokoroTimingToken.isPunctuationOnly("Hello"))
        XCTAssertFalse(KokoroTimingToken.isPunctuationOnly("a"))
        XCTAssertFalse(KokoroTimingToken.isPunctuationOnly("don't"))
        XCTAssertFalse(KokoroTimingToken.isPunctuationOnly(""))
    }
}
