#include "tokenizer/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "model/token.h"

namespace tlm {
namespace {

class TokenizerTest : public ::testing::Test {
 protected:
  void Init(std::vector<std::string> tokens, std::vector<std::string> merges) {
    tokens_ = std::move(tokens);
    merges_ = std::move(merges);
    std::vector<std::string_view> token_views(tokens_.begin(), tokens_.end());
    std::vector<std::string_view> merge_views(merges_.begin(), merges_.end());
    tokenizer_ = std::make_unique<Tokenizer>(token_views, merge_views);
  }

  std::vector<std::string> tokens_;
  std::vector<std::string> merges_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

// ── Empty input ───────────────────────────────────────────────────────────────

TEST_F(TokenizerTest, EmptyTextReturnsEmpty) {
  Init({"a"}, {});
  EXPECT_TRUE(tokenizer_->TextToToken("").empty());
}

// ── ASCII: no merges ──────────────────────────────────────────────────────────

TEST_F(TokenizerTest, AsciiSingleChar) {
  Init({"a"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("a"), (std::vector<Token>{{0}}));
}

TEST_F(TokenizerTest, AsciiMultiCharNoMerge) {
  Init({"a", "b"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("ab"),
            (std::vector<Token>{{0}, {1}}));
}

// ── ASCII: BPE merges ─────────────────────────────────────────────────────────

TEST_F(TokenizerTest, AsciiSimpleMerge) {
  // "a b" merge causes "a"+"b" -> "ab" (token id 2)
  Init({"a", "b", "ab"}, {"a b"});
  EXPECT_EQ(tokenizer_->TextToToken("ab"), (std::vector<Token>{{2}}));
}

TEST_F(TokenizerTest, AsciiMergePriorityOrdering) {
  // Merges: "b c" rank 0, "a bc" rank 1.
  // BPE on "abc": ("b","c") rank 0 fires first -> "bc"; then ("a","bc") rank 1 -> "abc".
  Init({"a", "b", "c", "bc", "abc"}, {"b c", "a bc"});
  EXPECT_EQ(tokenizer_->TextToToken("abc"), (std::vector<Token>{{4}}));
}

TEST_F(TokenizerTest, AsciiChainedMerges) {
  // "a b"->0, "ab c"->1, "abc d"->2; fully collapses "abcd" into one token.
  Init({"a", "b", "c", "d", "ab", "abc", "abcd"},
       {"a b", "ab c", "abc d"});
  EXPECT_EQ(tokenizer_->TextToToken("abcd"), (std::vector<Token>{{6}}));
}

// ── Unknown token ─────────────────────────────────────────────────────────────

TEST_F(TokenizerTest, UnknownTokenReturnsInvalid) {
  Init({"a"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("x"), (std::vector<Token>{Token::INVALID}));
}

// ── Space / byte mapping ──────────────────────────────────────────────────────

TEST_F(TokenizerTest, SpaceMapsToGSym) {
  // Space (0x20) byte-maps to U+0120 (Ġ), UTF-8: 0xc4 0xa0.
  // In a UTF-8 source file the literal "Ġ" is exactly those two bytes.
  Init({"Ġ"}, {});
  EXPECT_EQ(tokenizer_->TextToToken(" "), (std::vector<Token>{{0}}));
}

TEST_F(TokenizerTest, SpacePreTokenizationSplitsWords) {
  // "ab cd": pre-tokenizer yields ["ab", " cd"].
  // " cd" byte-maps to "Ġcd" -> symbols [Ġ, c, d].
  Init({"a", "b", "Ġ", "c", "d"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("ab cd"),
            (std::vector<Token>{{0}, {1}, {2}, {3}, {4}}));
}

// ── UTF-8 multi-byte characters ───────────────────────────────────────────────

TEST_F(TokenizerTest, Utf8MultiByteChar) {
  // "é" = bytes 0xc3 0xa9.
  // 0xc3 (195) ∈ [0xae,0xff] -> UnicodeCodePointToUtf8(0xc3) = "Ã" (0xc3 0x83)
  // 0xa9 (169) ∈ [0xa1,0xac] -> UnicodeCodePointToUtf8(0xa9) = "©" (0xc2 0xa9)
  // BPE splits "Ã©" into two 2-byte UTF-8 symbols: "Ã" and "©".
  Init({"Ã", "©"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("é"),
            (std::vector<Token>{{0}, {1}}));
}

TEST_F(TokenizerTest, Utf8MultiByteCharWithMerge) {
  // Same byte mapping as above, but with a merge that collapses "Ã" + "©" into "Ã©".
  // Merge key: "Ã ©" (the two UTF-8 symbols separated by a literal space).
  Init({"Ã", "©", "Ã©"}, {"Ã ©"});
  EXPECT_EQ(tokenizer_->TextToToken("é"), (std::vector<Token>{{2}}));
}

// ── Pre-tokenization: contractions ───────────────────────────────────────────

TEST_F(TokenizerTest, ContractionsPreTokenizationSplit) {
  // "don't": pre-tokenizer yields ["don", "'t"].
  // Neither chunk crosses into the other, so tokens are from two independent BPEs.
  Init({"d", "o", "n", "'", "t"}, {});
  EXPECT_EQ(tokenizer_->TextToToken("don't"),
            (std::vector<Token>{{0}, {1}, {2}, {3}, {4}}));
}

// ── TokenToText ───────────────────────────────────────────────────────────────
// Vocab symbols use the byte-level BPE alphabet: space is stored as Ġ (U+0120,
// UTF-8 C4 A0), newline as Ċ (U+010A, UTF-8 C4 8A). Printable ASCII maps to
// itself.

TEST_F(TokenizerTest, TokenToTextPlainAscii) {
  Init({"hello", ",", "\xC4\xA0world"}, {});   // "hello", ",", "Ġworld"
  EXPECT_EQ(tokenizer_->TokenToText({{0}, {1}, {2}}), "hello, world");
}

TEST_F(TokenizerTest, TokenToTextSpaceAndNewline) {
  Init({"\xC4\xA0hi", "\xC4\x8A"}, {});        // "Ġhi", "Ċ"
  EXPECT_EQ(tokenizer_->TokenToText({{0}, {1}}), " hi\n");
}

// Special-token strings are plain printable ASCII and pass through verbatim.
TEST_F(TokenizerTest, TokenToTextSpecialTokenString) {
  Init({"<|eot_id|>"}, {});
  EXPECT_EQ(tokenizer_->TokenToText({{0}}), "<|eot_id|>");
}

// A multi-byte character split across two tokens must reassemble: the bytes of
// 🙂 (U+1F642, UTF-8 F0 9F 99 82) map to codepoints {U+00F0, U+0141} and
// {U+013B, U+0124}, i.e. the two vocab symbols below. Decoding must
// concatenate raw bytes across tokens, not decode each token as text.
TEST_F(TokenizerTest, TokenToTextMultibyteSplitAcrossTokens) {
  Init({"\xC3\xB0\xC5\x81", "\xC4\xBB\xC4\xA4"}, {});
  EXPECT_EQ(tokenizer_->TokenToText({{0}, {1}}), "\xF0\x9F\x99\x82");  // 🙂
}

TEST_F(TokenizerTest, TokenToTextInvalidIdGivesUnknown) {
  Init({"a"}, {});
  EXPECT_EQ(tokenizer_->TokenToText({{5}}), "<|unknown|>");
  EXPECT_EQ(tokenizer_->TokenToText({{-1}}), "<|unknown|>");
}

TEST_F(TokenizerTest, TokenToTextEmpty) {
  Init({"a"}, {});
  EXPECT_EQ(tokenizer_->TokenToText({}), "");
}

// Encode then decode restores the original text, spaces and all.
TEST_F(TokenizerTest, RoundTripAsciiWithSpaces) {
  Init({"h", "i", "\xC4\xA0", "\xC4\xA0h", "\xC4\xA0hi"},
       {"\xC4\xA0 h", "\xC4\xA0h i"});   // merges: "Ġ h" then "Ġh i"
  const std::string text = "hi hi";
  EXPECT_EQ(tokenizer_->TokenToText(tokenizer_->TextToToken(text)), text);
}

}  // namespace
}  // namespace tlm
