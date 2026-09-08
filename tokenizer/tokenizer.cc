#include "tokenizer/tokenizer.h"

#include <limits>
#include <queue>
#include <sstream>
#include <string_view>
#include <vector>

#include <glog/logging.h>

#include <re2/re2.h>

#include "model/model.h"

namespace tlm {
namespace {
constexpr int kInvalidRank = std::numeric_limits<int>::max();
constexpr char kUnknownToken[] = "<|unknown|>";

const RE2 kWordSplitter(
    "("
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    // "|\\s+(?!\\S)" <--- Not supported by RE2.
    "|\\s+"
    ")",
    RE2::DefaultOptions);

std::vector<std::string> PreTokenize(const std::string& text) {
  std::vector<std::string> chunks;
  re2::StringPiece input(text);
  re2::StringPiece match;
  while (RE2::FindAndConsume(&input, kWordSplitter, &match)) {
    chunks.emplace_back(match);
  }
  return chunks;
}

int Utf8CharLength(char c) {
  uint8_t uc = static_cast<uint8_t>(c);
  // 0xxx xxxx
  if ((uc & 0x80) == 0) return 1;
  // 110x xxxx
  if ((uc & 0xe0) == 0xc0) return 2;
  // 1110 xxxx
  if ((uc & 0xf0) == 0xe0) return 3;
  // 1111 0xxx
  if ((uc & 0xf8) == 0xf0) return 4;

  return 1;
}

std::string UnicodeCodePointToUtf8(uint32_t code_point) {
  std::string ret;
  if (code_point <= 0x7f) {
    // 0xxx xxxx
    ret.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7ff) {
    // 110x xxxx 10xx xxxx
    ret.push_back(0xc0 | ((code_point >> 6) & 0x1f));
    ret.push_back(0x80 | (code_point & 0x3f));
  } else if (code_point <= 0xffff) {
    // 1110 xxxx 10xx xxxx 10xx xxxx
    ret.push_back(0xe0 | ((code_point >> 12) & 0x0f));
    ret.push_back(0x80 | ((code_point >> 6) & 0x3f));
    ret.push_back(0x80 | (code_point & 0x3f));
  } else if (code_point <= 0x10ffff) {
    // 1111 0xxx 10xx xxxx 10xx xxxx 10xx xxxx
    ret.push_back(0xf0 | ((code_point >> 18) & 0x7));
    ret.push_back(0x80 | ((code_point >> 12) & 0x3f));
    ret.push_back(0x80 | ((code_point >> 6) & 0x3f));
    ret.push_back(0x80 | (code_point & 0x3f));
  } else {
    LOG(ERROR) << "Invalid unicode code point " << std::hex << code_point;
  }
  return ret;
}

std::array<std::string, 256> BuildByteToUtf8Map() {
  std::array<std::string, 256> m;

  // Printable characters.
  for (uint32_t b = 0x21; b <= 0x7e; b++) {
    m[b] = UnicodeCodePointToUtf8(b);
    CHECK(!m[b].empty());
  }
  for (uint32_t b = 0xa1; b <= 0xac; b++) {
    m[b] = UnicodeCodePointToUtf8(b);
    CHECK(!m[b].empty());
  }
  for (uint32_t b = 0xae; b <= 0xff; b++) {
    m[b] = UnicodeCodePointToUtf8(b);
    CHECK(!m[b].empty());
  }

  // Others.
  uint32_t next_code_point = 0x100;
  for (uint32_t b = 0; b < 256; b++) {
    if (m[b].empty()) {
      m[b] = UnicodeCodePointToUtf8(next_code_point);
      next_code_point++;
    }
  }

  return m;
}

std::unordered_map<std::string, int> BuildUtf8ToByteMap(
    const std::array<std::string, 256>& byte_to_utf8) {
  std::unordered_map<std::string, int> utf8_to_byte;
  for (int i = 0; i < byte_to_utf8.size(); i++) {
    const std::string& utf8 = byte_to_utf8[i];
    utf8_to_byte[utf8] = i;
  }
  return utf8_to_byte;
}

std::unordered_map<std::string_view, int> BuildMergeToRankMap(
    const std::vector<std::string_view>& merges) {
  std::unordered_map<std::string_view, int> merge_to_rank;
  int rank = 0;
  for (const auto& m : merges) {
    merge_to_rank.emplace(m, rank);
    rank++;
  }
  return merge_to_rank;
}

std::unordered_map<std::string_view, Token> BuildSymbolToTokenMap(
    const std::vector<std::string_view>& tokens) {
  std::unordered_map<std::string_view, Token> symbol_to_token;
  int id = 0;
  for (const auto& t : tokens) {
    symbol_to_token.emplace(t, Token{.id = id});
    id++;
  }
  return symbol_to_token;
}

struct Symbol {
  std::string_view text;
  int prev = -1;
  int next = -1;
  bool deleted = false;
};

struct Merge {
  int index = -1;
  int rank = -1;

  // Smaller rank value is larger.
  bool operator<(const Merge& other) const {
    if (rank == other.rank) {
      return index > other.index;
    }

    return rank > other.rank;
  }
};
}  // namespace

Tokenizer::Tokenizer(const Model& model)
  : Tokenizer(model.GetTokenizerTokens(), model.GetTokenizerMerges()) {
}

Tokenizer::Tokenizer(const std::vector<std::string_view>& tokens,
                     const std::vector<std::string_view>& merges)
  : token_to_symbol_(tokens),
    byte_to_utf8_(BuildByteToUtf8Map()),
    utf8_to_byte_(BuildUtf8ToByteMap(byte_to_utf8_)),
    merge_to_rank_(BuildMergeToRankMap(merges)),
    symbol_to_token_(BuildSymbolToTokenMap(tokens)) {
}

Tokenizer::~Tokenizer() = default;

std::vector<Token> Tokenizer::TextToToken(const std::string& text) {
  std::vector<Token> tokens;
  std::vector<std::string> chunks = PreTokenize(text);

  for (const std::string& chunk : chunks) {
    std::string mapped_chunk;
    for (char c : chunk) {
      mapped_chunk.append(MapByteToUtf8(c));
    }

    BpeEncode(tokens, mapped_chunk);
  }

  return tokens;
}

std::string Tokenizer::TokenToText(const std::vector<Token>& tokens) {
  std::string text;
  for (const auto& token : tokens) {
    if (token.id < 0 || token.id >= token_to_symbol_.size()) {
      text.append(kUnknownToken);
      continue;
    }

    std::string_view symbol = token_to_symbol_[token.id];
    int c = 0;
    while (c < symbol.size()) {
      int utf8_len = Utf8CharLength(symbol[c]);
      // unordered_map<std::string, int>::find(std::string_view) can't compile.
      // Has to use std::string here.
      std::string utf8(symbol.substr(c, utf8_len));
      auto it = utf8_to_byte_.find(utf8);
      if (it != utf8_to_byte_.end()) {
        text.push_back(static_cast<char>(it->second));
      } else {
        text.append(kUnknownToken);
      }
      c += utf8_len;
    }
  }
  return text;
}

std::string Tokenizer::MapByteToUtf8(char c) {
  return byte_to_utf8_[static_cast<uint8_t>(c)];
}

void Tokenizer::BpeEncode(std::vector<Token>& output, const std::string& str) {
  std::vector<Symbol> symbols;
  int c = 0;
  while (c < str.size()) {
    int utf8_len = Utf8CharLength(str[c]);
    CHECK_LE(c + utf8_len, str.size());
    int current = static_cast<int>(symbols.size());
    Symbol symbol{.text = std::string_view(&str[c], utf8_len),
                  .prev = current - 1,
                  .next = current + 1};
    symbols.emplace_back(symbol);
    c += utf8_len;
  }

  std::priority_queue<Merge> merge_queue;
  for (int i = 0; i < symbols.size() - 1; i++) {
    std::string_view current = symbols[i].text;
    std::string_view next = symbols[i + 1].text;
    int rank = GetRank(current, next);
    if (rank != kInvalidRank) {
      Merge symbol_ptr{.index = i, .rank = rank};
      merge_queue.emplace(symbol_ptr);
    }
  }

  while (!merge_queue.empty()) {
    Merge merge = merge_queue.top();
    merge_queue.pop();

    Symbol& current = symbols[merge.index];
    if (current.deleted) {
      continue;
    }

    if (current.next == symbols.size()) {
      continue;
    }

    Symbol& next = symbols[current.next];
    if (next.deleted) {
      continue;
    }

    int rank = GetRank(current.text, next.text);
    if (rank != merge.rank) {
      // The new Merge is already enqueued in a previous merge, skip this one.
      continue;
    }

    // Merge symbols.
    current.text = std::string_view(current.text.data(),
                                    current.text.size() + next.text.size());
    current.next = next.next;
    next.deleted = true;
    if (current.next != symbols.size()) {
      Symbol& new_next = symbols[current.next];
      new_next.prev = merge.index;

      int new_rank = GetRank(current.text, new_next.text);
      if (new_rank != kInvalidRank) {
        Merge merge_next{.index = merge.index, .rank = new_rank};
        merge_queue.emplace(merge_next);
      }
    }

    if (current.prev != -1) {
      Symbol& prev = symbols[current.prev];
      int new_rank = GetRank(prev.text, current.text);
      if (new_rank != kInvalidRank) {
        Merge merge_prev{.index = current.prev, .rank = new_rank};
        merge_queue.emplace(merge_prev);
      }
    }
  }

  // We always merge right into left, so the first element is always valid.
  int symbol_index = 0;
  while (symbol_index < symbols.size()) {
    const Symbol& current = symbols[symbol_index];
    CHECK(!current.deleted);
    output.emplace_back(GetToken(current.text));
    symbol_index = current.next;
  }
}

int Tokenizer::GetRank(std::string_view current, std::string_view next) {
  std::string key;
  key.append(current);
  key.append(" ");
  key.append(next);
  auto it = merge_to_rank_.find(key);
  return it != merge_to_rank_.end() ? it->second : kInvalidRank;
}

Token Tokenizer::GetToken(std::string_view str) {
  auto it = symbol_to_token_.find(str);
  return it != symbol_to_token_.end() ? it->second : Token::INVALID;
}
}  // namespace tlm
