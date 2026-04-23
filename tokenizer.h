#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "token.h"

namespace tlm {
class Model;

class Tokenizer {
 public:
  explicit Tokenizer(const Model& model);
  ~Tokenizer();

  std::vector<Token> TextToToken(const std::string& text);
  std::string TokenToText(const std::vector<Token>& tokens);

 private:
  std::string MapByteToUtf8(char c);

  // Get rank of merge if exists. Otherwise return kInvalidRank.
  int GetRank(std::string_view current, std::string_view next);

  Token GetToken(std::string_view text);

  // BPE
  void BpeEncode(std::vector<Token>& output, const std::string& str);

  std::array<std::string, 256> byte_to_utf8_;
  std::unordered_map<std::string_view, int> merge_to_rank_;
  std::unordered_map<std::string_view, Token> symbol_to_token_;
};

}  // namespace tlm
