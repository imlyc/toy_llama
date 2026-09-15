// End-to-end golden + performance test against a real GGUF model, driven
// through the public ChatEngine API.
//
// Sends a fixed chat message and either records or checks a golden:
//
//   TOY_LLAMA_MODEL=<path.gguf>       required; the test is skipped otherwise.
//   TOY_LLAMA_GOLDEN=<path.txt>       optional. Missing file: record this run.
//                                     Existing file: compare against it.
//   TOY_LLAMA_MIN_PREFILL_SPEEDUP=x   optional, default 1.2. In compare mode,
//                                     golden_prefill_ms / prefill_ms must be
//                                     at least this.
//
// Workflow for a change that must not alter inference output but should make
// prompt processing faster (e.g. batched prefill): record the golden on the
// commit BEFORE the change, then run again on the commit AFTER it.
//
// Timing comes from the streaming callback: the gap from SendMessageAsync's
// start to the first callback is prompt processing (prefill + one sample);
// every later gap is one decode step. Timings are only meaningful on an
// optimized build (build-release).

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "chat_engine.h"
#include "model/model.h"
#include "model/model_loader.h"
#include "tensor/tensor_view.h"

namespace tlm {
namespace {

constexpr double kDefaultMinPrefillSpeedup = 1.2;

// About 100 tokens after the chat template: long enough that prompt
// processing is many times the cost of one decode step.
constexpr char kMessage[] =
    "Here is a short passage. Read it and answer the question at the end.\n\n"
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. He "
    "carried a brass lantern, a tin of matches, and a logbook in which he "
    "recorded the weather, the ships that passed, and the birds that rested on "
    "the railing. On stormy nights the beam swept the water in slow circles, "
    "and fishing boats used it to find the harbor mouth.\n\n"
    "Question: What three things did the keeper carry up the stairs?";

struct RunRecord {
  std::string model;
  int generated_token_count = 0;
  std::string reply;
  double prefill_ms = 0;
  double decode_ms_per_token = 0;
};

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// The reply is stored on one line with newlines and backslashes escaped.
std::string Escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

std::string Unescape(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      out += (s[i + 1] == 'n') ? '\n' : s[i + 1];
      ++i;
    } else {
      out += s[i];
    }
  }
  return out;
}

// Golden file: one "key value..." record per line.
void WriteRecord(const std::filesystem::path& path, const RunRecord& r) {
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open()) << "cannot write " << path;
  out << "model " << r.model << "\n";
  out << "generated_token_count " << r.generated_token_count << "\n";
  out << "reply " << Escape(r.reply) << "\n";
  out << "prefill_ms " << r.prefill_ms << "\n";
  out << "decode_ms_per_token " << r.decode_ms_per_token << "\n";
}

bool ReadRecord(const std::filesystem::path& path, RunRecord* r) {
  std::ifstream in(path);
  if (!in.is_open()) return false;
  std::string key;
  while (in >> key) {
    if (key == "model") {
      in >> r->model;
    } else if (key == "generated_token_count") {
      in >> r->generated_token_count;
    } else if (key == "reply") {
      std::string line;
      std::getline(in, line);
      if (!line.empty() && line[0] == ' ') line.erase(0, 1);
      r->reply = Unescape(line);
    } else if (key == "prefill_ms") {
      in >> r->prefill_ms;
    } else if (key == "decode_ms_per_token") {
      in >> r->decode_ms_per_token;
    } else {
      return false;
    }
  }
  return true;
}

// Fault every page of the (mmap'd) weights into the page cache before timing,
// so the first forward pass measures compute rather than disk reads.
void TouchWeights(const Model& model) {
  volatile std::byte sink{};
  auto touch = [&sink](auto view) {
    std::span<const std::byte> bytes = view.template As<const std::byte>();
    for (size_t i = 0; i < bytes.size(); i += 4096) sink = bytes[i];
  };
  touch(model.GetTokenEmbeddings());
  touch(model.GetOutputNormGamma());
  touch(model.GetRopeFreqsWeight());
  for (int i = 0; i < model.GetDecoderBlockCount(); ++i) {
    touch(model.GetDecoderBlockAttnNormWeight(i));
    touch(model.GetDecoderBlockAttnQWeight(i));
    touch(model.GetDecoderBlockAttnKWeight(i));
    touch(model.GetDecoderBlockAttnVWeight(i));
    touch(model.GetDecoderBlockAttnOWeight(i));
    touch(model.GetDecoderBlockFfnNormWeight(i));
    touch(model.GetDecoderBlockFfnUpWeight(i));
    touch(model.GetDecoderBlockFfnGateWeight(i));
    touch(model.GetDecoderBlockFfnDownWeight(i));
  }
}

TEST(E2eTest, GoldenOutputAndPrefillSpeed) {
  const char* model_path = std::getenv("TOY_LLAMA_MODEL");
  if (model_path == nullptr) {
    GTEST_SKIP() << "Set TOY_LLAMA_MODEL=<path.gguf> to run the e2e test.";
  }

  ModelLoader loader;
  std::unique_ptr<Model> model = loader.Load(model_path);
  ASSERT_NE(model, nullptr) << "failed to load " << model_path;

  ChatEngine chat_engine(*model);
  // Warm the weights AFTER the engine has allocated its KV cache: on a large
  // context that allocation can evict the freshly faulted weight pages, which
  // would put disk reads inside the timed prefill.
  TouchWeights(*model);

  // One timestamp per streamed token. The first marks the end of prompt
  // processing; each later one marks the end of a decode step. The final
  // decode step (the one that produces EOS) has no callback, so it is bounded
  // by the return of SendMessageAsync instead.
  RunRecord actual;
  actual.model = std::filesystem::path(model_path).filename().string();
  std::vector<Clock::time_point> token_times;
  const Clock::time_point start = Clock::now();
  chat_engine.SendMessageAsync(kMessage, [&](const std::string& text) {
    token_times.push_back(Clock::now());
    actual.reply += text;
  });
  const Clock::time_point end = Clock::now();

  ASSERT_FALSE(token_times.empty()) << "model produced no tokens";
  actual.generated_token_count = static_cast<int>(token_times.size());
  actual.prefill_ms = ElapsedMs(start, token_times.front());
  // After the first token there are exactly `generated_token_count` decode
  // steps: one per further streamed token, plus the one that yields EOS.
  actual.decode_ms_per_token =
      ElapsedMs(token_times.front(), end) / actual.generated_token_count;

  std::cout << "model:            " << actual.model << "\n"
            << "prefill:          " << actual.prefill_ms << " ms\n"
            << "decode:           " << actual.decode_ms_per_token
            << " ms/token over " << actual.generated_token_count
            << " steps\n"
            << "reply (" << actual.generated_token_count << " tokens): "
            << actual.reply << "\n";

  const char* golden_path = std::getenv("TOY_LLAMA_GOLDEN");
  if (golden_path == nullptr) {
    std::cout << "TOY_LLAMA_GOLDEN not set: nothing to compare.\n";
    return;
  }

  if (!std::filesystem::exists(golden_path)) {
    WriteRecord(golden_path, actual);
    std::cout << "Recorded golden at " << golden_path << "\n";
    return;
  }

  RunRecord golden;
  ASSERT_TRUE(ReadRecord(golden_path, &golden)) << "malformed " << golden_path;

  // 1. Inference output must be unchanged.
  EXPECT_EQ(golden.model, actual.model);
  EXPECT_EQ(golden.generated_token_count, actual.generated_token_count);
  EXPECT_EQ(golden.reply, actual.reply);

  // 2. Prompt processing must be faster than the golden by the required factor.
  double min_speedup = kDefaultMinPrefillSpeedup;
  if (const char* s = std::getenv("TOY_LLAMA_MIN_PREFILL_SPEEDUP")) {
    min_speedup = std::atof(s);
  }
  const double prefill_speedup = golden.prefill_ms / actual.prefill_ms;
  const double decode_ratio =
      golden.decode_ms_per_token / actual.decode_ms_per_token;
  std::cout << "prefill speedup vs golden: " << prefill_speedup << "x ("
            << golden.prefill_ms << " ms -> " << actual.prefill_ms
            << " ms)\n"
            << "decode speedup vs golden:  " << decode_ratio << "x ("
            << golden.decode_ms_per_token << " -> "
            << actual.decode_ms_per_token << " ms/token)\n";
  EXPECT_GE(prefill_speedup, min_speedup)
      << "prefill did not get faster: golden " << golden.prefill_ms
      << " ms, now " << actual.prefill_ms << " ms";
}

}  // namespace
}  // namespace tlm
