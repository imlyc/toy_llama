#include <iostream>
#include <string>

#include <glog/logging.h>

#include "chat_engine.h"
#include "model/model.h"
#include "model/model_loader.h"

namespace {
constexpr char kModelPath[] =
    "/Users/imlyc/Work/toy_llama/data/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
}  // namespace

int main(int argc, char** argv) {
  // FLAGS_logtostderr = 1;
  FLAGS_minloglevel = 0;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Welcome to Toy Llama.";

  tlm::ModelLoader loader;
  std::unique_ptr<tlm::Model> model = loader.Load(kModelPath);
  if (!model) {
    LOG(ERROR) << "Fail to load model at " << kModelPath;
  }

  LOG(INFO) << "Load model at " << kModelPath;

  tlm::ChatEngine chat_engine(*model);

  if (argc > 2) {
    std::cout << "Unsupported usage." << std::endl;
    return -1;
  }

  if (argc == 2) {
    std::string message(argv[1]);
    std::cout << "message: " << message << std::endl;
    std::string reply = chat_engine.SendMessage(message);
    std::cout << "reply: " << reply << std::endl;
    return 0;
  }

  while (true) {
    std::cout << "user: " << std::flush;

    std::string input;
    if (!std::getline(std::cin, input)) {
      break;
    }

    if (input.empty()) {
      continue;
    }

    std::cout << "assistant: " << std::flush;
    chat_engine.SendMessageAsync(input, [](const std::string& text) {
      std::cout << text << std::flush;
    });
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "assistant: Good bye!" << std::endl;
  return 0;
}
