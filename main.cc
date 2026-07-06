#include <glog/logging.h>

#include "chat_engine.h"
#include "model/model.h"
#include "model/model_loader.h"

namespace {
constexpr char kModelPath[] =
    "/Users/imlyc/Work/toy_llama/data/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
}  // namespace

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
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

  std::string message;
  if (argc == 2) {
    message = std::string(argv[1]);
  } else {
    message = "Hello, how are you?";
  }

  std::string reply = chat_engine.SendMessage(message);
  LOG(INFO) << ">>>>\n" << reply << "\n<<<<";

  return 0;
}
