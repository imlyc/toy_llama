#include <glog/logging.h>

#include "chat_engine.h"
#include "model_loader.h"
#include "model.h"

namespace {
constexpr char kModelPath[] =
  "/Users/imlyc/Work/toy_llama/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
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

  tlm::ChatEngine chat_engine(*model);

  std::string reply = chat_engine.SendMessage("Hello, how are you?");
  LOG(INFO) << ">>>>\n" << reply << "\n<<<<";

  return 0;
}
