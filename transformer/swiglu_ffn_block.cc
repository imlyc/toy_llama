#include "transformer/swiglu_ffn_block.h"

#include "compute/compute_engine.h"
#include "compute/storage.h"

namespace tlm {

SwiGluFfnBlock::SwiGluFfnBlock(ComputeEngine& compute) : compute_(compute) {
  up_storage_ = compute_.Alloc(wup_.shape[0]);
  up_ = up_storage_->AsVector(wup_.shape[0]);
  gate_storage_ = compute_.Alloc(wgate_.shape[0]);
  gate_ = gate_storage_->AsVector(wgate_.shape[0]);
  output_storage_ = compute_.Alloc(wdown_.shape[0]);
  output_ = output_storage_->AsVector(wdown_.shape[0]);
}
SwiGluFfnBlock::~SwiGluFfnBlock() = default;

SwiGluFfnBlock::SwiGluFfnBlock(SwiGluFfnBlock&&) = default;

VectorView SwiGluFfnBlock::Forward(VectorView input) {
  compute_.MatMul(up_, wup_, input);
  compute_.MatMul(gate_, wgate_, input);
  compute_.SwiGluMul(gate_, up_.View());
  compute_.MatMul(output_, wdown_, gate_.View());
  return output_.View();
}

}  // namespace tlm
