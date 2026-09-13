#include "transformer/swiglu_ffn_block.h"

#include "compute/compute_engine.h"
#include "compute/storage.h"

namespace tlm {

SwiGluFfnBlock::SwiGluFfnBlock(ComputeEngine& compute, const Param& param)
    : compute_(compute),
      wup_(param.wup),
      wgate_(param.wgate),
      wdown_(param.wdown),
      up_(compute_, wup_.shape[0]),
      gate_(compute_, wgate_.shape[0]),
      output_(compute_, wdown_.shape[0]) {}
SwiGluFfnBlock::~SwiGluFfnBlock() = default;

SwiGluFfnBlock::SwiGluFfnBlock(SwiGluFfnBlock&&) = default;

MatrixView SwiGluFfnBlock::Forward(MatrixView input) {
  up_.Reset(input.shape[0]);
  gate_.Reset(input.shape[0]);
  output_.Reset(input.shape[0]);

  compute_.MatMulT(up_.Matrix(), input, wup_);
  compute_.MatMulT(gate_.Matrix(), input, wgate_);
  compute_.SwiGluMul(gate_.Matrix(), up_.Matrix().View());
  compute_.MatMulT(output_.Matrix(), gate_.Matrix().View(), wdown_);
  return output_.Matrix().View();
}

}  // namespace tlm
