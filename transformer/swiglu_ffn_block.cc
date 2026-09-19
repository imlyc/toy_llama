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
  MutableMatrixView up = up_.Matrix().Top(input.shape[0]);
  MutableMatrixView gate = gate_.Matrix().Top(input.shape[0]);
  MutableMatrixView output = output_.Matrix().Top(input.shape[0]);

  compute_.MatMulT(up, input, wup_);
  compute_.MatMulT(gate, input, wgate_);
  compute_.SwiGluMul(gate, up);
  compute_.MatMulT(output, gate, wdown_);
  return output;
}

}  // namespace tlm
