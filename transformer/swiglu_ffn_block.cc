#include "transformer/swiglu_ffn_block.h"

#include "compute/compute_engine.h"

namespace tlm {

SwiGluFfnBlock::SwiGluFfnBlock(ComputeEngine& compute, const Param& param)
    : compute_(compute),
      wup_(param.wup),
      wgate_(param.wgate),
      wdown_(param.wdown),
      up_(compute_.AllocMatrix(wup_.shape[0])),
      gate_(compute_.AllocMatrix(wgate_.shape[0])),
      output_(compute_.AllocMatrix(wdown_.shape[0])) {}
SwiGluFfnBlock::~SwiGluFfnBlock() = default;

SwiGluFfnBlock::SwiGluFfnBlock(SwiGluFfnBlock&&) = default;

MatrixView SwiGluFfnBlock::Forward(MatrixView input) {
  MutableMatrixView up = up_.View().Top(input.shape[0]);
  MutableMatrixView gate = gate_.View().Top(input.shape[0]);
  MutableMatrixView output = output_.View().Top(input.shape[0]);

  compute_.MatMulT(up, input, wup_);
  compute_.MatMulT(gate, input, wgate_);
  compute_.SwiGluMul(gate, up);
  compute_.MatMulT(output, gate, wdown_);
  return output;
}

}  // namespace tlm
