#include "transformer/decoder_block.h"

namespace tlm {

DecoderBlock::DecoderBlock() = default;
DecoderBlock::~DecoderBlock() = default;

VectorView DecoderBlock::Forward(VectorView input) {
  return {};
}

}  // namespace tlm
