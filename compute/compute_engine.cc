#include "compute/compute_engine.h"

#include <cmath>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "compute/storage.h"
#include "tensor/dtype.h"

namespace tlm {
using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::Ref;
using Eigen::VectorXf;

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename M>
using StrideMap = Eigen::Map<M, 0, Eigen::OuterStride<>>;

namespace {
inline void SoftmaxHelper(Ref<VectorXf> vec) {
  vec.array() -= vec.maxCoeff();
  vec.array() = vec.array().exp();
  vec /= vec.sum();
}

inline void MaskedSoftmax(Ref<RowMatrixXf> mat) {
  CHECK_LE(mat.rows(), mat.cols());
  for (int64_t r = 0; r < mat.rows(); r++) {
    auto row = mat.row(r);
    SoftmaxHelper(row.head(mat.cols() - mat.rows() + r + 1).transpose());
    row.tail(mat.rows() - r - 1).setZero();
  }
}

template <bool Mutable, typename T>
using MaybeConst = std::conditional_t<Mutable, T, const T>;

template <bool Mutable>
inline Map<MaybeConst<Mutable, VectorXf>> CreateVectorXf(
    TensorView<1, Mutable> view) {
  CHECK_EQ(view.dtype, DType::F32);
  using Float = MaybeConst<Mutable, float>;
  std::span<Float> data = view.template As<Float>();
  Map<MaybeConst<Mutable, VectorXf>> vec(data.data(), data.size());
  return vec;
}

template <bool Mutable>
inline Map<MaybeConst<Mutable, RowMatrixXf>> CreateRowMatrixXf(
    TensorView<2, Mutable> view) {
  CHECK_EQ(view.dtype, DType::F32);
  using Float = MaybeConst<Mutable, float>;
  std::span<Float> data = view.template As<Float>();
  Map<MaybeConst<Mutable, RowMatrixXf>> mat(data.data(), view.shape[0],
                                         view.shape[1]);
  return mat;
}

void MatMulF32(MutableVectorView out, MatrixView lhs, VectorView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::F32);
  Map<VectorXf> out_vec = CreateVectorXf(out);
  Map<const RowMatrixXf> lhs_mat = CreateRowMatrixXf(lhs);
  Map<const VectorXf> rhs_vec = CreateVectorXf(rhs);
  out_vec.noalias() = lhs_mat * rhs_vec;
}

void MatMulQ8_0(MutableVectorView out, MatrixView lhs, VectorView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::Q8_0);
  CHECK_EQ(rhs.dtype, DType::F32);
  std::span<float> out_data = out.As<float>();
  std::span<const BlockQ8_0> lhs_data = lhs.As<const BlockQ8_0>();
  std::span<const float> rhs_data = rhs.As<const float>();

  const int64_t block_count_per_row = lhs.shape[1] / sizeof(BlockQ8_0::data);

  for (int64_t row = 0; row < lhs.shape[0]; row++) {
    float acc = 0;
    for (int64_t block = 0; block < block_count_per_row; block++) {
      // Block by block dot product, instead of element by element product. This
      // significantly improves the speed.
      const BlockQ8_0& block_data = lhs_data[row * block_count_per_row + block];
      Map<const Eigen::Vector<int8_t, sizeof(BlockQ8_0::data)>> row_data(
          block_data.data);
      Map<const Eigen::Vector<float, sizeof(BlockQ8_0::data)>> col_data(
          rhs_data.data() + sizeof(BlockQ8_0::data) * block);
      acc += block_data.scale * row_data.cast<float>().dot(col_data);
    }
    out_data[row] = acc;
  }
}

void MatMulTF32(MutableMatrixView out, MatrixView lhs, MatrixView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::F32);
  Map<RowMatrixXf> out_mat = CreateRowMatrixXf(out);
  Map<const RowMatrixXf> lhs_mat = CreateRowMatrixXf(lhs);
  Map<const RowMatrixXf> rhs_mat = CreateRowMatrixXf(rhs);
  out_mat.noalias() = lhs_mat * rhs_mat.transpose();
}

void MatMulTQ8_0(MutableMatrixView out, MatrixView lhs, MatrixView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::Q8_0);
  Map<RowMatrixXf> out_mat = CreateRowMatrixXf(out);
  Map<const RowMatrixXf> lhs_mat = CreateRowMatrixXf(lhs);
  std::span<const BlockQ8_0> rhs_data = rhs.As<const BlockQ8_0>();
  VectorXf acc(lhs.shape[0]);
  Eigen::Vector<float, sizeof(BlockQ8_0::data)> scaled_row;

  const int64_t block_count_per_row = rhs.shape[1] / sizeof(BlockQ8_0::data);
  // transpose(rhs) is handled by constructing a row major vector.
  for (int64_t row = 0; row < rhs.shape[0]; row++) {
    acc.setZero();
    for (int64_t block = 0; block < block_count_per_row; block++) {
      const BlockQ8_0& block_data = rhs_data[row * block_count_per_row + block];
      // 32 x 1
      Map<const Eigen::Vector<int8_t, sizeof(BlockQ8_0::data)>> row_data(
          block_data.data);
      scaled_row =
          row_data.cast<float>() * static_cast<float>(block_data.scale);
      acc.noalias() += lhs_mat.middleCols<sizeof(BlockQ8_0::data)>(
                           block * sizeof(BlockQ8_0::data)) *
                       scaled_row;
    }
    out_mat.col(row) = acc;
  }
}

void Dequant(MutableVectorView out, VectorView in) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(in.dtype, DType::Q8_0);
  CHECK_EQ(out.shape[0], in.shape[0]);
  CHECK_EQ(in.shape[0] % sizeof(BlockQ8_0::data), 0);

  std::span<float> out_data = out.As<float>();
  std::span<const BlockQ8_0> in_data = in.As<const BlockQ8_0>();
  for (size_t b = 0; b < in_data.size(); b++) {
    const BlockQ8_0& block = in_data[b];
    for (size_t i = 0; i < sizeof(BlockQ8_0::data); i++) {
      out_data[b * sizeof(BlockQ8_0::data) + i] = block.scale * block.data[i];
    }
  }
}
}  // namespace

ComputeEngine::ComputeEngine() = default;
ComputeEngine::~ComputeEngine() = default;

std::unique_ptr<Storage> ComputeEngine::Alloc(int64_t size) {
  return std::make_unique<Storage>(size);
}

void ComputeEngine::Copy(MutableVectorView dst, VectorView src) {
  CHECK_EQ(dst.dtype, DType::F32);
  CHECK_EQ(dst.shape[0], src.shape[0]);

  switch (src.dtype) {
    case DType::F32:
      dst.CopyFrom(src);
      break;

    case DType::Q8_0:
      Dequant(dst, src);
      break;

    default:
      LOG(FATAL) << "Unsupported src type " << src.dtype;
  }
}

void ComputeEngine::Add(MutableMatrixView out, MatrixView lhs, MatrixView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::F32);
  CHECK_EQ(out.shape[0], lhs.shape[0]);
  CHECK_EQ(out.shape[1], lhs.shape[1]);
  CHECK_EQ(out.shape[0], rhs.shape[0]);
  CHECK_EQ(out.shape[1], rhs.shape[1]);
  Map<RowMatrixXf> out_vec = CreateRowMatrixXf(out);
  out_vec = CreateRowMatrixXf(lhs) + CreateRowMatrixXf(rhs);
}

void ComputeEngine::MatMul(MutableVectorView out,
                           MatrixView lhs,
                           VectorView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::F32);
  CHECK_EQ(lhs.shape[1], rhs.shape[0]);
  CHECK_EQ(out.shape[0], lhs.shape[0]);

  switch (lhs.dtype) {
    case DType::F32:
      MatMulF32(out, lhs, rhs);
      break;

    case DType::Q8_0:
      MatMulQ8_0(out, lhs, rhs);
      break;

    default:
      LOG(FATAL) << "Unsupported mat mul dtype " << lhs.dtype;
  }
}

void ComputeEngine::MatMulT(MutableMatrixView out,
                            MatrixView lhs,
                            MatrixView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(lhs.shape[1], rhs.shape[1]);
  CHECK_EQ(out.shape[0], lhs.shape[0]);

  switch (rhs.dtype) {
    case DType::F32:
      MatMulTF32(out, lhs, rhs);
      break;

    case DType::Q8_0:
      if (out.shape[0] == 1) {
        MatMulQ8_0(out.At(0), rhs, lhs.At(0));
      } else {
        MatMulTQ8_0(out, lhs, rhs);
      }
      break;

    default:
      LOG(FATAL) << "Unsupported mat mul dtype " << rhs.dtype;
  }
}

void ComputeEngine::Attn(MutableMatrixView out,
                         MatrixView q,
                         MatrixView k,
                         MatrixView v,
                         int64_t head_count_q,
                         int64_t head_count_kv) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(q.dtype, DType::F32);
  CHECK_EQ(k.dtype, DType::F32);
  CHECK_EQ(v.dtype, DType::F32);
  CHECK_EQ(head_count_q % head_count_kv, 0);
  CHECK_GE(head_count_q, head_count_kv);
  CHECK_EQ(q.shape[1] % head_count_q, 0);
  CHECK_EQ(k.shape[1] % head_count_kv, 0);
  CHECK_EQ(v.shape[1] % head_count_kv, 0);

  const int64_t q_per_kv = head_count_q / head_count_kv;
  const int64_t q_head_size = q.shape[1] / head_count_q;
  const int64_t k_head_col = k.shape[1] / head_count_kv;
  const int64_t v_head_col = v.shape[1] / head_count_kv;
  const float scale = std::sqrt(static_cast<float>(k_head_col));

  CHECK_EQ(q_head_size, k_head_col);
  CHECK_EQ(k.shape[0], v.shape[0]);
  CHECK_EQ(out.shape[0], q.shape[0]);
  CHECK_EQ(out.shape[1], v_head_col * head_count_q);

  std::span<float> out_data = out.As<float>();
  std::span<const float> q_data = q.As<const float>();
  std::span<const float> k_data = k.As<const float>();
  std::span<const float> v_data = v.As<const float>();

  RowMatrixXf score(q.shape[0], k.shape[0]);

  // layout of q, k, v are [row, head, col].
  for (int64_t kv_h = 0; kv_h < head_count_kv; kv_h++) {
    // k.row * (k.col / head_count_kv)
    StrideMap<const RowMatrixXf> k_mat(
        k_data.data() + kv_h * k_head_col, k.shape[0], k_head_col,
        Eigen::OuterStride<>(k.stride[0] / sizeof(float)));
    // v.row * (v.col / head_count_kv)
    StrideMap<const RowMatrixXf> v_mat(
        v_data.data() + kv_h * v_head_col, v.shape[0], v_head_col,
        Eigen::OuterStride<>(v.stride[0] / sizeof(float)));

    // softmax(Q @ transpose(K) / sqrt(dk)) @ V.
    for (int64_t q_h = kv_h * q_per_kv; q_h < kv_h * q_per_kv + q_per_kv; q_h++) {
      // q.row * (q.size / head_count_q)
      StrideMap<const RowMatrixXf> q_mat(
          q_data.data() + q_h * q_head_size, q.shape[0], q_head_size,
          Eigen::OuterStride<>(q.stride[0] / sizeof(float)));

      // q.row * (v.col / head_count_kv)
      StrideMap<RowMatrixXf> out_mat(
          out_data.data() + q_h * v_head_col, q.shape[0], v_head_col,
          Eigen::OuterStride<>(out.stride[0] / sizeof(float)));

      // Q @ transpose(K) --> q.row * k.row
      score.noalias() = q_mat * k_mat.transpose();

      // Scale
      score /= scale;

      // Softmax
      MaskedSoftmax(score);

      // score @ V --> q.row * (v.col / head_count_kv)
      out_mat.noalias() = score * v_mat;
    }
  }
}

void ComputeEngine::RmsNorm(MutableMatrixView out,
                            MatrixView input,
                            VectorView gamma,
                            float epsilon) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(input.dtype, DType::F32);
  CHECK_EQ(gamma.dtype, DType::F32);
  CHECK_EQ(out.shape[0], input.shape[0]);
  CHECK_EQ(out.shape[1], input.shape[1]);
  CHECK_EQ(input.shape[1], gamma.shape[0]);

  Map<RowMatrixXf> out_mat = CreateRowMatrixXf(out);
  Map<const RowMatrixXf> input_mat = CreateRowMatrixXf(input);
  Map<const VectorXf> gamma_vec = CreateVectorXf(gamma);
  for (int64_t row = 0; row < input_mat.rows(); row++) {
    auto input_row = input_mat.row(row);
    float rms = sqrt(input_row.squaredNorm() / input_mat.cols() + epsilon);
    out_mat.row(row) = input_row.array() * gamma_vec.transpose().array() / rms;
  }
}

void ComputeEngine::SwiGluMul(MutableMatrixView gate, MatrixView up) {
  CHECK_EQ(gate.dtype, DType::F32);
  CHECK_EQ(up.dtype, DType::F32);
  CHECK_EQ(gate.shape[0], up.shape[0]);
  CHECK_EQ(gate.shape[1], up.shape[1]);

  Map<RowMatrixXf> gate_mat = CreateRowMatrixXf(gate);
  Map<const RowMatrixXf> up_mat = CreateRowMatrixXf(up);
  gate_mat.array() =
      gate_mat.array() / (1.f + (-gate_mat.array()).exp()) * up_mat.array();
}

void ComputeEngine::Softmax(MutableVectorView view) {
  Map<VectorXf> view_vec = CreateVectorXf(view);
  SoftmaxHelper(view_vec);
}

void ComputeEngine::Rope(MutableMatrixView view,
                         int64_t position,
                         float freq_base,
                         int dimension_count,
                         VectorView rope_freqs) {
  CHECK_EQ(view.dtype, DType::F32);
  CHECK_EQ(view.shape[1] % dimension_count, 0);
  CHECK_EQ(rope_freqs.shape[0], dimension_count / 2);
  std::span<float> data = view.As<float>();
  std::span<const float> rope_freqs_data = rope_freqs.As<const float>();
  int64_t pair_count = view.shape[1] / 2;
  for (int64_t row = 0; row < view.shape[0]; row++) {
    for (int64_t p = 0; p < pair_count; p++) {
      float a = data[row * view.shape[1] + 2 * p];
      float b = data[row * view.shape[1] + 2 * p + 1];
      int64_t p_mod = p % (dimension_count / 2);
      float angle = (position + row) *
                    std::pow(freq_base, -2.f * p_mod / dimension_count) /
                    rope_freqs_data[p_mod];
      float cosine = std::cos(angle);
      float sine = std::sin(angle);
      data[row * view.shape[1] + 2 * p] = a * cosine - b * sine;
      data[row * view.shape[1] + 2 * p + 1] = a * sine + b * cosine;
    }
  }
}

}  // namespace tlm
