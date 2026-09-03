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
      const BlockQ8_0& block_data = lhs_data[row * block_count_per_row + block];
      for (int64_t index = 0; index < sizeof(BlockQ8_0::data); index++) {
        int64_t col = block * sizeof(BlockQ8_0::data) + index;
        acc += block_data.data[index] * block_data.scale * rhs_data[col];
      }
    }
    out_data[row] = acc;
  }
}
}  // namespace

ComputeEngine::ComputeEngine() = default;
ComputeEngine::~ComputeEngine() = default;

std::unique_ptr<Storage> ComputeEngine::Alloc(int64_t size) {
  return std::make_unique<Storage>(size);
}

void ComputeEngine::Add(MutableVectorView out, VectorView lhs, VectorView rhs) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(lhs.dtype, DType::F32);
  CHECK_EQ(rhs.dtype, DType::F32);
  CHECK_EQ(out.shape[0], lhs.shape[0]);
  CHECK_EQ(out.shape[0], rhs.shape[0]);
  Map<VectorXf> out_vec = CreateVectorXf(out);
  out_vec = CreateVectorXf(lhs) + CreateVectorXf(rhs);
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

void ComputeEngine::Attn(MutableVectorView out,
                         VectorView q,
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
  CHECK_EQ(q.shape[0] % head_count_q, 0);
  CHECK_EQ(k.shape[1] % head_count_kv, 0);
  CHECK_EQ(v.shape[1] % head_count_kv, 0);

  const int64_t q_per_kv = head_count_q / head_count_kv;
  const int64_t q_head_size = q.shape[0] / head_count_q;
  const int64_t k_head_col = k.shape[1] / head_count_kv;
  const int64_t v_head_col = v.shape[1] / head_count_kv;
  const float scale = std::sqrt(static_cast<float>(k_head_col));

  CHECK_EQ(q_head_size, k_head_col);
  CHECK_EQ(k.shape[0], v.shape[0]);
  CHECK_EQ(out.shape[0], v_head_col * head_count_q);

  std::span<float> out_data = out.As<float>();
  std::span<const float> q_data = q.As<const float>();
  std::span<const float> k_data = k.As<const float>();
  std::span<const float> v_data = v.As<const float>();

  VectorXf score(k.shape[0]);

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

    // softmax(q @ transpose(K) / sqrt(dk)) @ V. The calculation treats vector
    // as N x 1 matrix, which is the default format of eigen vector. So the new
    // formular is:
    // transpose(V) @ softmax(K @ q / sqrt(dk))
    for (int64_t q_h = kv_h * q_per_kv; q_h < kv_h * q_per_kv + q_per_kv; q_h++) {
      // q.size / head_count_q * 1
      Map<const VectorXf> q_vec(q_data.data() + q_h * q_head_size, q_head_size);

      // v.col / head_count_kv * 1
      Map<VectorXf> out_vec(out_data.data() + q_h * v_head_col, v_head_col);

      // K @ q --> k.row * 1
      score.noalias() = k_mat * q_vec;

      // Scale
      score /= scale;

      // Softmax
      SoftmaxHelper(score);

      // transpose(V) @ score --> (v.col / head_count_kv) * 1
      out_vec.noalias() = v_mat.transpose() * score;
    }
  }
}

void ComputeEngine::RmsNorm(MutableVectorView out,
                            VectorView input,
                            VectorView gamma,
                            float epsilon) {
  CHECK_EQ(out.dtype, DType::F32);
  CHECK_EQ(input.dtype, DType::F32);
  CHECK_EQ(gamma.dtype, DType::F32);
  CHECK_EQ(out.shape[0], input.shape[0]);
  CHECK_EQ(input.shape[0], gamma.shape[0]);

  Map<VectorXf> out_vec = CreateVectorXf(out);
  Map<const VectorXf> input_vec = CreateVectorXf(input);
  Map<const VectorXf> gamma_vec = CreateVectorXf(gamma);
  float rms = sqrt(input_vec.squaredNorm() / input_vec.size() + epsilon);
  out_vec.array() = input_vec.array() * gamma_vec.array() / rms;
}

void ComputeEngine::Softmax(MutableVectorView view) {
  Map<VectorXf> view_vec = CreateVectorXf(view);
  SoftmaxHelper(view_vec);
}

void ComputeEngine::Rope(MutableVectorView view,
                         int64_t position,
                         float freq_base,
                         int dimension_count) {
  CHECK_EQ(view.dtype, DType::F32);
  CHECK_EQ(view.shape[0] % dimension_count, 0);
  std::span<float> data = view.As<float>();
  int64_t pair_count = data.size() / 2;
  for (int64_t p = 0; p < pair_count; p++) {
    float a = data[2 * p];
    float b = data[2 * p + 1];
    float angle =
        position * std::pow(freq_base, -2.f * (p % (dimension_count / 2)) /
                                           dimension_count);
    float cosine = std::cos(angle);
    float sine = std::sin(angle);
    data[2 * p] = a * cosine - b * sine;
    data[2 * p + 1] = a * sine + b * cosine;
  }
}

}  // namespace tlm
