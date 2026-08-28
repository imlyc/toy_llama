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
}  // namespace

ComputeEngine::ComputeEngine() = default;
ComputeEngine::~ComputeEngine() = default;

std::unique_ptr<Storage> ComputeEngine::Alloc(int64_t size) {
  return std::make_unique<Storage>(size);
}

void ComputeEngine::Add(MutableVectorView out, VectorView lhs, VectorView rhs) {
}

void ComputeEngine::MatMul(MutableVectorView out,
                           VectorView lhs,
                           MatrixView rhs) {}

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

void ComputeEngine::Softmax(MutableVectorView view) {
  CHECK_EQ(view.dtype, DType::F32);
  std::span<float> view_data = view.As<float>();
  Map<VectorXf> view_vec(view_data.data(), view_data.size());
  SoftmaxHelper(view_vec);
}

void ComputeEngine::Rope(MutableVectorView view, int64_t position) {}

}  // namespace tlm
