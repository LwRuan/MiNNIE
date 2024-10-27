#include "pd.h"

#include <Eigen/SVD>

namespace Rain {
void PD::GetPiola(const Mat3& F, Mat3& P) {
  // get rotation
  Mat3 R;
  Mat3 C = F.transpose() * F;
  Mat3 C2 = C * C;
  real det = F.determinant();
  real I_C = C(0, 0) + C(1, 1) + C(2, 2);
  real I_C2 = I_C * I_C;
  real II_C = real(0.5) * (I_C2 - C2(0, 0) - C2(1, 1) - C2(2, 2));
  real III_C = det * det;
  real k = I_C2 - 3 * II_C;

  Mat3 U_inv = Mat3::Zero();
  if (k < real(1e-7)) {
    real lambda_inv = 1.0 / sqrt(I_C / 3);
    U_inv(0, 0) = lambda_inv;
    U_inv(1, 1) = lambda_inv;
    U_inv(2, 2) = lambda_inv;
  } else {
    real l = I_C * (I_C2 - real(4.5) * II_C) + real(13.5) * III_C;
    real k_root = sqrt(k);
    real value = l / (k * k_root);
    if (value < -1.0) value = -1.0;
    if (value > 1.0) value = 1.0;
    real phi = acos(value);
    real lambda2 = (I_C + 2 * k_root * cos(phi / 3)) / 3;
    real lambda = sqrt(lambda2);

    real III_U = sqrt(III_C);
    if (det < 0) III_U = -III_U;
    real I_U = lambda + sqrt(-lambda2 + I_C + 2 * III_U / lambda);
    real II_U = (I_U * I_U - I_C) / 2;

    real inv_rate = 1 / (I_U * II_U - III_U);
    real factor = I_U * III_U * inv_rate;
    Mat3 U = factor * Mat3::Identity();
    factor = (I_U * I_U - II_U) * inv_rate;
    U += factor * C - inv_rate * C2;

    inv_rate = 1 / III_U;
    factor = II_U * inv_rate;
    U_inv(0, 0) = factor;
    U_inv(1, 1) = factor;
    U_inv(2, 2) = factor;
    factor = -I_U * inv_rate;
    U_inv += factor * U + inv_rate * C;
  }
  R = F * U_inv;

  P = 2 * mu_ * (F - R);
}

void PD::GetMixedPiola(const Mat3& F, Mat3& P) {
  // get rotation
  Mat3 R;
  Mat3 C = F.transpose() * F;
  Mat3 C2 = C * C;
  real det = F.determinant();
  real I_C = C(0, 0) + C(1, 1) + C(2, 2);
  real I_C2 = I_C * I_C;
  real II_C = real(0.5) * (I_C2 - C2(0, 0) - C2(1, 1) - C2(2, 2));
  real III_C = det * det;
  real k = I_C2 - 3 * II_C;

  Mat3 U_inv = Mat3::Zero();
  if (k < real(1e-7)) {
    real lambda_inv = 1.0 / sqrt(I_C / 3);
    U_inv(0, 0) = lambda_inv;
    U_inv(1, 1) = lambda_inv;
    U_inv(2, 2) = lambda_inv;
  } else {
    real l = I_C * (I_C2 - real(4.5) * II_C) + real(13.5) * III_C;
    real k_root = sqrt(k);
    real value = l / (k * k_root);
    if (value < -1.0) value = -1.0;
    if (value > 1.0) value = 1.0;
    real phi = acos(value);
    real lambda2 = (I_C + 2 * k_root * cos(phi / 3)) / 3;
    real lambda = sqrt(lambda2);

    real III_U = sqrt(III_C);
    if (det < 0) III_U = -III_U;
    real I_U = lambda + sqrt(-lambda2 + I_C + 2 * III_U / lambda);
    real II_U = (I_U * I_U - I_C) / 2;

    real inv_rate = 1 / (I_U * II_U - III_U);
    real factor = I_U * III_U * inv_rate;
    Mat3 U = factor * Mat3::Identity();
    factor = (I_U * I_U - II_U) * inv_rate;
    U += factor * C - inv_rate * C2;

    inv_rate = 1 / III_U;
    factor = II_U * inv_rate;
    U_inv(0, 0) = factor;
    U_inv(1, 1) = factor;
    U_inv(2, 2) = factor;
    factor = -I_U * inv_rate;
    U_inv += factor * U + inv_rate * C;
  }
  R = F * U_inv;

  P = 2 * mu_ * (F - R);
}

void PD::GetPiolaR(const Mat3& F, const Mat3& R, Mat3& P) {
  P = 2 * mu_ * (F - R);
}

void PD::GetMixedPiolaR(const Mat3& F, const Mat3& R, Mat3& P) {
  P = 2 * mu_ * (F - R);
}

void PD::GetdPdF(const Mat3& F, Mat9& dPdF) {
  dPdF = 2 * mu_ * Mat9::Identity();
}

void PD::GetMixeddPdF(const Mat3& F, Mat9& dPdF) {
  dPdF = 2 * mu_ * Mat9::Identity();
}
};  // namespace Rain