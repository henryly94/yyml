#include "math.h"
#include "matrix.h"
#include "tensor.h"
#include "tensor_view.h"
#include "vector.h"

int main() {
  // Initialize matrix.
  TensorShape<2> shape{2, 2};
  DoubleMatrix matrix(shape, [](size_t x, size_t y) { return x + y; });
  matrix.print();

  // Use TensorView to get random access to elements.
  TensorView<2, double> view(matrix.shape_.dim, matrix.buf_);
  view[0][1] = 3;
  std::cout << "View: " << view[0][0] << ' ' << (view[1][0] == view[1][1])
            << std::endl;
  matrix.print();

  // Try higher dimension!
  TensorShape<3> cubic_shape{3, 3, 3};
  Tensor<3, double> cubic(cubic_shape);
  Tensor<3, double>::view_type cubic_view(cubic.shape_.dim, cubic.buf_);
  cubic.print();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        cubic_view[i][j][k] = i * 9 + j * 3 + k;
      }
    }
  }
  cubic.print();

  // Math
  std::cout << "Math\n==============\n";
  TensorShape<1> vector_shape{2};
  DoubleMatrix src_matrix(shape, [](size_t x, size_t y) { return x + 2 * y; });
  DoubleVector src_vector(vector_shape), dst_vector(vector_shape);
  src_vector.buf_[0] = 1;
  src_vector.buf_[1] = 2;
  src_matrix.print();
  src_vector.print();
  MatrixMultiplyVector<double>(src_matrix, src_vector, &dst_vector);
  dst_vector.print();
  return 0;
}
