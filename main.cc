#include "math.h"
#include "matrix.h"
#include "tensor.h"
#include "tensor_view.h"
#include "vector.h"

int main() {
  TensorShape<2> shape;
  shape.dims_[0] = 2;
  shape.dims_[1] = 2;

  TensorShape<1> vector_shape;
  vector_shape.dims_[0] = 3;

  Tensor<2, int> tensor(shape);
  tensor.print();

  IntVector vector(vector_shape);
  vector.buf_[0] = 3;
  vector.print();

  IntVector vector2(vector_shape);
  vector2.buf_[0] = 1;
  vector.print();

  IntVector result_vector = AddIntVector(vector, vector2);
  result_vector.print();

  DoubleMatrix matrix(shape, 1);
  matrix.print();

  TensorView<2, double> view(matrix.shape_.dims_, matrix.buf_);
  view[0][1] = 3;
  std::cout << view[0][0] << std::endl;
  matrix.print();
  return 0;
}
