#include "dense_layer.h"
#include "input_layer.h"
#include "math.h"
#include "matrix.h"
#include "neural_network.h"
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

  // NN
  std::cout << "NN\n================\n";
  TensorShape<1> layer_shape_1{2};
  TensorShape<2> layer_shape_2{3, 2}, layer_shape_3{4, 3};
  auto initializer = [](size_t x, size_t y) { return (x + y) % 2; };
  InputLayer layer1(layer_shape_1);
  DenseLayer layer2(layer_shape_2, initializer),
      layer3(layer_shape_3, initializer);

  layer2.weights_.print();
  layer3.weights_.print();

  layer2.prev_layer_ = &layer1;
  layer3.prev_layer_ = &layer2;
  layer1.inputs_.print();
  layer2.outputs_.print();
  layer3.outputs_.print();

  layer1.inputs_.buf_[0] = 1;
  layer1.inputs_.buf_[1] = 2;
  Forward(layer1, layer2);
  Forward(layer2, layer3);

  layer1.inputs_.print();
  layer2.outputs_.print();
  layer3.outputs_.print();
  return 0;
}
