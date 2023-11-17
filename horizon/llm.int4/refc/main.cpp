// Copyright [2023-11-17] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

#include <bits/stdc++.h>
#include <immintrin.h>

#define N_INPUT 784
#define N_HIDDEN 512
#define N_ClASSES 10
#define QK8_0 32
#define FP16_TO_FP32(x) _cvtsh_ss(x)
#define FP32_TO_FP16(x) _cvtss_sh(x, 0)
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
  uint16_t d;         // delta
  int8_t  qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + QK8_0, "wrong q8_0 block size/padding");


void quantize_row_q8_0(const float * x, block_q8_0 * y, int channels) {
  assert(channels % QK8_0 == 0);
  const int nb = channels / QK8_0;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i*QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y[i].d = FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i*QK8_0 + j]*id;
      y[i].qs[j] = roundf(x0);
    }
  }
}

void vec_dot_q8_0_q8_0(const int channels, float * s, block_q8_0 * x, block_q8_0 * y) {
  const int qk = QK8_0;
  const int nb = channels / qk;

  assert(channels % qk == 0);

  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) {
      sumi += x[i].qs[j] * y[i].qs[j];
    }

    sumf += sumi*(FP16_TO_FP32(x[i].d) * FP16_TO_FP32(y[i].d));
  }

  *s = sumf;
}

void vec_dot_f32_f32(const int channels, float * s, float * x, float * y) {
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < channels; i++) {
    sumf += x[i] * y[i];
  }

  *s = sumf;
}

int main(int argc, char ** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s ./assets/mnist_model.state_dict ./assets/t10k-images.idx3-ubyte 5\n", argv[0]);
    exit(0);
  }

  uint8_t buf[784];
  std::vector<float> digit;

  // read a random digit from the test set
  {
    std::ifstream fin(argv[2], std::ios::binary);
    // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
    fin.seekg(16 + 784 * atoi(argv[3]));
    fin.read((char *) &buf, sizeof(buf));
    fin.close();
  }

  // render the digit in ASCII
  {
    digit.resize(sizeof(buf));
    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++) {
        fprintf(stderr, "%c ", (float)buf[row*28 + col] > 230 ? '*' : '_');
        digit[row*28 + col] = ((float)buf[row*28 + col]);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }

  float fc1_weight[512][784];
  float fc1_bias[512];
  float fc2_weight[10][512];
  float fc2_bias[10];
  float result[10];

  // load weight
  {
    std::ifstream fin(argv[1]);
    float number;
    for (int i = 0; i < 512; i++) {
      for (int j = 0; j < 784; j++) {
        fin >> number;
        fc1_weight[i][j] = number;
      }
    }
    for (int i = 0; i < 512; i++) {
      fin >> number;
      fc1_bias[i] = number;
    }
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 512; j++) {
        fin >> number;
        fc2_weight[i][j] = number;
      }
    }
    for (int i = 0; i < 10; i++) {
      fin >> number;
      fc2_bias[i] = number;
    }
    fin.close();
  }

  // compute
  {
    float activation[512];
    // fc1, (weight [512, 784]) * (image [784]) + (bias [512]) = (output [512])
    {
      for (int i = 0; i < 512; i++) {
        vec_dot_f32_f32(784, &activation[i], fc1_weight[i], digit.data());
        activation[i] += fc1_bias[i];
      }
    }
    // relu, (output [512]) = (output [512])
    {
      for (int i = 0; i < 512; i++) {
        activation[i] = activation[i] > 0 ? activation[i] : 0;
      }
    }
    // fc2, (weight [10, 512]) * (output [512]) + (bias [10]) = (output [10])
    {
      float weight_buf[1024];  // ensure that we have enough buffer to store quantized weight
      float act_buf[1024];  // ensure that we have enough buffer to store quantized activation
      for (int i = 0; i < 10; i++) {
        // quantize weight
        quantize_row_q8_0(fc2_weight[i], (block_q8_0 *)weight_buf, 512);
        // quantize actiavtion
        quantize_row_q8_0(activation, (block_q8_0 *)act_buf, 512);
        // do computation
        vec_dot_q8_0_q8_0(512, &result[i], (block_q8_0 *)weight_buf, (block_q8_0 *)act_buf);
        result[i] += fc2_bias[i];
      }
    }
  }

  for (int i = 0; i < 10; ++i) { printf("%d: %f\n", i, result[i]); }
  const int prediction = std::max_element(result, result + 10) - result;
  fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

  return 0;
}
