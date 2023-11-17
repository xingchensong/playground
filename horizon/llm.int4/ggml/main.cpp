#include "ggml/ggml.h"

#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// default hparams
struct mnist_hparams {
    int32_t n_input   = 784;
    int32_t n_hidden  = 512;
    int32_t n_classes = 10;
    int32_t ftype     = 1;
};

struct mnist_model {
    mnist_hparams hparams;

    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;

    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;

    struct ggml_context * ctx;
};

// load the model's weights from a file
bool mnist_model_load(const std::string & fname, mnist_model & model) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_input,   sizeof(hparams.n_input));
        fin.read((char *) &hparams.n_hidden,  sizeof(hparams.n_hidden));
        fin.read((char *) &hparams.n_classes, sizeof(hparams.n_classes));
        fin.read((char *) &hparams.ftype,     sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_input   = %d\n", __func__, hparams.n_input);
        printf("%s: n_hidden  = %d\n", __func__, hparams.n_hidden);
        printf("%s: n_classes = %d\n", __func__, hparams.n_classes);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    {
        const auto & hparams = model.hparams;
        ctx_size += hparams.n_input * hparams.n_hidden * ggml_type_sizef(GGML_TYPE_F32); // fc1 weight
        ctx_size +=                   hparams.n_hidden * ggml_type_sizef(GGML_TYPE_F32); // fc1 bias

        ctx_size += hparams.n_hidden * hparams.n_classes * ggml_type_sizef(wtype); // fc2 weight
        ctx_size +=                    hparams.n_classes * ggml_type_sizef(GGML_TYPE_F32); // fc2 bias

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size + 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // load weight
    {
        size_t total_size = 0;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            ggml_tensor * tensor;
            if (name == "fc1_weight") {
              model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                                    model.hparams.n_input, model.hparams.n_hidden);
              tensor = model.fc1_weight;
              ggml_set_name(model.fc1_weight, "fc1_weight");
            } else if (name == "fc1_bias") {
              model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
              tensor = model.fc1_bias;
              ggml_set_name(model.fc1_bias, "fc1_bias");
            } else if (name == "fc2_weight") {
              model.fc2_weight = ggml_new_tensor_2d(ctx, wtype,
                                                    model.hparams.n_hidden, model.hparams.n_classes);
              tensor = model.fc2_weight;
              ggml_set_name(model.fc2_weight, "fc2_weight");
            } else {
              model.fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_classes);
              tensor = model.fc2_bias;
              ggml_set_name(model.fc2_bias, "fc2_bias");
            }

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
                    name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
                    ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
        }
    }

    fin.close();

    return true;
}

// evaluate the model
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - digit:     784 pixel values
//
// returns 0 - 9 prediction
int mnist_eval(
        const mnist_model & model,
        const int n_threads,
        std::vector<float> digit,
        const char * fname_cgraph
        ) {

    const auto & hparams = model.hparams;

    static size_t buf_size = hparams.n_input * sizeof(float) * 32;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_input);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    // fc1 MLP = Ax + b
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input),                model.fc1_bias);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, ggml_relu(ctx0, fc1)), model.fc2_bias);

    ggml_tensor * probs = fc2;
    ggml_set_name(probs, "probs");

    // build / export / run the computation graph
    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    ggml_graph_dump_dot(gf, NULL, "mnist.dot");
    ggml_graph_print   (gf);

    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        ggml_graph_export(gf, "mnist.ggml");

        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    const float * probs_data = ggml_get_data_f32(probs);

    for (int i = 0; i < 10; ++i) { printf("%d: %f\n", i, probs_data[i]); }

    const int prediction = std::max_element(probs_data, probs_data + 10) - probs_data;

    ggml_free(ctx0);

    return prediction;
}

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 4) {
        fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin models/mnist/t10k-images.idx3-ubyte 3\n", argv[0]);
        exit(0);
    }

    uint8_t buf[784];
    mnist_model model;
    std::vector<float> digit;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mnist_model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, "models/ggml-model-f32.bin");
            return 1;
        }

        const int64_t t_load_us = ggml_time_us() - t_start_us;

        fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    // read a random digit from the test set
    {
        std::ifstream fin(argv[2], std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
            return 1;
        }

        // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seekg(16 + 784 * atoi(argv[3]));
        fin.read((char *) &buf, sizeof(buf));
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

    const int prediction = mnist_eval(model, 1, digit, "mnist.ggml");

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    ggml_free(model.ctx);

    return 0;
}
