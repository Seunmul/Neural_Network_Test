/*
  rnn.c
  Recurrent Neural Network in C.

  $ cc -o rnn rnn.c -lm
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mem_check.h"

#define DEBUG_LAYER 0

#define INPUT_LAYER_DATA_SIZE 10
#define HIDDEN_LAYER_DATA_SIZE 16
#define OUTPUT_LAYER_DATA_SIZE 10
#define MODEL_SEQUNCE 5
#define LEARNING_RATE 0.001
#define EPOCHS 15000
#define MINIBATCH 50

static int clocks_starts = 0;
static size_t used_memory = 0;

/* f: input generator */
static int f(int i)
{
    static int a[] = {5, 9, 4, 0, 5, 9, 6, 3, 0, 2};
    return a[i % INPUT_LAYER_DATA_SIZE];
}
/* g: function to learn */
static double g(int i)
{
    return ((i % INPUT_LAYER_DATA_SIZE) == 4) ? 1 : 0;
}

/*  Misc. functions
 */

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd()
{
    return (rnd() + rnd() + rnd() + rnd() - 2.0) * 1.724; /* std=1.0 */
}

#if 0
/* tanh(x): hyperbolic tangent */
static inline double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2*x)) - 1.0;
}
#endif
/* tanh_g(y): hyperbolic tangent gradient */
static inline double tanh_g(double y)
{
    return 1.0 - y * y;
}

/*  RNNLayer */
typedef struct _RNNLayer
{
    int lid;                 /* Layer ID */
    struct _RNNLayer *lprev; /* Previous Layer */
    struct _RNNLayer *lnext; /* Next Layer */

    int nnodes; /* Num. of Nodes */
    int ntimes; /* Num. of Times ==> sequences */

    /* array layout: [ v[t=0], v[t=-1], ..., v[t=-(ntimes-1)] ] */
    double *outputs; /* Node Outputs */
    double *errors;  /* Node Errors */
    double *temp;    /* Node Hidden (temporary) */

    int nxweights;      /* Num. of XWeights */
    double *xweights;   /* XWeights (trained) */
    double *u_xweights; /* XWeight Updates */
    int nhweights;      /* Num. of HWeights */
    double *hweights;   /* HWeights (trained) */
    double *u_hweights; /* HWeight Updates */

    int nbiases;      /* Num. of Biases */
    double *biases;   /* Biases (trained) */
    double *u_biases; /* Bias Updates */

} RNNLayer;

/* RNNLayer_create(lprev, nnodes)
   Creates a RNNLayer object.
*/
RNNLayer *RNNLayer_create(RNNLayer *lprev, int nnodes, int ntimes)
{
    RNNLayer *self = (RNNLayer *)calloc_c(1, sizeof(RNNLayer), &used_memory);
    if (self == NULL)
        return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->lid = 0;
    if (lprev != NULL)
    {
        assert(lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid + 1;
    }

    self->nnodes = nnodes;
    self->ntimes = ntimes;
    int n_seq = self->nnodes * self->ntimes;
    self->outputs = (double *)calloc_c(n_seq, sizeof(double), &used_memory);
    self->errors = (double *)calloc_c(n_seq, sizeof(double), &used_memory);
    self->temp = (double *)calloc_c(self->nnodes, sizeof(double), &used_memory);

    if (lprev != NULL)
    {
        /* Fully connected */
        self->nxweights = lprev->nnodes * self->nnodes;
        self->xweights = (double *)calloc_c(self->nxweights, sizeof(double), &used_memory);
        self->u_xweights = (double *)calloc_c(self->nxweights, sizeof(double), &used_memory);
        for (int i = 0; i < self->nxweights; i++)
        {
            self->xweights[i] = 0.1 * nrnd();
        }
        self->nhweights = self->nnodes * self->nnodes;
        self->hweights = (double *)calloc_c(self->nhweights, sizeof(double), &used_memory);
        self->u_hweights = (double *)calloc_c(self->nhweights, sizeof(double), &used_memory);
        for (int i = 0; i < self->nhweights; i++)
        {
            self->hweights[i] = 0.1 * nrnd();
        }

        self->nbiases = self->nnodes;
        self->biases = (double *)calloc_c(self->nbiases, sizeof(double), &used_memory);
        self->u_biases = (double *)calloc_c(self->nbiases, sizeof(double), &used_memory);
        for (int i = 0; i < self->nbiases; i++)
        {
            self->biases[i] = 0;
        }
    }

    return self;
}
long int RNNLayer_check_memory(RNNLayer *self, FILE *fp)
{
    long size = 0;
    if (self == NULL)
        return 0;
    size += sizeof(RNNLayer); // for RNNlayer itself

    /* nnodes -> layer's node size, ntimes -> rnn layer sequence size */
    size += self->nnodes * self->ntimes * sizeof(double); // for outputs
    size += self->nnodes * self->ntimes * sizeof(double); // for errors
    size += self->nnodes * sizeof(double);                // for temps

    if (self->lprev != NULL)
    {
        /* nxweights = lprev->nnodes * self->nnodes */
        size += self->nxweights * sizeof(double); // for xweights
        size += self->nxweights * sizeof(double); // for u_xweights

        /* nhweights = self->nnodes * self->nnodes */
        size += self->nhweights * sizeof(double); // for nhweights
        size += self->nhweights * sizeof(double); // for u_nhweights

        /* nbiases = self->nnodes */
        size += self->nbiases * sizeof(double); // for biases
        size += self->nbiases * sizeof(double); // for u_biases
    }
    fprintf(stderr, "size of RNNLayer%d : %ld bytes\n", self->lid, size);
    return size;
}

/* RNNLayer_destroy(self)
   Releases the memory.
*/
void RNNLayer_destroy(RNNLayer *self)
{
    assert(self != NULL);

    free(self->temp);
    free(self->outputs);
    free(self->errors);

    if (self->xweights != NULL)
    {
        free(self->xweights);
    }
    if (self->u_xweights != NULL)
    {
        free(self->u_xweights);
    }
    if (self->hweights != NULL)
    {
        free(self->hweights);
    }
    if (self->u_hweights != NULL)
    {
        free(self->u_hweights);
    }

    if (self->biases != NULL)
    {
        free(self->biases);
    }
    if (self->u_biases != NULL)
    {
        free(self->u_biases);
    }

    free(self);
}

/* RNNLayer_dump(self, fp)
   Shows the debug output.
*/
void RNNLayer_dump(const RNNLayer *self, FILE *fp)
{
    assert(self != NULL);
    RNNLayer *lprev = self->lprev;
    fprintf(fp, "RNNLayer%d", self->lid);
    if (lprev != NULL)
    {
        fprintf(fp, " (<- Layer%d)", lprev->lid);
    }
    fprintf(fp, ": nodes=%d\n", self->nnodes);

    if (self->xweights != NULL)
    {
        int k = 0;
        for (int i = 0; i < self->nnodes; i++)
        {
            fprintf(fp, "  xweights(%d) = [", i);
            for (int j = 0; j < lprev->nnodes; j++)
            {
                fprintf(fp, " %.4f", self->xweights[k++]);
            }
            fprintf(fp, "]\n");
        }
        assert(k == self->nxweights);
    }
    if (self->hweights != NULL)
    {
        int k = 0;
        for (int i = 0; i < self->nnodes; i++)
        {
            fprintf(fp, "  hweights(%d) = [", i);
            for (int j = 0; j < self->nnodes; j++)
            {
                fprintf(fp, " %.4f", self->hweights[k++]);
            }
            fprintf(fp, "]\n");
        }
        assert(k == self->nhweights);
    }

    if (self->biases != NULL)
    {
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nbiases; i++)
        {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
    }

    {
        int k = 0;
        for (int t = 0; t < self->ntimes; t++)
        {
            fprintf(fp, "  outputs(t=%d) = [", -t);
            for (int i = 0; i < self->nnodes; i++)
            {
                fprintf(fp, " %.4f", self->outputs[k++]);
            }
            fprintf(fp, "]\n");
        }
    }
    fprintf(fp, "\n");
}

/* RNNLayer_reset(self)
   Resets the hidden states.
*/
void RNNLayer_reset(RNNLayer *self)
{
    assert(self != NULL);

    for (int i = 0; i < self->nnodes; i++)
    {
        self->outputs[i] = 0;
    }
}

/* RNNLayer_feedForw(self)
   Performs feed forward updates.
*/
static void RNNLayer_feedForw(RNNLayer *self)
{
    assert(self->lprev != NULL);
    RNNLayer *lprev = self->lprev;

    /* Save the previous values. */
    for (int t = self->ntimes - 1; 0 < t; t--)
    {
        int idst = self->nnodes * t;
        int isrc = self->nnodes * (t - 1);
        for (int i = 0; i < self->nnodes; i++)
        {
            self->outputs[idst + i] = self->outputs[isrc + i];
        }
    }
    /* outputs[0..] will be replaced by the new values. */

    int kx = 0, kh = 0;
    for (int i = 0; i < self->nnodes; i++)
    {
        /* H = f(Bh + Wx * X + Wh * H) */
        double h = self->biases[i];
        for (int j = 0; j < lprev->nnodes; j++)
        {
            h += (lprev->outputs[j] * self->xweights[kx++]);
        }
        for (int j = 0; j < self->nnodes; j++)
        {
            h += (self->outputs[j] * self->hweights[kh++]);
        }
        self->temp[i] = h;
    }
    assert(kx == self->nxweights);
    assert(kh == self->nhweights);
    for (int i = 0; i < self->nnodes; i++)
    {
        self->outputs[i] = tanh(self->temp[i]);
    }

#if DEBUG_LAYER
    fprintf(stderr, "RNNLayer_feedForw(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++)
    {
        fprintf(stderr, " %.4f (%.4f)", self->outputs[i], self->temp[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

/* RNNLayer_feedBack(self)
   Performs backpropagation.
*/
static void RNNLayer_feedBack(RNNLayer *self)
{
    if (self->lprev == NULL)
        return;

    assert(self->lprev != NULL);
    RNNLayer *lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++)
    {
        lprev->errors[j] = 0;
    }

    for (int t = 0; t < self->ntimes; t++)
    {
        int kx = 0, kh = 0;
        int i0 = t * self->nnodes;
        int i1 = (t + 1) * self->nnodes;
        int j0 = t * lprev->nnodes;
        for (int i = 0; i < self->nnodes; i++)
        {
            /* Computer the weight/bias updates. */
            double y = self->outputs[i0 + i];
            double g = tanh_g(y);
            double dnet = self->errors[i0 + i] * g;
            if ((t + 1) < lprev->ntimes)
            {
                for (int j = 0; j < lprev->nnodes; j++)
                {
                    /* Propagate the errors to the previous layer. */
                    lprev->errors[j0 + j] += self->xweights[kx] * dnet;
                    self->u_xweights[kx] += dnet * lprev->outputs[j0 + j];
                    kx++;
                }
            }
            if ((t + 1) < self->ntimes)
            {
                for (int j = 0; j < self->nnodes; j++)
                {
                    self->errors[i1 + j] += self->hweights[kh] * dnet;
                    self->u_hweights[kh] += dnet * self->outputs[i1 + j];
                    kh++;
                }
            }
            self->u_biases[i] += dnet;
        }
        if ((t + 1) < lprev->ntimes)
        {
            assert(kx == self->nxweights);
        }
        if ((t + 1) < self->ntimes)
        {
            assert(kh == self->nhweights);
        }
    }

    /* Save the previous values. */
    for (int t = self->ntimes - 1; 0 < t; t--)
    {
        int idst = self->nnodes * t;
        int isrc = self->nnodes * (t - 1);
        for (int i = 0; i < self->nnodes; i++)
        {
            self->errors[idst + i] = self->errors[isrc + i];
        }
    }
    /* errors[0..] will be replaced by the new values. */

#if DEBUG_LAYER
    fprintf(stderr, "RNNLayer_feedBack(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++)
    {
        double y = self->outputs[i];
        double g = tanh_g(y);
        double dnet = self->errors[i] * g;
        fprintf(stderr, "  dnet = %.4f, dw = [", dnet);
        for (int j = 0; j < lprev->nnodes; j++)
        {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* RNNLayer_setInputs(self, values)
   Sets the input values.
*/
void RNNLayer_setInputs(RNNLayer *self, const double *values)
{
    assert(self != NULL);
    assert(self->lprev == NULL);

#if DEBUG_LAYER
    fprintf(stderr, "RNNLayer_setInputs(Layer%d):\n", self->lid);
    fprintf(stderr, "  values = [");
    for (int i = 0; i < self->nnodes; i++)
    {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Save the previous values. */
    for (int t = self->ntimes - 1; 0 < t; t--)
    {
        int idst = self->nnodes * t;
        int isrc = self->nnodes * (t - 1);
        for (int i = 0; i < self->nnodes; i++)
        {
            self->outputs[idst + i] = self->outputs[isrc + i];
        }
    }
    /* outputs[0..] will be replaced by the new values. */

    /* Set the input values as the outputs. */
    for (int i = 0; i < self->nnodes; i++)
    {
        self->outputs[i] = values[i];
    }

    /* Start feed forwarding. */
    RNNLayer *layer = self->lnext;
    while (layer != NULL)
    {
        RNNLayer_feedForw(layer);
        layer = layer->lnext;
    }
}

/* RNNLayer_getOutputs(self, outputs)
   Gets the output values.
*/
void RNNLayer_getOutputs(const RNNLayer *self, double *outputs)
{
    assert(self != NULL);
    for (int i = 0; i < self->nnodes; i++)
    {
        outputs[i] = self->outputs[i];
    }
}

/* RNNLayer_getErrorTotal(self)
   Gets the error total.
*/
double RNNLayer_getErrorTotal(const RNNLayer *self)
{
    assert(self != NULL);
    double total = 0;
    for (int i = 0; i < self->nnodes; i++)
    {
        double e = self->errors[i];
        total += e * e;
    }
    return (total / self->nnodes);
}

/* RNNLayer_learnOutputs(self, values)
   Learns the output values.
*/
void RNNLayer_learnOutputs(RNNLayer *self, const double *values)
{
    assert(self != NULL);
    assert(self->lprev != NULL);
    for (int i = 0; i < self->nnodes; i++)
    {
        self->errors[i] = (self->outputs[i] - values[i]);
    }

#if DEBUG_LAYER
    fprintf(stderr, "RNNLayer_learnOutputs(Layer%d):\n", self->lid);
    fprintf(stderr, "  values = [");
    for (int i = 0; i < self->nnodes; i++)
    {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n  errors = [");
    for (int i = 0; i < self->nnodes; i++)
    {
        fprintf(stderr, " %.4f", self->errors[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Start backpropagation. */
    RNNLayer *layer = self;
    while (layer != NULL)
    {
        RNNLayer_feedBack(layer);
        layer = layer->lprev;
    }
}

/* RNNLayer_update(self, rate)
   Updates the weights.
*/
void RNNLayer_update(RNNLayer *self, double rate)
{
#if DEBUG_LAYER
    fprintf(stderr, "RNNLayer_update(Layer%d): rate = %.4f\n", self->lid, rate);
#endif

    /* Update the bias and weights. */
    if (self->biases != NULL)
    {
        for (int i = 0; i < self->nbiases; i++)
        {
            self->biases[i] -= rate * self->u_biases[i];
            self->u_biases[i] = 0;
        }
    }
    if (self->xweights != NULL)
    {
        for (int i = 0; i < self->nxweights; i++)
        {
            self->xweights[i] -= rate * self->u_xweights[i];
            self->u_xweights[i] = 0;
        }
    }
    if (self->hweights != NULL)
    {
        for (int i = 0; i < self->nhweights; i++)
        {
            self->hweights[i] -= rate * self->u_hweights[i];
            self->u_hweights[i] = 0;
        }
    }

    /* Update the previous layer. */
    if (self->lprev != NULL)
    {
        RNNLayer_update(self->lprev, rate);
    }
}

/* main */
int main(int argc, char *argv[])
{
    /* model variables */
    int ntimes = MODEL_SEQUNCE;  // rnn sequence
    double rate = LEARNING_RATE; // learning rate
    int nepochs = EPOCHS;        // epcohs
    int minibatch = MINIBATCH;   // minibatch size

    /* Use a fixed random seed for debugging. */
    srand(1);
    fprintf(stderr, "\n------ model configuration ------\n");
    fprintf(stderr, "default size of RNN Layers : %ld\n", sizeof(RNNLayer));
    fprintf(stderr, "Model Sequence : %d\n", ntimes);
    fprintf(stderr, "Input Layer Data Size : %d\n", INPUT_LAYER_DATA_SIZE);
    fprintf(stderr, "Hidden Layer Data Size : %d\n", HIDDEN_LAYER_DATA_SIZE);
    fprintf(stderr, "Output Layer Data Size : %d\n", OUTPUT_LAYER_DATA_SIZE);
    fprintf(stderr, "\n------ hyperparamters ------\n");
    fprintf(stderr, "Learning Rate : %.4f\n", rate);
    fprintf(stderr, "Epochs : %d\n", nepochs);
    fprintf(stderr, "Minibatch Size : %d\n", minibatch);
    fprintf(stderr, "\n------ initialize model ------\n");

    /* Initialize layers. */
    RNNLayer *linput = RNNLayer_create(NULL, INPUT_LAYER_DATA_SIZE, ntimes);
    RNNLayer *lhidden1 = RNNLayer_create(linput, HIDDEN_LAYER_DATA_SIZE, ntimes);
    RNNLayer *lhidden2 = RNNLayer_create(lhidden1, HIDDEN_LAYER_DATA_SIZE, ntimes);
    RNNLayer *lhidden3 = RNNLayer_create(lhidden2, HIDDEN_LAYER_DATA_SIZE, ntimes);
    RNNLayer *loutput = RNNLayer_create(lhidden3, OUTPUT_LAYER_DATA_SIZE, ntimes);

    /* Check memory of eacy layers */
    /*
        레이어 크기는 다음과 같이 할당됩니다.
        레이어 본인 : sizeof(RNNLayer)

        error : 노드 수 * 시퀀스 수 * sizeof(double)
        output : 노드 수 * 시퀀스 수 * sizeof(double)
        temp : 노드 수 * sizeof(double)

        xweights : 이전 레이어 노드 수 * 노드 수 * sizeof(double)
        u_xweights : 이전 레이어 노드 수 * 노드 수 * sizeof(double)

        hweights : 노드 수 * 노드 수 * sizeof(double)
        u_hweights : 노드 수 * 노드 수 * sizeof(double)

        biases : 노드 수 * sizeof(double)
        u_biases : 노드 수 * sizeof(double)
    */
    RNNLayer_check_memory(linput, stderr);
    RNNLayer_check_memory(lhidden1, stderr);
    RNNLayer_check_memory(lhidden2, stderr);
    RNNLayer_check_memory(lhidden3, stderr);
    RNNLayer_check_memory(loutput, stderr);

    fprintf(stderr, "\nUsed Memory : %ld bytes\n\n", used_memory_in_bytes(used_memory));
    RNNLayer_dump(linput, stderr);
    RNNLayer_dump(lhidden1, stderr);
    RNNLayer_dump(lhidden2, stderr);
    RNNLayer_dump(lhidden3, stderr);
    RNNLayer_dump(loutput, stderr);
    fprintf(stderr, "------ model initialized ------\n");

    /* Time Checking*/
    reset_timer(clocks_starts);
    fprintf(stderr, "\n------ train model ------\n");
    /* Run the network. */
    for (int n = 0; n < nepochs; n++) // 각 에폭마다 러닝 레이트에 따라 가중치와 편향값 조정
    {
        int i = rand() % 10000;          // 0~9999 사이의 랜덤값을 부여하며, 해당 숫자부터 1에폭간 n개의 데이터를 학습
        double x[INPUT_LAYER_DATA_SIZE]; // 입력값 배열 => 노드 수에 따라 배열 크기 결정
        // double y1[HIDDEN_LAYER_DATA_SIZE]; // 중간 output 값 배열 => 노드 수에 따라 배열 크기 결정
        // double y2[HIDDEN_LAYER_DATA_SIZE]; // 중간 output 값 배열 => 노드 수에 따라 배열 크기 결정
        // double y3[HIDDEN_LAYER_DATA_SIZE]; // 중간 output 값 배열 => 노드 수에 따라 배열 크기 결정
        double r[OUTPUT_LAYER_DATA_SIZE]; // 출력값 배열 => 노드 수에 따라 배열 크기 결정

        /* reset previous sequnce outputs */
        RNNLayer_reset(linput);
        RNNLayer_reset(lhidden1);
        RNNLayer_reset(lhidden2);
        RNNLayer_reset(lhidden3);
        RNNLayer_reset(loutput);
        // fprintf(stderr, "reset: i=%d\n", i);

        /* repeat and learing from i to i + minibatch size => 1epoch  */
        for (int j = i; j < i + minibatch; j++)
        {
            /* get input data */
            int p = f(j);

            /* data formatting */
            for (int k = 0; k < INPUT_LAYER_DATA_SIZE; k++)
            {
                x[k] = (k == p) ? 1 : 0; // data p will formatted with input nodes size
            }

            /* get answer */
            r[0] = g(j);

            /* learning */
            RNNLayer_setInputs(linput, x); // 순전파 알고리즘 동작
            // RNNLayer_getOutputs(lhidden1, y1);  // 순전파 알고리즘 동작 후 결과값 저장
            // RNNLayer_getOutputs(lhidden2, y2);  // 순전파 알고리즘 동작 후 결과값 저장
            // RNNLayer_getOutputs(loutput, y3);  // 순전파 알고리즘 동작 후 결과값 저장
            RNNLayer_learnOutputs(loutput, r); // 역전파 알고리즘 동작 => 에러 체크

            // /* check outputs */
            // double etotal = RNNLayer_getErrorTotal(loutput);
            // fprintf(stderr, "x[%d]=%d, y=%.4f, r=%.4f, etotal=%.4f\n",
            //         j, p, y1[0], r[0], etotal);
            // fprintf(stderr, "x[%d]=%d, ", j, p);
            // for (int k = 0; k < HIDDEN_LAYER_DATA_SIZE; k++)
            //     fprintf(stderr, "y1[%d]=%.4f, ", k, y1[k]);
            // for (int k = 0; k < HIDDEN_LAYER_DATA_SIZE; k++)
            //     fprintf(stderr, "y2[%d]=%.4f, ", k, y2[k]);
            // for (int k = 0; k < HIDDEN_LAYER_DATA_SIZE; k++)
            //     fprintf(stderr, "y3[%d]=%.4f, ", k, y3[k]);
            // fprintf(stderr, "r=%.4f, etotal=%.4f\n", r[0], etotal);
        }
        RNNLayer_update(loutput, rate); // 러닝 레이트에 따라 가중치와 편향값 조정
        if (n % 1000 == 0)
        {
            fprintf(stderr, "epoch : %d, check time used..", n);
            show_elapsed_time_in_sec(clocks_starts);
        }
    }

    /* Time Checking*/
    show_elapsed_time_in_sec(clocks_starts);
    fprintf(stderr, "\nUsed Memory : %ld bytes\n\n", used_memory_in_bytes(used_memory));

    /* Dump the finished network. */
    fprintf(stderr, "---------Training finished.---------\n starts dumping and testing...\n");
    RNNLayer_dump(linput, stderr);
    RNNLayer_dump(lhidden1, stderr);
    RNNLayer_dump(lhidden2, stderr);
    RNNLayer_dump(lhidden3, stderr);
    RNNLayer_dump(loutput, stderr);

    /* Testing the network with test data sequnce. */
    RNNLayer_reset(linput);
    RNNLayer_reset(lhidden1);
    RNNLayer_reset(lhidden2);
    RNNLayer_reset(lhidden3);
    RNNLayer_reset(loutput);
    for (int i = 0; i < 30; i++)
    {
        double x[INPUT_LAYER_DATA_SIZE];
        double y[OUTPUT_LAYER_DATA_SIZE];
        int p = f(i);
        for (int k = 0; k < INPUT_LAYER_DATA_SIZE; k++)
        {
            x[k] = (k == p) ? 1 : 0;
        }
        RNNLayer_setInputs(linput, x);
        RNNLayer_getOutputs(loutput, y);
        fprintf(stderr, "x[%d]=%d, y=%.4f, %.4f\n", i, p, y[0], g(i));
    }

    // /* random data test the network. */
    // fprintf(stderr, "random test\n");
    // RNNLayer_reset(linput);
    // RNNLayer_reset(lhidden1);
    // RNNLayer_reset(lhidden2);
    // RNNLayer_reset(lhidden3);
    // RNNLayer_reset(loutput);
    // double testx[INPUT_LAYER_DATA_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // double testy[OUTPUT_LAYER_DATA_SIZE];

    // int test_i = 0;
    // while (test_i < 100)
    // {
    //     RNNLayer_setInputs(linput, testx);
    //     RNNLayer_getOutputs(loutput, testy);
    //     int r = (int)((double)rand()) % 10;
    //     fprintf(stderr, "r : %d , testx = ", r);
    //     testx[r] = 1;
    //     for (int i = 0; i < INPUT_LAYER_DATA_SIZE; i++)
    //     {
    //         fprintf(stderr, "%.4f ", testx[i]);
    //     }
    //     fprintf(stderr, "\ntest_i : %d testy=%.4f\n", test_i, testy[0]);
    //     testx[r] = 0;
    //     test_i++;
    // }

    RNNLayer_destroy(linput);
    RNNLayer_destroy(lhidden1);
    RNNLayer_destroy(lhidden2);
    RNNLayer_destroy(lhidden3);
    RNNLayer_destroy(loutput);
    return 0;
}
