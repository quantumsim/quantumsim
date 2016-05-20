#include <cuda.h>
#include <cuComplex.h>

// the block size is 1<<LOG_BK_SIZE in x and y direction
#define LOG_BK_SIZE (5)

#define NO_QUBITS 10


//this defines the scheme by which the density matrix is stored in a dm10
//we always require x <= y for the real part and
#define ADDR_BARE(x,y,ri_flag) (((((x)<<NO_QUBITS) | (y)) << 1) | (ri_flag))
#define ADDR_TRIU(x,y,ri_flag) (((x) <= (y)) ? ADDR_BARE(x,y,ri_flag) : ADDR_BARE(y,x,ri_flag))
#define ADDR_REAL(x,y) ADDR_TRIU(x,y,0)
#define ADDR_IMAG(x,y) ADDR_TRIU(x,y,1)


//do the cphase gate. set a bit in mask to 1 if you want the corresponding qbit to 
//partake in the cphase.
//run in a 2d grid that stretches over dm10
__global__ void cphase(float *dm10, unsigned int mask) {
    const int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    const int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    //if exactly one of x,y has all bits in mask set
    if (((x & mask) == mask) != ((y & mask) == mask)) {

        if (x <= y) { //real part
            dm10[ADDR_REAL(x,y)] = -dm10[ADDR_REAL(x,y)];
        }
        else { //imaginary part
            dm10[ADDR_IMAG(y,x)] = -dm10[ADDR_IMAG(y,x)];
        }
    }
}


//do the hadamard on a qbit
//this needs to be run on a grid over d10
//mask must have exactly one bit flipped, denoting which byte is involved
//the results is multiplied with mul, to obtain unitary, set mul = 0.5
__global__ void hadamard(float *dm10, unsigned int mask, float mul) { 

    int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    if((x&mask) && (~y&mask)) { //real part
        x = x & ~mask;
        if (x <= y) {
            float a = dm10[ADDR_REAL(x, y)];
            float b = dm10[ADDR_REAL(x|mask, y)];
            float c = dm10[ADDR_REAL(x, y|mask)];
            float d = dm10[ADDR_REAL(x|mask, y|mask)];

            float new_a = a+b+c+d;
            float new_b = a-b+c-d;
            float new_c = a+b-c-d;
            float new_d = a-b-c+d;

            dm10[ADDR_REAL(x, y)] = mul*new_a;
            dm10[ADDR_REAL(x|mask, y)] = mul*new_b;
            dm10[ADDR_REAL(x, y|mask)] = mul*new_c;
            dm10[ADDR_REAL(x|mask, y|mask)] = mul*new_d;
        }
    }
    if ((~x&mask) && (y&mask)) { //do the imaginary part
        y = y & ~mask;
        if (y <= x){
            float a = dm10[ADDR_IMAG(y, x)];
            float b = dm10[ADDR_IMAG(y|mask, x)];
            float c = dm10[ADDR_IMAG(y, x|mask)];
            float d = dm10[ADDR_IMAG(y|mask, x|mask)];

            float new_a = a+b+c+d;
            float new_b = a-b+c-d;
            float new_c = a+b-c-d;
            float new_d = a-b-c+d;

            dm10[ADDR_IMAG(y, x)] = mul*new_a;
            dm10[ADDR_IMAG(y|mask, x)] = mul*new_b;
            dm10[ADDR_IMAG(y, x|mask)] = mul*new_c;
            dm10[ADDR_IMAG(y|mask, x|mask)] = mul*new_d;
        }
    }
}


//model the effect of amplitude damping and phase damping 
// gamma = probability for amplitude decay, i.e gamma = 1-exp(-t/T1)
// s1mgamma = sqrt(1-gamma)
// s1mlambda = sqrt(1-lambda), where lambda is the probability for a phase flip, i.e. lambda = 1- exp(-t/T2)
__global__ void amp_ph_damping(float *dm10, unsigned int mask, float gamma, float s1mgamma, float s1mlambda) {

    int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    int ri_flag = 1;

    if (x >= y) ri_flag = 0;


    float f = dm10[ADDR_TRIU(x, y, ri_flag)];

    if (x&y&mask) { //c block

        dm10[ADDR_TRIU(x&~mask,y&~mask,ri_flag)]  += gamma * f;
        f = f - gamma*f;

    } 
    else if ((~x)&(~y)&mask) {
        // a block, do nothing
    }
    else { // b block
        f *= s1mgamma * s1mlambda;
    }

    dm10[ADDR_TRIU(x, y, ri_flag)] = f;
}
