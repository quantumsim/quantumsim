#include <cuda.h>
#include <cuComplex.h>

// the block size is 1<<LOG_BK_SIZE in x and y direction
#define LOG_BK_SIZE = 5 



//do the cphase gate. set a bit in mask to 1 if you want the corresponding qbit to 
//partake in the cphase.
//no shared memory needed
//run in a 2d grid that stretches over dm10
__global__ void cphase(cuFloatComplex *dm10, unsigned int mask) {

    const int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    const int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    //if exactly one of x,y has all bits in mask set
    if ((x & mask == mask) != (y & mask == mask)) {
        dm10[(x << 10) + y] = -dm10[(x << 10) + y];
    }
}


//do the hadamard on a qbit
//this needs to be run on a grid over d10
//mask must have exactly one bit flipped, denoting which byte is involved
//the results is multiplied with mul, to obtain unitary, set mul = 1/sqrt(2)

__global__ void hadamard(cuFloatComplex *dm10, unsigned int mask, float mul) { 

    const int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    const int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    if (x>y) return;
    if ((x & mask) || (y & mask)) return;

    float *a_re = &dm10[ ((x & ~qmask) << 10) + (y&~qmask) ];
    float *a_im = &dm10[ (x & ~qmask) + ((y&~qmask) << 10) ];

    float *b_re = &dm10[ ((x &~qmask) << 10) + (y|qmask) ];
    float *b_im = &dm10[ (x &~qmask) + ((y|qmask) << 10) ];

    float *c_re = &dm10[ ((x | qmask) << 10) + (y | qmask) ];
    float *a_im = &dm10[ (x  | qmask) + ((y |qmask) << 10) ];


    //this is most probably very bad, and
    //we should reconsider having simply one thread per 2x2 matmul.

    //we are writing the a_block
    //if (!(x & mask) && !(y & mask)) {
        float new_a_re = &a_re + &c_re + 2*&b_re;
        float new_a_im = &a_im + &c_im;
        *a_re = mul*new_a_re;
        *a_im = mul*new_a_im;
    //}

    //we are writing the b block
    //if (!(x & mask) && (y & mask)) {
        float new_b_re = a_re - c_re;
        float new_b_im = a_im - c_im - 2*b_im;
        *b_re = mul*new_b_re;
        *b_im = mul*new_b_im;
    //}

    //we are writing the c block
    //if((x & mask) && (y & mask)) {
        float new_c_re = a_re + c_re - 2*b_re;
        float new_c_im = a_im + c_im;
        *c_re = mul*new_c_re;
        *c_im = mul*new_c_im;
    //}
}



