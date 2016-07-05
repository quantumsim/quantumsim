/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/

#include <cuda.h> 
#include <cuComplex.h>


//this defines the scheme by which the density matrix is stored in a dm10
//we always require x <= y for the real part and
#define ADDR_BARE(x,y,ri_flag,n) (((((x)<<(n)) | (y)) << 1) | (ri_flag))
#define ADDR_TRIU(x,y,ri_flag,n) (((x) <= (y)) ? ADDR_BARE(x,y,ri_flag,n) : ADDR_BARE(y,x,ri_flag,n))
#define ADDR_REAL(x,y,n) ADDR_TRIU(x,y,0,n)
#define ADDR_IMAG(x,y,n) ADDR_TRIU(x,y,1,n)


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
//run with a 2d grid of total size (2**no_qubits)^2
__global__ void bit_to_pauli_basis(double *complex_dm, unsigned int mask, unsigned int no_qubits) {
    const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    const double sqrt2 =  0.70710678118654752440;
    //const double sqrt2 =  1;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    int b_addr = ((x|mask)<<no_qubits | (y&~mask)) << 1;
    int c_addr = ((x&~mask)<<no_qubits | (y|mask)) << 1;

    if (x&mask && (~y&mask)){
        double b = complex_dm[b_addr];
        double c = complex_dm[c_addr];
        complex_dm[b_addr] = (b+c)*sqrt2;
        complex_dm[c_addr] = (b-c)*sqrt2;
    }
    if ((~x&mask) && (y&mask)){
        b_addr+=1;
        c_addr+=1;
        double b = complex_dm[b_addr];
        double c = complex_dm[c_addr];
        complex_dm[b_addr] = (b+c)*sqrt2;
        complex_dm[c_addr] = (b-c)*sqrt2;
    }
}


//pauli_reshuffle
//this function collects the values from a complex density matrix in (0, x, iy, 1) basis
//and collects the real values only; furthermore it rearranges the address bit order 
//from (d_state_bits, d_state_bits) to 
// (alpha_d, alpha_d-1, ..., alpha_0) where alpha = (00, 01, 10, 11) for 0, x, y, 1
//if direction = 0, the copy is performed from complex to real, otherwise from real to complex
__global__ void pauli_reshuffle(double *complex_dm, double *real_dm, unsigned int no_qubits, unsigned int direction) {

    const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;


    //do we need imaginary part? That is the case if we have an odd number of bits for y in our adress (bit in y is 1, bit in x is 0)
    unsigned int v = ~x & y;

    //short version: while (v>1) { v = (v >> 1) ^ v ;}
    //bit bang version
    v ^= v >> 1;
    v ^= v >> 2;
    v = (v & 0x11111111U) * 0x11111111U;
    v = (v >> 28) & 1;

    const unsigned int addr_complex = (((x << no_qubits) | y) << 1) + v;


    //the adress in pauli basis is obtained by interleaving
    unsigned int addr_real = 0;
    for (int i = 0; i < 16; i++) { 
          addr_real |= (x & 1U << i) << i | (y & 1U << i) << (i + 1);
    }
    

    if(direction == 0) {
        real_dm[addr_real] = complex_dm[addr_complex];
    }
    else {
        complex_dm[addr_complex] = real_dm[addr_real];
    }
}


//do the cphase gate. set a bit in mask to 1 if you want the corresponding qubit to 
//partake in the cphase.
//run in a 2d grid that stretches over dm10
__global__ void cphase(double *dm10, unsigned int mask, unsigned int no_qubits) {
}


// apply a 4x4 pauli transfer matrix (in 0, x, y, 1 basis)
// to the specified qubit
__global__ void single_qubit_ptm(double *dm, double *ptm_g,  unsigned int bit, unsigned int no_qubits) {

    const unsigned int x = threadIdx.x;
    const unsigned int high_x = blockIdx.x * blockDim.x;

    if (high_x + x >= (1 << (2*no_qubits))) return;

    //the two lowest bits of thread id are used to index the target bit,
                                                //      xx <- target bit
    int high_mask = ~ ( (1 << (2*bit+2)) - 1 ); // 1111100000000
    int low_mask  = ~high_mask & (~0x3);        // 0000011111100

    int pos = high_x | x;
    int global_from = (pos & high_mask) | ((pos & 0x3) << (2*bit)) | ((pos & low_mask)>>2);

    extern __shared__ double ptm[];
    double *data = &ptm[16]; //need blockDim.x double floats

    //first fetch the transfer matrix to shared memory
    if(x < 16) ptm[x] = ptm_g[x];

    if(no_qubits < 2) { //what a boring situation
        ptm[x+4] = ptm_g[x+4];
        ptm[x+8] = ptm_g[x+8];
        ptm[x+12] = ptm_g[x+12];
    }

    //fetch block to shared memory
    data[x] = dm[global_from];
    __syncthreads();

    //do calculation

    int row = x & 0x3;
    int idx = x & ~0x3;

    double acc = 0;

    acc += ptm[4*row    ] * data[idx    ];
    acc += ptm[4*row + 1] * data[idx + 1];
    acc += ptm[4*row + 2] * data[idx + 2];
    acc += ptm[4*row + 3] * data[idx + 3];

    //upload back to global memory
    __syncthreads();
    dm[global_from] = acc;
}





//copy the two diagonal blocks of one ancilla into reduced density matrices
//multiply the two with two numbers (inverse of the traces for instance, to implement measurement)
//note that because major bit banging is required to figure out the new adresses,
//the qubit index is passed as an integer, not as a bitmask!
__global__ void dm_reduce(double *dm10, unsigned int bit_idx, double *dm9_0, double *dm9_1, double mul0, double mul1, unsigned int no_qubits) {
}



//trace kernel
//copy the diagonal elements to out, in order to do effective 
//calculation of subtraces.
//run over a 1x9 grid!
__global__ void get_diag(double *dm9, double *out, unsigned int no_qubits) {
    int x = (blockIdx.x *blockDim.x) + threadIdx.x;

    if (x >= (1 << no_qubits)) return;
    unsigned int addr_real = 0;
    for (int i = 0; i < 16; i++) { 
          addr_real |= (x & 1U << i) << i | (x & 1U << i) << (i + 1);
    }
    out[x] = dm9[addr_real];
}


