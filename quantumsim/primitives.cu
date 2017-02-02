/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/

#include <cuda.h> 

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
//and collects the real or values only; furthermore it rearranges the address bit order 
//from (d_state_bits, d_state_bits) to 
// (alpha_d, alpha_d-1, ..., alpha_0) where alpha = (00, 01, 10, 11) for 0, x, y, 1
//if direction = 0, the copy is performed from complex to real, otherwise from real to complex
__global__ void pauli_reshuffle(double *complex_dm, double *real_dm, unsigned int no_qubits, unsigned int direction) {

    const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;


    //do we need imaginary part? That is the case if we have an odd number of bits for y in our adress (bit in y is 1, bit in x is 0)
    unsigned int v = ~x & y;


    unsigned int py = 0;
    while (v) {
        py += v&1;
        v >>= 1;
    }

    py = py & 0x3;

    //short version: while (v>1) { v = (v >> 1) ^ v ;}
    //bit bang version
    /*v ^= v >> 1;*/
    /*v ^= v >> 2;*/
    /*v = (v & 0x11111111U) * 0x11111111U;*/
    /*v = (v >> 28) & 1;*/

    const unsigned int addr_complex = (((x << no_qubits) | y) << 1) + (py&1);


    //the adress in pauli basis is obtained by interleaving
    unsigned int addr_real = 0;
    for (int i = 0; i < 16; i++) { 
          addr_real |= (x & 1U << i) << i | (y & 1U << i) << (i + 1);
    }
    

    if(direction == 0) {
        real_dm[addr_real] = ((py==3 || py==2)? -1 : 1)*complex_dm[addr_complex];
    }
    else {
        complex_dm[addr_complex] = ((py==3 || py == 2)? -1 : 1)*real_dm[addr_real];
    }
}



__global__ void two_qubit_general_ptm(double *dm, double *ptm_g, 
        unsigned int dim_a, unsigned int stride_a,
        unsigned int dim_b, unsigned int stride_b,
        unsigned int dim_rho) {

    const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;


    // external memory required: (blockDim.x + dim_a*dim_b) double floats
    extern __shared__ double ptm[];
    double *data = &ptm[dim_a*dim_b]; 

    // load ptm to shared memory (ptm should be smaller than block, but in case it is not, loop here)
    for(int i=0; i < dim_a*dim_b; i+=blockDim.x) {
        if(i+threadIdx.x < dim_a*dim_b) {
            ptm[i+threadIdx.x] = ptm_g[i+threadIdx.x];
        }
    }

    if (idx >= dim_rho) return;

    //adress calculation
    //the index is of the form idx = X Y Z ib ia, 
    //where the address is of the form addr = X ib Y ia Z
    //this integer arithmetic is possibly is quite slow. one might want to do it in float instead

    unsigned int i = idx;
    unsigned int reduced_stride_of_y = stride_b/(stride_a*dim_a);
    unsigned int idx_a = i % dim_a;
    i = i / dim_a;
    unsigned int idx_b = i % dim_b;
    i = i / dim_b;
    unsigned int     z = i % (stride_a);
    i = i / stride_a;
    unsigned int     y = i % reduced_stride_of_y;
    unsigned int     x = i / reduced_stride_of_y;

    unsigned int addr = z + stride_a*idx_a + stride_a*dim_a*y + stride_b*idx_b + stride_b*dim_b*x;

    //fetch data to memory
    //data[threadIdx.x] = dm[addr];
    __syncthreads();


    /*int row = idx_b*dim_b + idx_a;//000 ib ia;*/
    /*int offset = idx - row;          //x y z00;*/

    double acc=1;
    /*for(int i=0; i<dim_a*dim_b; i++) {*/
        /*acc += ptm[dim_a*dim_b*row + i]*data[offset+i];*/
    /*}*/

    //upload back to global memory
    __syncthreads();
    dm[addr] = acc;
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


__global__ void two_qubit_ptm(double *dm, double *ptm_g, unsigned int bit0, unsigned int bit1, unsigned int no_qubits) {
    const unsigned int x = threadIdx.x;
    const unsigned int high_x = blockIdx.x * blockDim.x;



    extern __shared__ double ptm[];
    double *data = &ptm[256]; //need blockDim.x double floats

    // the lowest to bits of x are used to address bit0, the next two are used to address bit1 
    // global address = <- pos = 
    // aaaxxbbbbyycccc  <- aaabbbbccccxxyy

    int higher_bit = max(bit0, bit1);
    int lower_bit = min(bit0, bit1);
    int high_mask = ~ ( (1 << (2*higher_bit+2)) - 1 ); //a mask (of pos)
    int mid_mask = (~ ( (1 << (2*lower_bit + 4)) - 1)) & (~high_mask);  //b mask
    int low_mask  = ~(high_mask | mid_mask) & (~0xf);  //c mask

    int pos = high_x | x;
    int global_from = 
              (pos & high_mask) 
            | ((pos & mid_mask) >> 2)
            | ((pos & low_mask) >> 4)
            | ((pos & 0x3) << (2 * bit0))  
            | (((pos & 0xc) >>2)  << (2 * bit1));

    //fetch ptm to shared memmory
    //need to fetch several values per thread if blockDim.x is less than 256 (only for small dms...)
    for(int i=0; i < 256; i+=blockDim.x) {
        if(i+x < 256) {
            ptm[i+x] = ptm_g[i+x];
        }
    }
    if (high_x + x >= (1 << (2*no_qubits))) return;


    //fetch data block to shared memory
    data[x] = dm[global_from];
    __syncthreads();

    unsigned int row = x & 0xf;
    unsigned int idx = x & ~0xf;

    double acc=0;
    for(int i=0; i<16; i++) {
        acc += ptm[16*row + i]*data[idx+i];
    }


    __syncthreads();
    dm[global_from] = acc;

}


//copy the two diagonal blocks of one ancilla into reduced density matrices
//the qubit index is passed as an integer, not as a bitmask!
__global__ void dm_reduce(double *dm, unsigned int bit, double *dm0, unsigned int state,
        unsigned int no_qubits) {

    const int addr = blockIdx.x*blockDim.x + threadIdx.x;

    if(addr >= (1<< (2*no_qubits))) return;

    const int low_mask = (1 << (2*bit))-1;      //0000011111
    const int high_mask = (~low_mask) << 2;     //1110000000

    if(((addr >> (2*bit)) & 0x3) == state*0x3) {
        dm0[ (addr & low_mask) | ((addr & high_mask) >> 2) ] = dm[addr];
    }
}



//get_diagonal kernel
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

//trace kernel. Calculate the sum of a diagonal, must run in one block!
//shared memory: 2**no_qubits doubles
//if bit is positive or zero, diag[0] and diag[1] will hold the partial traces of this bit being one/zero (!note the switch)
//if bit is -1, diag[0] will hold the full trace.
__global__ void trace(double *diag, int bit) { 
    unsigned int x = threadIdx.x;
    unsigned int mask = 0;

    if(bit >= 0) {
        mask = 1 << bit;
    }

    extern __shared__ double s_diag[];
    s_diag[x] = diag[x];
    __syncthreads(); 

    double a;

    for(unsigned int i=1; i < blockDim.x; i <<= 1) {
        if(i != mask && i <= x) { 
            a = s_diag[x-i];
        
        }
        __syncthreads();
        if(i != mask && i <= x) { 
            s_diag[x] += a;
        }
        __syncthreads();
    }

    __syncthreads();
    //copy result back
    if(x == 0) {
        diag[blockIdx.x] = s_diag[blockDim.x - 1];
        return;
    }
    if(x == 1 && bit >= 0) {
        diag[blockIdx.x + 1] = s_diag[blockDim.x - 1 - mask];
        return;
    }
}

//swap kernel
//exchange two qubits. The only purpose of this kernel is to arrange a certain qubit as to be the most significant so that
//projection is trivial. Actual swap gates should be implemented by relabeling!
__global__ void swap(double *dm, unsigned int bit1, unsigned int bit2, unsigned int no_qubits) {
    unsigned int addr = threadIdx.x + blockDim.x*blockIdx.x;

    if (addr >= (1<<2*no_qubits)) return;

    unsigned int bit1_mask = (0x3 << (2*bit1));
    unsigned int bit2_mask = (0x3 << (2*bit2));
    
    unsigned int addr2 = ( addr & ~(bit1_mask | bit2_mask)) |
        ((addr & bit1_mask) << (2*(bit2 - bit1))) |
        ((addr & bit2_mask) >> (2*(bit2 - bit1)));
   
    double t;
    if (addr > addr2) {
        t = dm[addr2];
        dm[addr2] = dm[addr];
        dm[addr] = t;
    }
}
