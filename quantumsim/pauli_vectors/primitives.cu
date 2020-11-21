/*This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)*/
/*(c) 2020 Brian Tarasinski, Viacheslav Ostroukh, Boris Varbanov*/
/*Distributed under the GNU GPLv3. See LICENSE or https://www.gnu.org/licenses/gpl.txt*/

#include <cuda.h>


//Apply a general pauli transfer matrix, to (up to) two subsystems (arbitrary dimension)
//a is the msb, b is the lsb.
//If the PTM is diagonal, this works in-place, i.e. with dm_in == dm_out. Otherwise NOT!
//You need to give the dimensions of two of the intermediate bystander spaces.

//it is also important that a must be the msb in the ptm, but also the msb in the density matrix.
//if not, you must reshape the ptm (switch a and b) before calling the kernel.
__global__ void two_qubit_general_ptm(
        double *dm_in, double *dm_out,
        const double * __restrict__ ptm_g,
        unsigned int dim_a_in,
        unsigned int dim_b_in,
        unsigned int dim_z, unsigned int dim_y,
        unsigned int dim_rho) {

    /*const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;*/
    /*if (idx >= dim_rho) return;*/


    //(structure worked out in Notebook II, p. 203 ff)

    //blockDim.x = dim_a_out
    //blockDim.y = dim_b_out
    //blockDim.z = d_internal (arbitrary, to make blocksize and shared memory reasonable)

    const unsigned int dim_a_out = blockDim.x;
    const unsigned int dim_b_out = blockDim.y;
    const unsigned int d_internal = blockDim.z;

    const unsigned int ax = threadIdx.x;
    const unsigned int bx = threadIdx.y;
    const unsigned int zx = threadIdx.z;
    const unsigned int ix = blockIdx.x;

    unsigned int xyz, x, y, z;
    xyz = ix*d_internal + zx;
    z = xyz % dim_z;
    xyz /= dim_z;
    y = xyz % dim_y;
    xyz /= dim_y;
    x = xyz;
    // can do/need termination statement here?

    // external memory required: (blockDim.x + dim_a*dim_b**2) double floats
    extern __shared__ double data[];

    // load ptm to shared memory

    const int row = (ax*dim_b_out + bx)*dim_b_in*dim_a_in;
    /*for(int g = zx; g < dim_a_in*dim_b_in; g += d_internal) {*/
        /*ptm[row + g] = ptm_g[row + g];*/
    /*}*/



    // load data to memory
    const int column = zx*dim_a_in*dim_b_in;
    unsigned int addr_in;
    for(int ai = ax; ai < dim_a_in; ai += dim_a_out) {
        for(int bi = bx; bi < dim_b_in; bi += dim_b_out) {
            addr_in = (((x*dim_a_in + ai)*dim_y + y)*dim_b_in + bi)*dim_z + z;
            data[column + ai*dim_b_in + bi] = dm_in[addr_in];
        }
    }
    const int addr_out = (((x*dim_a_out + ax)*dim_y + y)*dim_b_out + bx)*dim_z + z;

    if (addr_out >= dim_rho)
        return;

    //done loading
    __syncthreads();

    //calculate the vector product
    double acc=0.0;
    for(int delta=0; delta < dim_a_in*dim_b_in; delta++) {
        acc += ptm_g[row + delta]*data[column + delta];
    }

    //upload back to global memory
    __syncthreads();
    dm_out[addr_out] = acc;
}


//multitake
//given a list of index lists `idx` in sparse format
//idx_j = flatten(idx), idx_i = cumsum(len(idx))
//and in and out array with shapes
//as well as dim = len(inshape) = len(outshape)
//set out = in[np.ix_(idx)]

//indices are given in C order, i.e. most significant first
__global__ void multitake(const double * __restrict__ in,
        double * __restrict__ out,
        const unsigned int * __restrict__ idx_i,
        const unsigned int * __restrict__ idx_j,
        const unsigned int * __restrict__ inshape,
        const unsigned int * __restrict__ outshape,
        unsigned int dim) {

    unsigned int addr_out, addr_in, s;
    unsigned int i, ia, ja;

    int acc;

    acc = addr_out = blockDim.x*blockIdx.x + threadIdx.x;
    addr_in = 0;
    s = 1;

    for(i=dim; i > 0;) {
        i--;
        /*ia = rint(remquo((double)acc, (double)outshape[i], &acc));*/
        ia = acc % outshape[i];
        acc = acc / outshape[i];
        ja = idx_j[idx_i[i] + ia];
        addr_in += ja*s;
        s *= inshape[i];
    }

    // guard
    if(acc == 0)
        out[addr_out] = in[addr_in];
}
