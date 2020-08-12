/*This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)*/
/*(c) 2020 Brian Tarasinski, Viacheslav Ostroukh, Boris Varbanov*/
/*Distributed under the GNU GPLv3. See LICENSE or https://www.gnu.org/licenses/gpl.txt*/

//multitake
//given a list of index lists `idx` in sparse format
//idx_j = flatten(idx), idx_i = cumsum(len(idx))
//and in and out array with shapes
//as well as dim = len(inshape) = len(outshape)
//set out = in[np.ix_(idx)]

//indices are given in C order, i.e. most significant first
__kernel void multitake(__global const double *restrict in,
        __global double *restrict out,
        __global const unsigned int *restrict idx_i,
        __global const unsigned int *restrict idx_j,
        __global const unsigned int *restrict inshape,
        __global const unsigned int *restrict outshape,
        unsigned int dim) {

    unsigned int addr_out, addr_in, s;
    unsigned int i, ia, ja;

    int acc;

    acc = addr_out = get_global_id(0);
    addr_in = 0;
    s = 1;

    for(i=dim; i > 0;) {
        i--;
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