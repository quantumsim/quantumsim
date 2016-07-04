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
__global__ void bit_to_pauli_basis(double *complex_dm, unsigned int mask, unsigned int no_qubits) {
    const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    const double sqrt2 =  0.70710678118654752440;

    if (x == y) return;

    int b_addr = ((x|mask)<<no_qubits | y) << 1;
    int c_addr = (x<<no_qubits | (y|mask)) << 1;

    if (x < y) {
        b_addr += 1;
        c_addr += 1;
    }

    double b = complex_dm[b_addr];
    double c = complex_dm[c_addr];

    complex_dm[b_addr] = (b+c)*sqrt2;
    complex_dm[c_addr] = (b-c)*sqrt2;
}


//do the cphase gate. set a bit in mask to 1 if you want the corresponding qubit to 
//partake in the cphase.
//run in a 2d grid that stretches over dm10
__global__ void cphase(double *dm10, unsigned int mask, unsigned int no_qubits) {
    const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    //if exactly one of x,y has all bits in mask set
    if (((x & mask) == mask) != ((y & mask) == mask)) {

        if (x <= y) { //real part
            dm10[ADDR_REAL(x,y,no_qubits)] = -dm10[ADDR_REAL(x,y,no_qubits)];
        }
        else { //imaginary part
            dm10[ADDR_IMAG(y,x,no_qubits)] = -dm10[ADDR_IMAG(y,x,no_qubits)];
        }
    }
}



//do a rotation around the y axis on the gate specified by bit: 
//sine and cosine are the sine and cosine of the angle over two!: 
// U = [[cos, sin], [-sin, cos]]
__global__ void rotate_y(double *dm10, unsigned int mask, double cosine, double sine, unsigned int no_qubits) { 

    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    if((x&mask) && ((~y)&mask)) { //real part
        x = x & ~mask;
        if (x <= y) {
            double a = dm10[ADDR_REAL(x, y, no_qubits)];
            double b = dm10[ADDR_REAL(x|mask, y, no_qubits)];
            double c = dm10[ADDR_REAL(x, y|mask, no_qubits)];
            double d = dm10[ADDR_REAL(x|mask, y|mask, no_qubits)];

            double new_a = cosine*a + sine*b;
            double new_b = -sine*a + cosine*b;
            double new_c = cosine*c + sine*d;
            double new_d = -sine*c + cosine*d;

            a = cosine*new_a + sine*new_c;
            b = cosine*new_b + sine*new_d;
            c = -sine*new_a + cosine*new_c;
            d = -sine*new_b + cosine*new_d;


            dm10[ADDR_REAL(x, y, no_qubits)] = a;
            dm10[ADDR_REAL(x|mask, y, no_qubits)] = b;
            dm10[ADDR_REAL(x, y|mask, no_qubits)] = c;
            dm10[ADDR_REAL(x|mask, y|mask, no_qubits)] = d;
        }
    }
    if (((~x)&mask) && (y&mask)) { //do the imaginary part
        y = y & ~mask;
        if (y <= x){
            double a = dm10[ADDR_IMAG(y, x, no_qubits)];
            double b = dm10[ADDR_IMAG(y|mask, x, no_qubits)];
            double c = dm10[ADDR_IMAG(y, x|mask, no_qubits)];
            double d = dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)];

            if (y|mask >= x) b*=-1;

            double new_a = cosine*a + sine*b;
            double new_b = -sine*a + cosine*b;
            double new_c = cosine*c + sine*d;
            double new_d = -sine*c + cosine*d;

            a = cosine*new_a + sine*new_c;
            b = cosine*new_b + sine*new_d;
            c = -sine*new_a + cosine*new_c;
            d = -sine*new_b + cosine*new_d;

            if (y|mask >= x) b*=-1;

            dm10[ADDR_IMAG(y, x, no_qubits)] = a;
            dm10[ADDR_IMAG(y|mask, x, no_qubits)] = b;
            dm10[ADDR_IMAG(y, x|mask, no_qubits)] = c;
            dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)] = d;
        }
    }

}

__global__ void rotate_z(double *dm10, unsigned int mask, double cos, double sin, unsigned int no_qubits) { 

    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    if (y>x) return;

    if ((x&mask) && ((~y)&mask)) { //do the b block
            double b_re = dm10[ADDR_REAL(x, y, no_qubits)];
            double b_im = dm10[ADDR_IMAG(x, y, no_qubits)];

            double nb_re = cos*b_re + sin*b_im;
            double nb_im = cos*b_im - sin*b_re;

            dm10[ADDR_REAL(x, y, no_qubits)] = nb_re;
            dm10[ADDR_IMAG(x, y, no_qubits)] = nb_im;
    }
    if (((~x)&mask) && (y&mask)) { //do the c block
            double b_re = dm10[ADDR_REAL(x, y, no_qubits)];
            double b_im = dm10[ADDR_IMAG(x, y, no_qubits)];

            double nb_re = cos*b_re - sin*b_im;
            double nb_im = cos*b_im + sin*b_re;

            dm10[ADDR_REAL(x, y, no_qubits)] = nb_re;
            dm10[ADDR_IMAG(x, y, no_qubits)] = nb_im;
    }

}

__global__ void rotate_x(double *dm10, unsigned int mask, double cos, double sin, unsigned int no_qubits) { 

    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    if((x&mask) && ((~y)&mask)) { //real part
        x = x & ~mask;
        if (x <= y) {
            double a = dm10[ADDR_REAL(x, y, no_qubits)];
            double b = dm10[ADDR_IMAG(x|mask, y, no_qubits)];
            double c = dm10[ADDR_IMAG(x, y|mask, no_qubits)];
            double d = dm10[ADDR_REAL(x|mask, y|mask, no_qubits)];

            if (x|mask >= y) b*=-1;

            double na = cos*cos*a + sin*cos*(b-c) + sin*sin*d;
            double nb = sin*cos*(d-a) + sin*sin*c + cos*cos*b;
            double nc = sin*cos*(a-d) + sin*sin*b + cos*cos*c;
            double nd = cos*cos*d + sin*cos*(c-b) + sin*sin*a;

            if (x|mask >= y) nb*=-1;

            dm10[ADDR_REAL(x, y, no_qubits)] = na;
            dm10[ADDR_IMAG(x|mask, y, no_qubits)] = nb;
            dm10[ADDR_IMAG(x, y|mask, no_qubits)] = nc;
            dm10[ADDR_REAL(x|mask, y|mask, no_qubits)] = nd;
        }
    }
    if (((~x)&mask) && (y&mask)) { //do the imaginary part
        y = y & ~mask;
        if (y <= x){
            double a = dm10[ADDR_IMAG(y, x, no_qubits)];
            double b = dm10[ADDR_REAL(y|mask, x, no_qubits)];
            double c = dm10[ADDR_REAL(y, x|mask, no_qubits)];
            double d = dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)];

            double na = cos*cos*a - sin*cos*(b-c) + sin*sin*d;
            double nb = sin*cos*(a-d) + sin*sin*c + cos*cos*b;
            double nc = sin*cos*(d-a) + sin*sin*b + cos*cos*c;
            double nd = cos*cos*d - sin*cos*(c-b) + sin*sin*a;

            dm10[ADDR_IMAG(y, x, no_qubits)] = na;
            dm10[ADDR_REAL(y|mask, x, no_qubits)] = nb;
            dm10[ADDR_REAL(y, x|mask, no_qubits)] = nc;
            dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)] = nd;
        }
    }

}

//do the hadamard on a qubit
//mask must have exactly one bit flipped, denoting which byte is involved
//the results is multiplied with mul, to obtain trace preserving map, set mul = 0.5
__global__ void hadamard(double *dm10, unsigned int mask, double mul, unsigned int no_qubits) { 

    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;

    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    if((x&mask) && ((~y)&mask)) { //real part
        x = x & ~mask;
        if (x <= y) {
            double a = dm10[ADDR_REAL(x, y, no_qubits)];
            double b = dm10[ADDR_REAL(x|mask, y, no_qubits)];
            double c = dm10[ADDR_REAL(x, y|mask, no_qubits)];
            double d = dm10[ADDR_REAL(x|mask, y|mask, no_qubits)];

            double new_a = a+b+c+d;
            double new_b = a-b+c-d;
            double new_c = a+b-c-d;
            double new_d = a-b-c+d;

            dm10[ADDR_REAL(x, y, no_qubits)] = mul*new_a;
            dm10[ADDR_REAL(x|mask, y, no_qubits)] = mul*new_b;
            dm10[ADDR_REAL(x, y|mask, no_qubits)] = mul*new_c;
            dm10[ADDR_REAL(x|mask, y|mask, no_qubits)] = mul*new_d;
        }
    }
    if (((~x)&mask) && (y&mask)) { //do the imaginary part
        y = y & ~mask;
        if (y <= x){
            double a = dm10[ADDR_IMAG(y, x, no_qubits)];
            double b = dm10[ADDR_IMAG(y|mask, x, no_qubits)];
            double c = dm10[ADDR_IMAG(y, x|mask, no_qubits)];
            double d = dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)];

            double new_a = a+b+c+d;
            double new_b = a-b+c-d;
            double new_c = a+b-c-d;
            double new_d = a-b-c+d;

            dm10[ADDR_IMAG(y, x, no_qubits)] = mul*new_a;
            dm10[ADDR_IMAG(y|mask, x, no_qubits)] = mul*new_b;
            dm10[ADDR_IMAG(y, x|mask, no_qubits)] = mul*new_c;
            dm10[ADDR_IMAG(y|mask, x|mask, no_qubits)] = mul*new_d;
        }
    }
}


// amplitude damping and phase damping on one qubit
// mask: bit mask selecting the qbit
// gamma = probability for amplitude decay, i.e gamma = 1-exp(-t/T1)
// s1mgamma = sqrt(1-gamma)
// s1mlambda = sqrt(1-lambda), where lambda is the probability for a phase flip, i.e. lambda = 1- exp(-t/T2)
__global__ void amp_ph_damping(double *dm10, unsigned int mask, double gamma, double s1mgamma, double s1mlambda, unsigned int no_qubits) {

    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;
    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    int ri_flag = 1;

    if (x >= y) ri_flag = 0;


    double f = dm10[ADDR_TRIU(x, y, ri_flag, no_qubits)];

    if (x&y&mask) { //c block
        dm10[ADDR_TRIU(x^mask,y^mask,ri_flag, no_qubits)]  += gamma * f;
        f = f - gamma*f;
    } 
    else if ((~x)&(~y)&mask) {
        return;
    }
    else { // b block
        f *= s1mgamma * s1mlambda;
    }

    dm10[ADDR_TRIU(x, y, ri_flag, no_qubits)] = f;
}

//copy the two diagonal blocks of one ancilla into reduced density matrices
//multiply the two with two numbers (inverse of the traces for instance, to implement measurement)
//note that because major bit banging is required to figure out the new adresses,
//the qubit index is passed as an integer, not as a bitmask!

__global__ void dm_reduce(double *dm10, unsigned int bit_idx, double *dm9_0, double *dm9_1, double mul0, double mul1, unsigned int no_qubits) {
    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y = (blockIdx.y *blockDim.y) + threadIdx.y;
    if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

    int ri_flag = 1;
    if (x >= y) ri_flag = 0;

    unsigned int mask = (1 << bit_idx);                  // e.g. 00010000 if bit_idx == 4

    int block; //0 or 1?
    
    if(x&y&mask) block = 1;
    else if((~x)&(~y)&mask) block = 0;
    else return;

    unsigned int lower_mask =  (1 << bit_idx) - 1;       // e.g. 00001111
    unsigned int upper_mask = ~((1 << (bit_idx+1)) - 1); // e.g. 11100000

    unsigned int x9, y9; //new adresses

    x9 =  ((x&upper_mask) >> 1) | (x & lower_mask);
    y9 =  ((y&upper_mask) >> 1) | (y & lower_mask);

    
    if(block == 0) {
        dm9_0[ADDR_TRIU(x9, y9, ri_flag, no_qubits-1)] = mul0*dm10[ADDR_TRIU(x,y,ri_flag, no_qubits)];
    }
    if(block == 1) {
        dm9_1[ADDR_TRIU(x9, y9, ri_flag, no_qubits-1)] = mul1*dm10[ADDR_TRIU(x,y,ri_flag, no_qubits)];
    }
}



//trace kernel
//copy the diagonal elements to out, in order to do effective 
//calculation of subtraces.
//run over a 1x9 grid!
__global__ void get_diag(double *dm9, double *out, unsigned int no_qubits) {
    int x = (blockIdx.x *blockDim.x) + threadIdx.x;
    if (x >= (1 << no_qubits)) return;
    out[x] = dm9[ADDR_BARE(x,x,0,no_qubits)];
}

//inverse of dm_reduce
//run over 9x9 grid!
__global__ void dm_inflate(double *dm10, unsigned int bit_idx, double *dm9_0, double *dm9_1, unsigned int no_qubits) {
    int x9 = (blockIdx.x *blockDim.x) + threadIdx.x;
    int y9 = (blockIdx.y *blockDim.y) + threadIdx.y;
    if (x9 >= (1 << no_qubits-1) || y9 >= (1 << no_qubits-1)) return;

    int ri_flag = 1;
    if (x9 >= y9) ri_flag = 0;

    unsigned int mask = (1 << bit_idx);                  // e.g. 00010000 if bit_idx == 4

    unsigned int lower_mask =  (1 << bit_idx) - 1;       // e.g. 00001111
    unsigned int upper_mask = ~((1 << bit_idx) - 1);     // e.g. 11110000

    //calculate new adresses
    unsigned int x, y;

    x =  ((x9&upper_mask) << 1) | (x9 & lower_mask);
    y =  ((y9&upper_mask) << 1) | (y9 & lower_mask);

    dm10[ADDR_TRIU(x, y, ri_flag, no_qubits)] = dm9_0[ADDR_TRIU(x9, y9, ri_flag, no_qubits-1)];
    dm10[ADDR_TRIU(x|mask, y|mask, ri_flag, no_qubits)] = dm9_1[ADDR_TRIU(x9, y9, ri_flag, no_qubits-1)];
}
