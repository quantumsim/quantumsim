#include <cuda.h> 
#include <cuComplex.h>

// the block size is 1<<LOG_BK_SIZE in x and y direction
#define LOG_BK_SIZE (5)

//this defines the scheme by which the density matrix is stored in a dm10
//we always require x <= y for the real part and
#define ADDR_BARE(x,y,ri_flag,n) (((((x)<<(n)) | (y)) << 1) | (ri_flag))
#define ADDR_TRIU(x,y,ri_flag,n) (((x) <= (y)) ? ADDR_BARE(x,y,ri_flag,n) : ADDR_BARE(y,x,ri_flag,n))
#define ADDR_REAL(x,y,n) ADDR_TRIU(x,y,0,n)
#define ADDR_IMAG(x,y,n) ADDR_TRIU(x,y,1,n)



//do the cphase gate. set a bit in mask to 1 if you want the corresponding qubit to 
//partake in the cphase.
//run in a 2d grid that stretches over dm10
__global__ void cphase(double *dm10, unsigned int mask, unsigned int no_qubits) {
    const int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    const int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

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


//do the hadamard on a qubit
//mask must have exactly one bit flipped, denoting which byte is involved
//the results is multiplied with mul, to obtain trace preserving map, set mul = 0.5
__global__ void hadamard(double *dm10, unsigned int mask, double mul, unsigned int no_qubits) { 

    int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

    if((x&mask) && (~y&mask)) { //real part
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
    if ((~x&mask) && (y&mask)) { //do the imaginary part
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

    int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

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
    int x = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

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
    int x = (blockIdx.x <<  LOG_BK_SIZE) + threadIdx.x;
    out[x] = dm9[ADDR_BARE(x,x,0,no_qubits)];
}

//inverse of dm_reduce
//run over 9x9 grid!
__global__ void dm_inflate(double *dm10, unsigned int bit_idx, double *dm9_0, double *dm9_1, unsigned int no_qubits) {
    int x9 = (blockIdx.x << LOG_BK_SIZE) + threadIdx.x;
    int y9 = (blockIdx.y << LOG_BK_SIZE) + threadIdx.y;

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
