/*  The source code is modified from https://github.com/GreenWaves-Technologies/benchmarks
** 
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "pulp.h"
#include "rt/rt_api.h"

#define NOGPIO

#define FULL_PRECISION float

// This defines how many loops the fuctions execute
#define ITERATIONS 100


#define ALIM_1_VOLT 0
#define FREQ_FC (250*1000000)
#define FREQ_CL (175*1000000)


// The benchmark names
char tests_names[][50] = {
{"Standard-Full-Precison"},
{"Parallel-Full-Precison"},
	
{"Scaled-XNOR-Net-Conv"},
{"Parallel-Scaled-XNOR-Net"},

{"XOR-Net-S-layer=1"},
{"Parallel-XOR-Net-S-1"},

{"CI-BCNN"},
{"Parallel-CI-BCNN"},

{"XOR-Net"},
{"Parallel-XOR-Net"}
};




char tests_titles[][50] = {
{"Full Precision"},
{"Original XNOR-Net Algorithm"},
{"XOR-Net-S Algorithm"},
{"CI-BCNN"},
{"XOR-Net"}
};



#define ALIGN(Value, Size)      (((Value)&((1<<(Size))-1))?((((Value)>>(Size))+1)<<(Size)):(Value))

#ifndef RISCV
#define Min(a, b)       __builtin_pulp_minsi((a), (b))
#define Max(a, b)       __builtin_pulp_maxsi((a), (b))
#else
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))
#endif

#ifdef __EMUL__
#define plp_cluster_fetch(a)
#define plp_cluster_wait(a)
#endif



#ifdef NOGPIO
#define WriteGpio(a, b)
#else
#define WriteGpio(a, b) rt_gpio_set_pin_value(0, a, b)
#endif

#ifdef RISCV
#define TOT_TEST 1
int test_num[TOT_TEST] = { 5 };

#ifndef __EMUL__
#endif


#define L2_MEM                          __attribute__((section(".heapl2ram")))
#define L1_CL_MEM                       __attribute__((section(".heapsram")))
#define L1_FC_MEM                       __attribute__((section(".fcTcdm")))
/* HW timer */
#define ARCHI_FC_TIMER_ADDR                     ( ARCHI_FC_PERIPHERALS_ADDR + ARCHI_FC_TIMER_OFFSET  )
#define PLP_TIMER_VALUE_LO                      0x08
#define PLP_TIMER_CFG_REG_LO                    0x00
#define PLP_TIMER_ENABLE_BIT            0
#define PLP_TIMER_RESET_BIT             1
#define PLP_TIMER_IRQ_ENABLE_BIT        2
#define PLP_TIMER_IEM_BIT               3
#define PLP_TIMER_CMP_CLR_BIT           4
#define PLP_TIMER_ONE_SHOT_BIT          5
#define PLP_TIMER_PRESCALER_ENABLE_BIT  6
#define PLP_TIMER_CLOCK_SOURCE_BIT      7
#define PLP_TIMER_PRESCALER_VALUE_BIT   8
#define PLP_TIMER_PRESCALER_VALUE_BITS  8
#define PLP_TIMER_64_BIT                31

#define plp_timer_conf_get(a,b,c,d,e,f,g,h,i)      ((a << PLP_TIMER_ENABLE_BIT) \
        | (b << PLP_TIMER_RESET_BIT) \
        | (c << PLP_TIMER_IRQ_ENABLE_BIT) \
        | (d << PLP_TIMER_IEM_BIT) \
        | (e << PLP_TIMER_CMP_CLR_BIT) \
        | (f << PLP_TIMER_ONE_SHOT_BIT) \
        | (g << PLP_TIMER_PRESCALER_ENABLE_BIT) \
        | (h << PLP_TIMER_PRESCALER_VALUE_BIT) \
        | (i << PLP_TIMER_64_BIT) \
        )
#define gap8_resethwtimer()                     pulp_write32(ARCHI_FC_TIMER_ADDR + PLP_TIMER_CFG_REG_LO, plp_timer_conf_get(1,1,0,0,0,0,0,0,0))
#define gap8_readhwtimer()                      pulp_read32(ARCHI_FC_TIMER_ADDR + PLP_TIMER_VALUE_LO)

#else
#define TOT_TEST 5
int test_num[TOT_TEST] = { 2,2,2,2,2 };
#include "Gap8.h"

static int CoreCountDynamic = 1;
static int ActiveCore = 8;

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap8_ncore();
	//Log2Core = gap8_fl1(NCore);
	Log2Core=3;
	Chunk = (X >> Log2Core)  + ((X&(NCore - 1)) != 0);
	return Chunk;
}
#endif

#define STACK_SIZE      2048
#define MOUNT           1
#define UNMOUNT         0
#define CID             0


typedef struct ClusterArg {
	int test_num;
	int Iter;
	int Iter_operations;
	int H;
	int W;
	int C;
	int Filternum;
} ClusterArg_t;

ClusterArg_t Arg;


char str[100];
static char *float_to_string(float in) {

	int d1 = in;
	float f2 = in - d1;
	int d2 = trunc(f2 * 10000);

	sprintf(str, "%d.%04d", d1, d2);
	return str;
}

#ifndef RISCV
#define B_ins(dst, src, size, off)      gap8_bitinsert(dst, src, size, off)
#define B_ins_r(dst, src, size, off)    gap8_bitinsert_r(dst, src, size, off)
#define B_ext(x, size, off)             gap8_bitextract(x, size, off)
#define B_extu(x, size, off)            gap8_bitextractu(x, size, off)
#define B_ext_r(x, size, off)           gap8_bitextract_r(x, size, off)
#define B_extu_r(x, size, off)          gap8_bitextractu_r(x, size, off)
#define B_popc(src)                     __builtin_popcount((src))
#else
static __attribute__((always_inline)) unsigned int bitcount32(unsigned int b)
{
	b -= (b >> 1) & 0x55555555;
	b = (b & 0x33333333) + ((b >> 2) & 0x33333333);
	b = (b + (b >> 4)) & 0x0f0f0f0f;
	return (b * 0x01010101) >> 24;
}
#define B_ins(dst, src, size, off)      (((dst) & ~(((1<<(size))-1)<<(off))) | (((src) & ((1<<(size))-1))<<(off)))
#define B_ins_r(dst, src, size, off)    (((dst) & ~(((1<<(size))-1)<<(off))) | (((src) & ((1<<(size))-1))<<(off)))
#define B_ext(x, size, off)             (((((x)>>(off))&((unsigned int)(1<<(size))-1))<<(32-(size)))>>(32-(size)))
#define B_extu(x, size, off)            (((x)>>(off))&((unsigned int)(1<<(size))-1))
#define B_ext_r(x, size, off)           (((((x)>>(off))&((unsigned int)(1<<(size))-1))<<(32-(size)))>>(32-(size)))
#define B_extu_r(x, size, off)          (((x)>>(off))&((unsigned int)(1<<(size))-1))
#define B_popc(src)                     bitcount32((src))
#endif

#define VSOC	1000
#define GPIO	17



/****************************************************
 * Memory allocation, I change this to use L2 cache
 * L1_CL_MEM 64KB
 * L1_FC_MEM 8KB
 * L2_MEM 512KB
 * Allocate Only 480*1000 BYTE
 * **************************************************
 */
/* The Memory Allocation By Greenwaves 
#ifdef BYTE
#define MAX_MEM	55000
#else
#define MAX_MEM	(55000/2)
#endif
*/
#define MAX_MEM	(480000/4)
FULL_PRECISION L2_MEM Mem[MAX_MEM];



typedef struct {
    FULL_PRECISION *__restrict__ In;
    int H;
    int W;
    int C;
    int Filternum;
    FULL_PRECISION *__restrict__ Filter;
    FULL_PRECISION *__restrict__ Out;
} ArgConvTnew;

typedef struct {
    FULL_PRECISION *__restrict__ Input;
    int H;
    int W;
    int C;
    int Filternum;
    unsigned int *__restrict__ PackedInput;
    unsigned int *__restrict__ Filter;
    FULL_PRECISION *__restrict__ Ffactor;
    FULL_PRECISION *__restrict__ Out;
    int layer;
    FULL_PRECISION *__restrict__ Kmatrix;
} ArgConvTxor;


void CheckMem(int Size)

{
	if (Size > MAX_MEM) {
		printf("Memory Overflow (%d>%d). Aborting\n", Size, MAX_MEM); exit(0);
	}
}




/****************************************************************************************************
 * The functions for benchmarking. You can extract the 5 sequential version XOR-Net and replace the B_ins_r and B_popc functions to run on Windows/Linux.
 * Full Precision 3X3 Convolution
 * Original XNOR-Net 3X3 Conv
 * Optimized XOR-Net-S 3X3 Conv
 * Original BCNN 3X3 Conv
 * Optimized XOR-Net 3X3 Conv
 * **************************************************************************************************
 */
// Full Precision 3X3 Conv
void __attribute__((noinline)) FP3X3Convolution(FULL_PRECISION  *__restrict__ In, int H, int W, int C, int Filternum,
        FULL_PRECISION  *__restrict__ Filter, FULL_PRECISION  *__restrict__ Out)
{
    int Wo=W-2;
    int Ho=H-2;
    int KW=3, KH=3;

    for (int fn = 0; fn < Filternum; fn++) {
         for (int ih = 0; ih < (H - 2); ih++) {
            for (int iw = 0; iw < (W - 2); iw++) {
                FULL_PRECISION R =Out[fn * Wo * Ho + ih * Wo + iw]= 0;
                // Use C, H, W format for data locality
                for (int kc = 0; kc < C; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // We use the C, H, W format
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 0]*In[kc * W * H + (ih + 0) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 1]*In[kc * W * H + (ih + 0) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 2]*In[kc * W * H + (ih + 0) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 0]*In[kc * W * H + (ih + 1) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 1]*In[kc * W * H + (ih + 1) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 2]*In[kc * W * H + (ih + 1) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 0]*In[kc * W * H + (ih + 2) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 1]*In[kc * W * H + (ih + 2) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 2]*In[kc * W * H + (ih + 2) * W + iw + 2];
                }
                Out[fn * Wo * Ho + ih * Wo + iw] = R;
            }
        }
    }
}


// Parallel Full Precision 3X3 Conv
void __attribute__((noinline)) Parallel_FP3X3Convolution(ArgConvTnew *Arg)
{
    FULL_PRECISION *__restrict__ In=Arg->In;
    int H=Arg->H;
    int W=Arg->W;
    int C=Arg->C;
    int Filternum=Arg->Filternum;
    FULL_PRECISION *__restrict__ Filter=Arg->Filter;
    FULL_PRECISION *__restrict__ Out=Arg->Out;
    int Wo=W-2;
    int Ho=H-2;

    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk;
    unsigned int First;
    unsigned int Last ;

    int Wo_F; int Wo_L;
    int Ho_F = 0; int Ho_L = Ho;
    int Fi_F,Fi_L;
    const int KW=3, KH=3;

    // If filter<8, parallel across width
    if (Filternum<8){
        Chunk = ChunkSize(Wo);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Wo);
        Wo_F = First;
        Wo_L = Last;
        Fi_F = 0;
        Fi_L = Filternum;
    }
    //else, parallel across filternum
    else{
        Chunk = ChunkSize(Filternum);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Filternum);
        Wo_F = 0;
        Wo_L = Wo;
        Fi_F = First;
        Fi_L = Last;
    }


    // direct conv
    for (int fn = Fi_F; fn < Fi_L; fn++) {
        for (int ih = Ho_F; ih < Ho_L; ih++) {
             for (int iw = Wo_F; iw < Wo_L; iw++) {
                FULL_PRECISION R = Out[fn * Wo * Ho + ih * Wo + iw]=0;
                // Use C, H, W format for data locality
                for (int kc = 0; kc < C; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // We use the C, H, W format
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 0]*In[kc * W * H + (ih + 0) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 1]*In[kc * W * H + (ih + 0) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 2]*In[kc * W * H + (ih + 0) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 0]*In[kc * W * H + (ih + 1) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 1]*In[kc * W * H + (ih + 1) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 2]*In[kc * W * H + (ih + 1) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 0]*In[kc * W * H + (ih + 2) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 1]*In[kc * W * H + (ih + 2) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 2]*In[kc * W * H + (ih + 2) * W + iw + 2];
                }
                Out[fn * Wo * Ho + ih * Wo + iw] = R;
            }
        }
    }
    //gap8_waitbarrier(0);
}


// Original XNOR Conv 3X3
void __attribute__((noinline)) XnorConv3X3(FULL_PRECISION *__restrict__ Input,  int H, int W, int C, int Filternum,
                                                unsigned int *__restrict__ Filter, FULL_PRECISION *__restrict__ Ffactor, FULL_PRECISION *__restrict__ Out)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the packed input matrix which is the sign(X)
    unsigned int PackedInput[H*W*C/32];
    // This is for the scaling factor matrix K
    FULL_PRECISION Inputsum[H*W];
    FULL_PRECISION Infactor[Ho*Wo];
    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XNOR)
    int xnornum=9 * PackedC * 32;


    // Pad the input, and get the average matrix.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad;
            FULL_PRECISION sum=0;
            for (int ic = 0; ic < PackedC; ic++){
                pad=0;
                for (int i= 0; i<32; i++)
                {
                    if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                        // pad=pad + ((32-ic%32)<<1);
                        pad=B_ins_r(pad, 0b1, 1, i);
                        sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                    }
                    else
                    {
                        sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                    }
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
            Inputsum[ih * W + iw]=sum/C;
        }
    }

    FULL_PRECISION sum;
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            sum=0;
            sum=sum+Inputsum[(ih + 0) * W + iw + 0];
            sum=sum+Inputsum[(ih + 0) * W + iw + 1];
            sum=sum+Inputsum[(ih + 0) * W + iw + 2];
            sum=sum+Inputsum[(ih + 1) * W + iw + 0];
            sum=sum+Inputsum[(ih + 1) * W + iw + 1];
            sum=sum+Inputsum[(ih + 1) * W + iw + 2];
            sum=sum+Inputsum[(ih + 2) * W + iw + 0];
            sum=sum+Inputsum[(ih + 2) * W + iw + 1];
            sum=sum+Inputsum[(ih + 2) * W + iw + 2];
            Infactor[ih * Wo + iw]=sum/KWH;
        }
    }

    // XNOR-Net Conv
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XNOR, '1' shands for +1
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]));
                }
                R = 2 * R - xnornum;
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Infactor[ih * Wo + iw];
            }
        }
    }
}



// Parallel Original XNOR Conv 3X3
void __attribute__((noinline)) Parallel_XnorConv3X3(ArgConvTxor *Arg)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */
    FULL_PRECISION *__restrict__ Input=Arg->Input;
    int H=Arg->H;
    int W=Arg->W;
    int C=Arg->C;
    int Filternum=Arg->Filternum;
    unsigned int *__restrict__ Filter=Arg->Filter;
    unsigned int *__restrict__ PackedInput=Arg->PackedInput;
    FULL_PRECISION *__restrict__ Ffactor=Arg->Ffactor;
    FULL_PRECISION *__restrict__ Out=Arg->Out;
    FULL_PRECISION *__restrict__ Kmatrix=Arg->Kmatrix;
    int layer=Arg->layer;

    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk;
    unsigned int First;
    unsigned int Last ;
    int W_F, W_L;
    int F_F, F_L;

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XOR)
    int xnornum=9 * PackedC * 32;

    Chunk = ChunkSize(W);
    First = Chunk*CoreId;
    Last = Min(First + Chunk, W);
    W_F = First;
    W_L = Last;

    // Pad the input, and get the average matrix.
    for (int iw = W_F; iw < W_L; iw++) {
        for (int ih = 0; ih < H; ih++) {
            int pad;
            FULL_PRECISION sum=0;
            for (int ic = 0; ic < PackedC; ic++){
                pad=0;
                for (int i= 0; i<32; i++)
                {
                    if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                        // pad=pad + ((32-ic%32)<<1);
                        pad=B_ins_r(pad, 0b1, 1, i);
                        sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                    }
                    else
                    {
                        sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                    }
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
            Input[(ih * W + iw)*C]=sum/C;
        }
    }

    Last = Min(First + Chunk, Wo);
    W_L = Last;

    rt_team_barrier();

    for (int iw = W_F; iw < W_L; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            FULL_PRECISION sum=0;
            sum=sum+Input[((ih + 0) * W + iw + 0)*C];
            sum=sum+Input[((ih + 0) * W + iw + 1)*C];
            sum=sum+Input[((ih + 0) * W + iw + 2)*C];
            sum=sum+Input[((ih + 1) * W + iw + 0)*C];
            sum=sum+Input[((ih + 1) * W + iw + 1)*C];
            sum=sum+Input[((ih + 1) * W + iw + 2)*C];
            sum=sum+Input[((ih + 2) * W + iw + 0)*C];
            sum=sum+Input[((ih + 2) * W + iw + 1)*C];
            sum=sum+Input[((ih + 2) * W + iw + 2)*C];
            Kmatrix[ih * Wo + iw]=sum/KWH;
        }
    }


    // If filter<8, parallel across width
    if (Filternum<8){
        Chunk = ChunkSize(Wo);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Wo);
        W_F = First;
        W_L = Last;
        F_F = 0;
        F_L = Filternum;
    }
    //else, parallel across filternum
    else{
        Chunk = ChunkSize(Filternum);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Filternum);
        W_F = 0;
        W_L = Wo;
        F_F = First;
        F_L = Last;
    }


    rt_team_barrier();
    // XNOR-Net Conv
    for (int fn = F_F; fn < F_L; fn++) {
        for (int ih = 0; ih < Ho; ih++) {
             for (int iw = W_F; iw < W_L; iw++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XNOR, '1' shands for +1
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]));
                }
                R = 2 * R - xnornum;
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Kmatrix[ih * Wo + iw];
            }
        }
    }

}

   


// Optimized XOR Conv 3X3 with scaling factors in the paper
void __attribute__((noinline)) XORSConv3X3(FULL_PRECISION *__restrict__ Input, int H, int W, int C, int Filternum, int layer,
                                             unsigned int *__restrict__ Filter, FULL_PRECISION *__restrict__ Ffactor, FULL_PRECISION *__restrict__ Out, FULL_PRECISION *__restrict__ Kmatrix)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * the layer type: layer: 0: only produce new K; 1: *K and produce new K; 2: *K and get final output;
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the packed input matrix which is the sign(X)
    unsigned int PackedInput[H * W * C / 32];
    // This is the scaling factor matrix K
    FULL_PRECISION Inputsum[H*W];
    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a popc(XOR)
    int xornum=9 * PackedC * 32;


    // Calculate the average across the channel for the input.
    // Pad the input, and get the average matrix.
    if (layer==0){
        for (int iw = 0; iw < W; iw++) {
            for (int ih = 0; ih < H; ih++) {
                unsigned int pad;
                FULL_PRECISION sum=0;
                for (int ic = 0; ic < PackedC; ic++){
                    pad=0;
                    for (int i= 0; i<32; i++)
                    {
                        if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                            // pad=pad + ((32-ic%32)<<1);
                            pad=B_ins_r(pad, 0b1, 1, i);
                            sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                        }
                        else
                        {
                            sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                        }
                    }
                    PackedInput[(ih * W + iw) * PackedC + ic]=pad;
                }
                Inputsum[ih * W + iw]=sum;
            }
        }
    }
    else{
        for (int iw = 0; iw < W; iw++) {
            for (int ih = 0; ih < H; ih++) {
                unsigned int pad;
                FULL_PRECISION sum=0;
                for (int ic = 0; ic < PackedC; ic++){
                    pad=0;
                    for (int i= 0; i<32; i++)
                    {
                        if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                            // pad=pad + ((32-ic%32)<<1);
                            pad=B_ins_r(pad, 0b1, 1, i);
                            sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                        }
                        else
                        {
                            sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                        }
                    }
                    PackedInput[(ih * W + iw) * PackedC + ic]=pad;
                }
                Inputsum[ih * W + iw]=sum*Kmatrix[ih * W + iw];
            }
        }
    }
    
    FULL_PRECISION sum;
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            sum=0;
            sum=sum+Inputsum[(ih + 0) * W + iw + 0];
            sum=sum+Inputsum[(ih + 0) * W + iw + 1];
            sum=sum+Inputsum[(ih + 0) * W + iw + 2];
            sum=sum+Inputsum[(ih + 1) * W + iw + 0];
            sum=sum+Inputsum[(ih + 1) * W + iw + 1];
            sum=sum+Inputsum[(ih + 1) * W + iw + 2];
            sum=sum+Inputsum[(ih + 2) * W + iw + 0];
            sum=sum+Inputsum[(ih + 2) * W + iw + 1];
            sum=sum+Inputsum[(ih + 2) * W + iw + 2];
            //sum=sum+Inputsum[(ih + kh) * W + iw + kw];
            Kmatrix[ih * Wo + iw]=sum;
        }
    }


    // XOR-Net-S Conv
    if (layer==2){
        for (int iw = 0; iw < Wo; iw++) {
            for (int ih = 0; ih < Ho; ih++) {
                for (int fn = 0; fn < Filternum; fn++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < PackedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the final layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Kmatrix[ih * Wo + iw];
                }
            }
        }
    }
    else{
        for (int iw = 0; iw < Wo; iw++) {
            for (int ih = 0; ih < Ho; ih++) {
                for (int fn = 0; fn < Filternum; fn++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < PackedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the first or a continuous layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn];
                }
            }
        }
    }
}



// Parallel Optimized XOR Conv 3X3 with scaling factors in the paper
void __attribute__((noinline)) Parallel_XORSConv3X3(ArgConvTxor *Arg)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * the layer type: layer: 0: only produce new K; 1: *K and produce new K; 2: *K and get final output;
     * */
    FULL_PRECISION *__restrict__ Input=Arg->Input;
    int H=Arg->H;
    int W=Arg->W;
    int C=Arg->C;
    int Filternum=Arg->Filternum;
    unsigned int *__restrict__ PackedInput=Arg->PackedInput;
    unsigned int *__restrict__ Filter=Arg->Filter;
    FULL_PRECISION *__restrict__ Ffactor=Arg->Ffactor;
    FULL_PRECISION *__restrict__ Out=Arg->Out;
    FULL_PRECISION *__restrict__ Kmatrix=Arg->Kmatrix;
    int layer=Arg->layer;

    // For parallel processing
    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk;
    unsigned int First;
    unsigned int Last ;
    int W_F, W_L;
    int F_F, F_L;

    // The kernel size
    int KW=3;
    int KH=3;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a popc(XOR)
    int xornum=9 * PackedC * 32;


    Chunk = ChunkSize(W);
    First = Chunk*CoreId;
    Last = Min(First + Chunk, W);
    W_F = First;
    W_L = Last;


    // Calculate the average across the channel for the input.
    // Pad the input, and get the average matrix.
    if (layer==0){
        for (int iw = W_F; iw < W_L; iw++) {
            for (int ih = 0; ih < H; ih++) {
                unsigned int pad;
                FULL_PRECISION sum=0;
                for (int ic = 0; ic < PackedC; ic++){
                    pad=0;
                    for (int i= 0; i<32; i++)
                    {
                        if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                            // pad=pad + ((32-ic%32)<<1);
                            pad=B_ins_r(pad, 0b1, 1, i);
                            sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                        }
                        else
                        {
                            sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                        }
                    }
                    PackedInput[(ih * W + iw) * PackedC + ic]=pad;
                }
                //Inputsum[ih * W + iw]=sum;
                Input[(ih * W + iw)*C]=sum;
            }
        }
    }
    else{
        for (int iw = W_F; iw < W_L; iw++) {
            for (int ih = 0; ih < H; ih++) {
                unsigned int pad;
                FULL_PRECISION sum=0;
                for (int ic = 0; ic < PackedC; ic++){
                    pad=0;
                    for (int i= 0; i<32; i++)
                    {
                        if (Input[(ih * W + iw)*C + ic*32+i] <0) {
                            // pad=pad + ((32-ic%32)<<1);
                            pad=B_ins_r(pad, 0b1, 1, i);
                            sum=sum-Input[(ih * W + iw)*C + ic*32+i];
                        }
                        else
                        {
                            sum=sum+Input[(ih * W + iw)*C + ic*32+i];
                        }
                    }
                    PackedInput[(ih * W + iw) * PackedC + ic]=pad;
                }
                //Inputsum[ih * W + iw]=sum*Kmatrix[ih * W + iw];
                Input[(ih * W + iw)*C]=sum*Kmatrix[ih * W + iw];
            }
        }
    }
    
    Last = Min(First + Chunk, Wo);
    W_L = Last;
    
    rt_team_barrier();
    
    FULL_PRECISION sum;
    for (int iw = W_F; iw < W_L; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            sum=0;
            sum=sum+Input[((ih + 0) * W + iw + 0)*C];
            sum=sum+Input[((ih + 0) * W + iw + 1)*C];
            sum=sum+Input[((ih + 0) * W + iw + 2)*C];
            sum=sum+Input[((ih + 1) * W + iw + 0)*C];
            sum=sum+Input[((ih + 1) * W + iw + 1)*C];
            sum=sum+Input[((ih + 1) * W + iw + 2)*C];
            sum=sum+Input[((ih + 2) * W + iw + 0)*C];
            sum=sum+Input[((ih + 2) * W + iw + 1)*C];
            sum=sum+Input[((ih + 2) * W + iw + 2)*C];
            Kmatrix[ih * Wo + iw]=sum;
        }
    }

    
    // If filter<8, parallel across width
    if (Filternum<8){
        Chunk = ChunkSize(Wo);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Wo);
        W_F = First;
        W_L = Last;
        F_F = 0;
        F_L = Filternum;
    }
    //else, parallel across filternum
    else{
        Chunk = ChunkSize(Filternum);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Filternum);
        W_F = 0;
        W_L = Wo;
        F_F = First;
        F_L = Last;
    }

    rt_team_barrier();
	
    // XOR-Net-S Conv
    if (layer==2){
        for (int fn = F_F; fn < F_L; fn++) {
            for (int ih = 0; ih < Ho; ih++) {
                for (int iw = W_F; iw < W_L; iw++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < PackedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the final layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Kmatrix[ih * Wo + iw];
                }
            }
        }
    }
    else{
        for (int fn = F_F; fn < F_L; fn++) {
            for (int ih = 0; ih < Ho; ih++) {
               for (int iw = W_F; iw < W_L; iw++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < PackedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
                        R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the first or a continuous layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn];
                }
            }
        }
    }
}




// Original XNOR Conv 3X3 without scaling factor
void __attribute__((noinline)) BCNNConv3X3(FULL_PRECISION *__restrict__ Input,  int H, int W, int C, int Filternum,
                                                unsigned int *__restrict__ Filter, FULL_PRECISION *__restrict__ Out)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the packed input matrix which is the sign(X)
    unsigned int PackedInput[H*W*C/32];
    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XNOR)
    int xnornum=9 * PackedC * 32;


    // Pad the input, and get the average matrix.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad=0;
            for (int ic = 0; ic < PackedC; ic++){
                // pad=0;
                for (int i= 0; i<32; i++)
                {
					pad=B_ins_r(pad, Input[(ih * W + iw)*C + ic*32+i], 1, i);
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
        }
    }


    // BCNN Conv
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XNOR, '1' shands for +1
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]));
                }
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = 2 * R - xnornum;
            }
        }
    }
}



// Parallel Original XNOR Conv 3X3 without scaling factor
void __attribute__((noinline)) Parallel_BCNNConv3X3(ArgConvTxor *Arg)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */
    FULL_PRECISION *__restrict__ Input=Arg->Input;
    int H=Arg->H;
    int W=Arg->W;
    int C=Arg->C;
    int Filternum=Arg->Filternum;
    unsigned int *__restrict__ Filter=Arg->Filter;
    unsigned int *__restrict__ PackedInput=Arg->PackedInput;
    FULL_PRECISION *__restrict__ Out=Arg->Out;


    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk;
    unsigned int First;
    unsigned int Last ;
    int W_F, W_L;
    int F_F, F_L;

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XOR)
    int xnornum=9 * PackedC * 32;

    Chunk = ChunkSize(W);
    First = Chunk*CoreId;
    Last = Min(First + Chunk, W);
    W_F = First;
    W_L = Last;

    // Pad the input
    for (int iw = W_F; iw < W_L; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad=0;
            for (int ic = 0; ic < PackedC; ic++){
                // pad=0;
                for (int i= 0; i<32; i++)
                {
                    pad=B_ins_r(pad, Input[(ih * W + iw)*C + ic*32+i], 1, i);
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
        }
    }



    // If filter<8, parallel across width
    if (Filternum<8){
        Chunk = ChunkSize(Wo);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Wo);
        W_F = First;
        W_L = Last;
        F_F = 0;
        F_L = Filternum;
    }
    //else, parallel across filternum
    else{
        Chunk = ChunkSize(Filternum);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Filternum);
        W_F = 0;
        W_L = Wo;
        F_F = First;
        F_L = Last;
    }


    rt_team_barrier();
	
    // BCNN Conv
    for (int fn = F_F; fn < F_L; fn++) {
        for (int ih = 0; ih < Ho; ih++) {
           for (int iw = W_F; iw < W_L; iw++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XNOR, '1' shands for +1
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc]^PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc]^PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]));
                    R +=B_popc(~(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc]^PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]));
                }
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = 2 * R - xnornum;
            }
        }
    }
}



// XOR Conv 3X3 without scaling factor in the paper
void __attribute__((noinline)) XORConv3X3(FULL_PRECISION *__restrict__ Input,  int H, int W, int C, int Filternum,
                                                unsigned int *__restrict__ Filter, FULL_PRECISION *__restrict__ Out)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the packed input matrix which is the sign(X)
    unsigned int PackedInput[H*W*C/32];
    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XNOR)
    int xornum=9 * PackedC * 32;


    // Pad the input, and get the average matrix.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad=0;
            for (int ic = 0; ic < PackedC; ic++){
                // pad=0;
                for (int i= 0; i<32; i++)
                {
					pad=B_ins_r(pad, Input[(ih * W + iw)*C + ic*32+i], 1, i);
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
        }
    }


    // XOR-Net Conv
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XOR, '1' stands for -1
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
			   }
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = xornum - 2 * R;
            }
        }
    }
}



// Parallel XOR Conv 3X3 without scaling factor in the paper
void __attribute__((noinline)) Parallel_XORConv3X3(ArgConvTxor *Arg)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter packed: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */
    FULL_PRECISION *__restrict__ Input=Arg->Input;
    int H=Arg->H;
    int W=Arg->W;
    int C=Arg->C;
    int Filternum=Arg->Filternum;
    unsigned int *__restrict__ Filter=Arg->Filter;
    unsigned int *__restrict__ PackedInput=Arg->PackedInput;
    FULL_PRECISION *__restrict__ Out=Arg->Out;

    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk;
    unsigned int First;
    unsigned int Last ;
    int W_F, W_L;
    int F_F, F_L;

    // The kernel size
    int KW=3;
    int KH=3;
    FULL_PRECISION KWH=KW*KH;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // the packed channel num
    int PackedC=C/32;
    // the number of bits that do a pop_count(XOR)
    int xornum=9 * PackedC * 32;

    Chunk = ChunkSize(W);
    First = Chunk*CoreId;
    Last = Min(First + Chunk, W);
    W_F = First;
    W_L = Last;

    // Pad the input
    for (int iw = W_F; iw < W_L; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad=0;
            for (int ic = 0; ic < PackedC; ic++){
                // pad=0;
                for (int i= 0; i<32; i++)
                {
                    pad=B_ins_r(pad, Input[(ih * W + iw)*C + ic*32+i], 1, i);
                }
                PackedInput[(ih * W + iw)*PackedC+ic]=pad;
            }
        }
    }



    // If filter<8, parallel across width
    if (Filternum<8){
        Chunk = ChunkSize(Wo);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Wo);
        W_F = First;
        W_L = Last;
        F_F = 0;
        F_L = Filternum;
    }
    //else, parallel across filternum
    else{
        Chunk = ChunkSize(Filternum);
        First = Chunk*CoreId;
        Last = Min(First + Chunk, Filternum);
        W_F = 0;
        W_L = Wo;
        F_F = First;
        F_L = Last;
    }


    rt_team_barrier();
	
    // XOR-Net Conv
    for (int fn = F_F; fn < F_L; fn++) {
        for (int ih = 0; ih < Ho; ih++) {
             for (int iw = W_F; iw < W_L; iw++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < PackedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XOR, '1' stands for -1
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (0 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 0) * W + iw + 2 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (1 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 1) * W + iw + 2 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 0) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 0 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 1) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 1 ) * PackedC + kc]);
					R +=B_popc(Filter[fn * 9 * PackedC+ (2 * KW + 2) * PackedC + kc] ^ PackedInput[((ih + 2) * W + iw + 2 ) * PackedC + kc]);
			   }
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] =xornum - 2 * R ;
            }
        }
    }
}




/***************************************************************************************
 * The benchmarking function calls all methods above
 * *************************************************************************************
 */

int RunTests(ClusterArg_t * ArgC )
{
    int Which = ArgC->test_num;
    int Iter = ArgC->Iter;
    int * num_ops = &(ArgC->Iter_operations);
    int H=ArgC->H;
    int W=ArgC->W;
    int C=ArgC->C;
    int Filternum=ArgC->Filternum;

    // Factory benchmark types
    unsigned int Ti;
    ArgConvTnew FArg;
    ArgConvTxor XArg;

    /***************************************
     * The types used in my benchmarks
     * All float for full precision numbers
     * unsigned int for XNOR operations
     * *************************************
     */
    FULL_PRECISION *In, *Out, *Filter, *Ffactor, *K;
    unsigned int *XNORFilter, *PackedInput;
    int KW=3,KH=3;


    switch (Which) {
        // Full Precision 3X3 Conv
        case 0:
            In = Mem; Out = Mem+H*W*C; Filter=Mem+H*W*C+(H-2)*(W-2)*Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KW*KH*C*Filternum);
            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) FP3X3Convolution(In, H, W, C, Filternum, Filter, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;

        // Parallel Precision 3X3 Conv
        case 1:
            In = Mem; Out = Mem+H*W*C; Filter=Mem+H*W*C+(H-2)*(W-2)*Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KW*KH*C*Filternum);
            FArg.In=In; FArg.H=H; FArg.W=W; FArg.C=C; FArg.Filternum=Filternum; FArg.Filter=Filter;FArg.Out=Out;

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) rt_team_fork(gap8_ncore(), (void *)Parallel_FP3X3Convolution, (void *)&FArg);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;

        // Original XNOR-Net 3X3 Conv
        case 2:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum);

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) XnorConv3X3(In, H, W, C, Filternum, XNORFilter, Ffactor, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;

        // Parallel Original XNOR-Net 3X3 Conv
        case 3:
            In = Mem; Out = Mem+H*W*C;
            PackedInput = (unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;
            XNORFilter = (unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum+H*W*C/32;
            Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+H*W*C/32;
            K=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+H*W*C/32+Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W*C/32+H*W);

            XArg.Input=In; XArg.H=H; XArg.W=W; XArg.C=C; XArg.PackedInput=PackedInput;
            XArg.Filternum=Filternum; XArg.Filter=XNORFilter;XArg.Ffactor=Ffactor;
            XArg.Out=Out; XArg.layer=0; XArg.Kmatrix=K;
            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) {
                rt_team_fork(gap8_ncore(), (void *)Parallel_XnorConv3X3, (void *)&XArg);
            }
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;


        // Optimized XOR-Net 3X3 Conv with scaling factors in the paper, layer=1
        case 4:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            K=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W);

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) XORSConv3X3(In, H, W, C, Filternum, 1, XNORFilter, Ffactor, Out, K);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;


        // Parallel optimized XOR-Net 3X3 Conv with scaling factors in the paper, layer=1
        case 5:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter= (unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;
            Ffactor =   Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            K =         Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum;
            PackedInput=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W+H*W*C/32);

            XArg.Input=In; XArg.H=H; XArg.W=W; XArg.C=C; XArg.PackedInput=PackedInput;
            XArg.Filternum=Filternum; XArg.Filter=XNORFilter;XArg.Ffactor=Ffactor;
            XArg.Out=Out; XArg.Kmatrix=K; XArg.layer=1;

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) {
                rt_team_fork(gap8_ncore(), (void *)Parallel_XORSConv3X3, (void *)&XArg); 
            }
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;

	// CI-BCNN Conv
        case 6:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            K=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W);

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) BCNNConv3X3(In, H, W, C, Filternum, XNORFilter, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;


        // Parallel CI-BCNN Conv
        case 7:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter= (unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;
            Ffactor =   NULL;
            K =         NULL;
            PackedInput=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+H*W*C/32);

            XArg.Input=In; XArg.H=H; XArg.W=W; XArg.C=C; XArg.PackedInput=PackedInput;
            XArg.Filternum=Filternum; XArg.Filter=XNORFilter;XArg.Ffactor=Ffactor;
            XArg.Out=Out; XArg.Kmatrix=K; XArg.layer=1;

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) {
                rt_team_fork(gap8_ncore(), (void *)Parallel_BCNNConv3X3, (void *)&XArg); 
            }
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;

			
	// XOR-Net Conv in the paper
        case 8:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            K=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+Filternum+H*W);

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) XORConv3X3(In, H, W, C, Filternum, XNORFilter, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;


        // Parallel XOR-Net Conv in the paper
        case 9:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter= (unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum;
            Ffactor =   NULL;
            K =         NULL;
            PackedInput=(unsigned int *)Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum+H*W*C/32);

            XArg.Input=In; XArg.H=H; XArg.W=W; XArg.C=C; XArg.PackedInput=PackedInput;
            XArg.Filternum=Filternum; XArg.Filter=XNORFilter;XArg.Ffactor=Ffactor;
            XArg.Out=Out; XArg.Kmatrix=K; XArg.layer=1;

            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) {
                rt_team_fork(gap8_ncore(), (void *)Parallel_XORConv3X3, (void *)&XArg); 
            }
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            break;
			
    }
    return 1;
}


int main()
{
	long start_time, end_time;
	long int tot_time, op_num, kernel_op_num;
	float res, res_kernel;
	int cur_test = 0;
    int tcycle=15; // The total number of layers as shows below
    int H[]={ 7, 7,28,28,14,14,14,14,14,14, 14,14, 14, 14, 14};
    int W[]={ 7, 7,28,28,14,14,14,14,14,14, 14,14, 14, 14, 14};
    int C[]={32,64,32,64,32,64,64,64,64,64, 64,96,128,160,192};
    int F[]={32,32,32,32,32, 8,16,32,64,96,128,32, 32, 32, 32};


#if !ALIM_1_VOLT
	PMU_set_voltage(1150, 0);
	PMU_set_voltage(1200, 0);
#else
	PMU_set_voltage(1000, 0);
#endif

#ifndef NOGPIO
	rt_padframe_profile_t *profile_gpio = rt_pad_profile_get("hyper_gpio");

	if (profile_gpio == NULL) {
		printf("pad config error\n");
		return 1;
	}
	rt_padframe_set(profile_gpio);
	// GPIO initialization
	rt_gpio_init(0, GPIO);
	rt_gpio_set_dir(0, 1 << GPIO, RT_GPIO_IS_OUT);

#endif

	printf("\n\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      ---------------   GAP8 benchmarks   --------------------\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      --------------------------------------------------------\n\n\n");


	printf("Gap8 Input Voltage    : %s\n", ALIM_1_VOLT ? "1.0 Volt" : "1.2 Volts");
	printf("Fabric Controller Freq: %d MhZ\n", FREQ_FC / 1000000);
	printf("Cluster  Freq         : %d MhZ\n\n\n", FREQ_CL / 1000000);

	printf("Number of iterations for each benchmark: %d\n\n\n", ITERATIONS);


	if (rt_event_alloc(NULL, 8)) return -1;

	rt_cluster_mount(MOUNT, CID, 0, NULL);

	//Set Fabric Controller and Cluster Frequencies
	rt_freq_set(RT_FREQ_DOMAIN_FC, FREQ_FC);
	rt_freq_set(RT_FREQ_DOMAIN_CL, FREQ_CL);


    // Iterate on the configuration of layers, e.g. CHW={32, 14, 14}
    for (int cycle=4; cycle<tcycle; cycle++)
    {
	    cur_test=0;
		int numk;
		numk=0;
        // Iterate on the type of networks, e.g., Full-Precision, XNOR-Net, XOR-Net
        for (int j = 0;j < TOT_TEST; j++) {
            printf("\n                      ---------------   %15s   ---------------\n", tests_titles[j]);
            // Iterate on the sequential and parallel version of the convolution functions
            for (int i = 0; i < test_num[j]; i++) {

                Arg.test_num = cur_test++;
                Arg.Iter = ITERATIONS; 
                Arg.H=H[cycle];
                Arg.W=W[cycle];
                Arg.C=C[cycle];
                Arg.Filternum=F[cycle];

 
                start_time = rt_time_get_us(); 
                rt_cluster_call(NULL, CID, (void *)RunTests, &Arg, NULL, 0, 0, 8, NULL);
                end_time = rt_time_get_us();

                tot_time = end_time - start_time;
                op_num = Arg.Iter_operations;

                printf("%30s Input: %d x %d x %d, 3x3 Filter: %d (x%d iterations) Time: %10ld uSec. Cycles: %10ld\n", tests_names[numk], H[cycle], W[cycle], C[cycle], F[cycle],  ITERATIONS, tot_time, op_num);
				numk++;
            }
        }
    }

	rt_cluster_mount(UNMOUNT, CID, 0, NULL);

	return 0;
}
