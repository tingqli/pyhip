mem type	latency(cycles)	throughput/CU (cycles)	throughput/CU (bytes/cycles)	issue-cycle @4-waves/CU (cycles/instruction)
buffer load dwordx4	500~800	~32	32	32x4 = 128
buffer load dword lds	500~800	~11	23	11x4 = 44
LDS read (no bank-conflict) b128/b32	64/52	16/4 [^1]	64/64 [^1]	8x4 = 32
LDS read b32 (4-bank-conflict)	120	16	16	
LDS read b32 (full-bank-conflict)	119	64/64	4	


mem type	latency(cycles)	throughput/CU (cycles)	throughput/CU (bytes/cycles)	issue-cycle @4-waves/CU (cycles/instruction)
buffer load dwordx4	500~800	~32	32	32x4 = 128
buffer load dword lds	500~800	~11	23	11x4 = 44
LDS read (no bank-conflict) b128/b32	64/52	16/4 [^1]	64/64 [^1]	8x4 = 32
LDS read b32 (4-bank-conflict)	120	16	16	
LDS read b32 (full-bank-conflict)	119	64/64	4	

##REGS:
1. HBM data need to put into regs first , 32KB divided by 4 waves, Each  wave8KB,  each lane 8KB/64 =128 VGPRs for HBM,
   Total:128BGPRs.
2. REGs for [128, 8]x[128x8]= [128, 128],   A,B KL=4,  A: 128/32xKL/2=8REGA,  B: 128/32xKL/2= 8REGB, ,  C: 4x4x4 = 64 
   Total: 16VGPRs + 64 ACC when innter most K is 8.

##LDS:
1. 32KB for [256,32]x[256,32]
2. Can add ping-pong. Split the prefetch and MFMA needed LDS.


##Latency in 1st level:
  

Total  LDS per CU =   [256,32]x[256,32] = 32KB

FMA latency per Wave:          [128,32] x [128, 32] = [128, 128]               MFMA_f32_32x32x8_f16,        per ISA cyles:32?                              Total : 32x 4x4x4=2,048 , 2048 cycles for FMA , 
Load HBM latency per CU:         [256,32]x[256,32]=[256,256]                  A+B=32KB                      HBM tput per CU, 32B                   HBM cycles:            (256x32x2)x2/32 = 1,024 cycles.


##PIPE LANE in 1st level:
Assume LDS load to REGs can be hidden by FMA, let us see how load HBM, store into LDS and FMA pipelined.


            FMA:                                               ||                     2048                   ||                     2048                   ||                     2048                   ||

       HBM load:               ||1024+800, load  |  512, store || 1024+800, load  |  512, store  |           ||      1024+800, load  |  512, store  |      ||      1024+800, load  |  512, store  |      ||

FMA and load, store. So  wait vmcnt(0) should be about at  1024+800 , wait lgkmcnt(0) should be at end of each iteration.           

##Latency in 2nd level: FMA hide LDS loading. Ping-pong registers for A, B

Each WARP calculate: [128,8]x[128,8], FMA latency= 32x4x4=512 cycles;
Loading A+B:   assume LBS is 128B/cycle for 4 SIMD16, Each SIMD16 is 32B/cycles.   256x8x2=4KB,     4KB/32=128 cycles.

##PIPE LANE in 1st level:

          FMA:                                              |            512             |            512             |            512             |            512             ||
     LDA load:                                     ||LDS 128|LDS 128|                    |LDS 128|                    |LDS 128|                    |
                   ||1024+800, load  |  512, store ||            1024+800, load                                                            |  512, store                         ||


QA: load from LDS and store prefetch data into LDS interleaved?  



## ILP learning:
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ILP doesn't use either ping-pong LDS or ping-pong registers for A,B or pingpong prefetch reigsters. Just manually optimize the pipeline.

CU calculate [256, 256, 32] matrix, using MFA_f32_32x32x8_f16 ISAs. Buffer load using 4xDWORD, LDS load/store also using 4xDWORDS. 8 elemnts each time.

### ISA number:
FMA per wave:   **MFMA_F32_32x32x8_f16** , For each wavefront, [128,128,32], needs MFMA ISAs number 128x128x32/( 32x32x8) = 64 MFA ISAs.

**buffer_load_dwordx4 & ds_write_128b**  per lane:  256x32x2/4 waves/64 lanes/ 8element = 8 ISAs, 

**ds_read_128b** per lane: 128x32x2/64lanes/8 elements = 16 ISAs. 

buffer_load_dwordx4/ds_write_128 is 8 ISAs per lane, but ds_read_128b per lane is 16 ISAs(NOT 8 ISAs) per lane. 

Because the 4 waves in one CU, 2x2 tiles.For the  buffer_load_dwordx4/ds_write_128 no duplicate data is access across 4 waves,divided evenly across 4 waves.

But for the ds_read_128b, duplicate read across the waves because of data reuse. 4 waves would read 2 times data.


### LDS occupancy:
LDS size for [256, 256, 32] = 256x32x2x2= 32KB. Total 64 KB, only use 50%.  

## VGPR&ACC number for FMA :
FOR 32x32x8, KL=4. So for FP16 input, REG_KL=KL/sizeof(FP16) = 2. As for 128x128x32 case:

**REGM=4, REGN=4, REGK=4, REG_KL=2**. SO MFMA need regs=REGMxREGKxREG_KL+REGNxREGKxREGKL=64 regs.

For ACC result needs regs 128x128/64 = 256 registers. So all the ACC registers would be used to hold result.

**So FMA needs 256 ACC registers and 64 VGPRs.**

For for the MFMA 128x128x32, A would be divided into REGM*REGK tiles. **A[0:3, 0:3]**

B would be divided into REGNxREGK tiles. **B[0:3, 0:3]**

C would be divied into REGMxREGN tiels. **C[0:3, 0:3]**

### VGPR number for prefetch:
To prefech all 256x256x32 per CU, the CU needs registers  (256x2x32)/2=8K. So each lane would need 8K/4/64=32 VGPRs



### ILP Prologue:
```
8 buffer0_load_dwordx8_[0:8]           ->    trigger buffer_0 , buffer_0 is all A+B for CU to calculate 256x256x32
wait_cnt(vcnt(0))                      ->    buffer_0 prefetch done
8  ds_write_128b_[0:8]                 ->    trigger buffer_0  into LDS
8 buffer1_load_dwordx8[0:8]            ->    tirgger buffer_1 prefetch.
wait_cnt(lgkmcnt(0))                   ->    ready: buffer_0 into LDS  8 ds_write_128b  done.
sbarrier()                                       
ds_read_128b_[0:7]                     ->    trigger read the first half of LDS.   A[0:3,0:1] + B[0:3,0:1]
```


### ILP hot loop:
```
///////////////////hot loop start:
///////////////////status: N-th buffer prefetch done, buffer_N in LDS;  Trigger first half of buffer_N  (A[0:3,0:1] + B[0:3,0:1]) into LDS ; buffer_N+1  prefetch triggered.

wait_cnt(lgkmcnt(3))             ->    ready: A[0:3,0:1]  ready B[0,0:1].



/////part 0:16 FMAs: to accumulate  K0-K7, 4 REGM, 4REGN,  1st REGK
/////interleaved: LDS load A[0:3,2:3], B[0:3,2:3] done
/////interleaved: buffer1_0 done, buffer2_0 triggered.

MFMA A[0,0], B[0,0] , C[0,0]
ds_read_128b_8                    ->    trigger A[0,2:3] read
MFMA A[1,0]  B[0,0],  C[1,0]
ds_read_128b_9                    ->    trigger B[0,2:3] read
MFMA A[2,0], B[0,0],  C[2,0]
ds_read_128b_10                   ->    trigger A[1,2:3] read
MFMA A[3,0], B[0,0],  C[3,0],
ds_read_128b_11                   ->    trigger A[2,2:3] read
wait_cnt(lgkmcnt(4))              ->    ready: read A[0:3,0:1] and B[0:3,0:1]


MFMA A[0,0], B[1,0] , C[0,1]
ds_read_128b_12                    ->    trigger A[3,2:3] read
MFMA A[1,0], B[1,0] , C[1,1]
ds_read_128b_13                    ->    trigger B[1,2:3] read
MFMA A[2,0], B[1,0] , C[2,1]
ds_read_128b_14                    ->    trigger B[2,2:3] read
MFMA A[3,0], B[1,0] , C[3,1]
ds_read_128b_15                    ->    trigger B[3,2:3] read


MFMA A[0,0], B[2,0] , C[0,2]
MFMA A[1,0], B[2,0] , C[1,2]
MFMA A[2,0], B[2,0] , C[2,2]
MFMA A[3,0], B[2,0] , C[3,2]
MFMA A[0,0], B[3,0] , C[0,3]
MFMA A[1,0], B[3,0] , C[1,3]
wait_cnt(lgkmcnt(0))             -> ready: A[0:3,2:3], B[0:3,2:3] read done. All the A,B data of buffer 0 is in registers. So LDS data can be written..
sbarrier()                       -> will start writing buffer 1 into LDS. all the thread needs to done.


MFMA A[2,0], B[3,0] , C[2,3]
wait_cnt(vcnt(7))              ->   ready: buffer1_0 prefetched ok.
ds_write_128b[0]               ->   trigger: buffer1_0 into LDS, the prefetch reg0 can be used immediately

MFMA A[3,0], B[3,0] , C[3,3]
buffer2_load_dwordx8[0]        ->   prefetch buffer2_0， ds_write_128b[0]  done? prefetch_reg[0] available? no need to wait_cnt(lgkmcnt(0)) ??



//////part 1: 16 FMAs to accumalater K8-K15, 4 REGM, 4REGN,  2nd REGK,
/////interleaved: buffer1_[1:3] done, buffer2_[1:3] triggered.

wait_cnt(lgkmcnt(9))          -> bug??????????????????????????????????????????, no use. lgkmcnt should be 0 now.

MFMA A[0,1], B[0,1] , C[0,0]
MFMA A[1,1]  B[0,1],  C[1,0]
MFMA A[2,1], B[0,1],  C[2,0]
MFMA A[3,1], B[0,1],  C[3,0],
wait_cnt(vcnt(7))               -> buffer1_1 prefetched ok.
ds_write_128b[1]                -> buffer1_1 into LDS,
MFMA A[0,1], B[1,1] , C[0,1]
buffer2_load_dwordx8[1]        ->  prefetch buffer2_1，


MFMA A[1,1], B[1,1] , C[1,1]
MFMA A[2,1], B[1,1] , C[2,1]
MFMA A[3,1], B[1,1] , C[3,1]
MFMA A[0,1], B[2,1] , C[0,2]
wait_cnt(vcnt(7))              -> buffer1_2 prefetched ok.
ds_write_128b[2]               ->buffer1_2 into LDS,
MFMA A[1,1], B[2,1] , C[1,2]
buffer2_load_dwordx8[2]        -> prefetch buffer2_2，



MFMA A[2,1], B[2,1] , C[2,2]
MFMA A[3,1], B[2,1] , C[3,2]
MFMA A[0,1], B[3,1] , C[0,3]
MFMA A[1,1], B[3,1] , C[1,3]

wait_cnt(vcnt(7))              -> buffer1_3 prefetched ok.
ds_write_128b[3]               ->buffer1_3 into LDS,
MFMA A[2, 1], B[3, 1] , C[2,3]
buffer2_load_dwordx8[3]        -> prefetch buffer2_3，


MFMA A[3, 1], B[3, 1] , C[3,3]



////////////////////////16 FMAs to accumalater K16-K23, 4 REGM, 4REGN,  3rd REGK,
/////interleaved: buffer1_[4:6] done, buffer2_[4:6] triggered.

wait_cnt(lgkmcnt(4))          -> bug??????????????????????????????????????????, no use.

MFMA A[0,2], B[0,2] , C[0,0]
MFMA A[1,2]  B[0,2],  C[1,0]
MFMA A[2,2], B[0,2],  C[2,0]
MFMA A[3,2], B[0,2],  C[3,0]
wait_cnt(vcnt(7))              -> buffer1_4 prefetched ok.
ds_write_128b[4]               ->buffer1_4 into LDS,
MFMA A[0,2], B[1,2] , C[0,1]
buffer2_load_dwordx8[4]        -> prefetch buffer2_4，


MFMA A[1,2], B[1,2] , C[1,1]
MFMA A[2,2], B[1,2] , C[2,1]
MFMA A[3,2], B[1,2] , C[3,1]
MFMA A[0,2], B[2,2] , C[0,2]
wait_cnt(vcnt(7))              -> buffer1_5 prefetched ok.
ds_write_128b[5]               ->buffer1_5 into LDS,
MFMA A[1,2], B[2,2] , C[1,2]
buffer2_load_dwordx8[5]        -> prefetch buffer2_5，


MFMA A[2,2], B[2,2] , C[2,2]
MFMA A[3,2], B[2,2] , C[3,2]
MFMA A[0,2], B[3,2] , C[0,3]
MFMA A[1,2], B[3,2] , C[1,3]
wait_cnt(vcnt(7))              -> buffer1_6 prefetched ok.
ds_write_128b[6]               ->buffer1_6 into LDS,
MFMA A[2,2], B[3,2] , C[2,3]
buffer2_load_dwordx8[5]        -> prefetch buffer2_6，


MFMA A[3,2], B[3,2] , C[3,3]




////////////////////////16 FMAs to accumalater K24-K31
/////interleaved: buffer1_[7] done, buffer2_[7] triggered.
/////interleaved: trigger first half of LDS  to (A[0:3,0:1] + B[0:3,0:1])

wait_cnt(lgkmcnt(7))          -> bug??????????????????????????????????????????, no use.

MFMA A[0,3], B[0,3] , C[0,0]
MFMA A[1,3]  B[0,3],  C[1,0]
MFMA A[2,3], B[0,3],  C[2,0]
wait_cnt(vcnt(7))                 -> buffer1_7 prefetched ok.
ds_write_128b[3]                  ->buffer1_7 into LDS,
MFMA A[3,3], B[0,3],  C[3,0],
buffer2_load_dwordx8[3]           -> prefetch buffer2_7，



MFMA A[0,3], B[1,3] , C[0,1]
wait_cnt(lgkmcnt(7))              -> buffer1 all into LDS..., buffer2 all trigger prefetched...
sbarrier                          -> will start read the buffer1 from LDS to  A[0:3,0:1] + B[0:3,0:1]


MFMA A[1,3], B[1,3] , C[1,1]
ds_read_128b_0                   -> triger LDA to A[0, 0:1]

MFMA A[2,3], B[1,3] , C[2,1]
ds_read_128b_1                   -> triger LDA to B[0, 0:1]
MFMA A[3,3], B[1,3] , C[3,1]
ds_read_128b_2                   -> triger LDA to A[1, 0:1]
MFMA A[0,3], B[2,3] , C[0,2]
ds_read_128b_3                   -> triger LDA to A[2, 0:1]

MFMA A[1,3], B[2,3] , C[1,2]
ds_read_128b_4                   -> triger LDA to A[3, 0:1]



MFMA A[2,3], B[2,3] , C[2,2]
ds_read_128b_5                   -> triger LDA to B[1, 0:1]
MFMA A[3,3], B[2,3] , C[3,2]
ds_read_128b_6                   -> triger LDA to B[2, 0:1]
MFMA A[0,3], B[3,3] , C[0,3]
ds_read_128b_7                   -> triger LDA to B[3, 0:1]

MFMA A[1,3], B[3,3] , C[1,3]
MFMA A[2,3], B[3,3] , C[2,3]
MFMA A[3,3], B[3,3] , C[3,3]

updated buffer offset
///////////////////hot loop start:
###########status: Nth buffer matmul done.
                   (N+1)th buffer prefetch done, (N+1)th buffer in LDS;
                   half of (N+1)th in LDS  into A[0:3,0:1] + B[0:3,0:1] triggered.  (N+2)th buffer prefetch triggered..
```

### ILP tail: