# PSC_HPC_Ex
PSC HPC Acceleration training repository. Copyright PSC and XSEDE. Instructor: John Urbanic.

# OpenACC
Check OpenACC specifications: `http://www.openacc-standard.org`

## A. Introduction

### 1. Kernel
Kernel is a trunk of code that run in GPU.

In C:
```c
#pragma acc kernels [clause ...]
{
structured block
}
...
```

In Fortran:
```fortran
!$acc kernels [clause ...]
  structured block
!$acc end kernels
```

#### Example: SAXPY.py
##### 1. Example code

```c
void saxpy(int n,
float a,
float *x,
float *restrict y)
{
#pragma acc kernels
for (int i = 0; i < n; ++i)
y[i] = a*x[i] + y[i];
}
...
// Somewhere in main
// call SAXPY on 1M elements
saxpy(1<<20, 2.0, x, y);
...
```

##### The **restrick** keyword in C or C++ code: not being aliased by anything else.

By default, Fortran doesn't have pointers (except modern Fortran). However, it's possible two points can be directed to the same array (which is often the case in C. This is the reason why Fortran code is usually a little bit faster than C.). By adding `restrict` keyword, the compiler will know explicitly not guessing from other name, limits the effects of ointer aliasing.

##### 2. Compile and Run
  **Compile**
* C: `nvc -acc -Minfo=accel saxpy.c`
* Fortran: `nvfortran -acc -Minfo=accel saxpy.f90`
  **Run**
* run: `a.out`

`-Minfo`: to enable the compiler feedback when you are writing your own OpenACC programs.  

### 2. Data Dependency
Given a situation that the Processor 1 needs the result of Process 0's last iteration, e.g.,
```fortran
for(index=0; index<1000000; index++)
Array[index] = 4 * Array[index] – Array[index-1];
```
That is a data dependency. If the compiler even suspects that there is a data dependency, it will, for the sake of correctness, refuse to parallelize that loop
with kernels.
```
11, Loop carried dependence of 'Array' prevents parallelization
Loop carried backward dependence of 'Array' prevents vectorization
```
Basically, you have three options to do with data dependencies:
* Rearrange your code to make it more obvious to the compiler that there
is not really a data dependency.
* Eliminate a real dependency by changing your code.
  1) There is a common bag of tricks developed for this as this issue goes
back 40 years in HPC. Many are quite trivial to apply.
  2) The compilers have gradually been learning these themselves.
* Override the compiler’s judgment (independent clause) at the risk of
invalid results. Misuse of restrict has similar consequences.

#### Example: Laplce Solver
𝛁<sup>𝟐</sup> 𝒇(𝒙, 𝒚) =do j=1,columns
do i=1,rows
temperature(i,j)= 0.25 * (temperature_last(i+1,j)+temperature_last(i-1,j) + &
temperature_last(i,j+1)+temperature_last(i,j-1) )
enddo
enddo 0

Solve it using Jacobi Iteration
𝐴<sup>𝑘+1</sup>(,𝑗) =1/4 * (𝐴<sup>𝑘</sup>(𝑖-1,𝑗) + 𝐴<sup>𝑘</sup>(𝑖+1,𝑗) + 𝐴<sup>𝑘</sup>(𝑖,𝑗−1) + 𝐴<sup>𝑘</sup>(𝑖,𝑗+1))

Serial code implemention:
C:
```c
for(i = 1; i <= ROWS; i++) {
  for(j = 1; j <= COLUMNS; j++) {
    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                Temperature_last[i][j+1] + Temperature_last[i][j-1]);
  }
}
```
F:
```fortran
do j=1,columns
  do i=1,rows
    temperature(i,j)= 0.25 * (temperature_last(i+1,j)+temperature_last(i-1,j) + &
                              temperature_last(i,j+1)+temperature_last(i,j-1) )
  enddo
enddo
```
Test it:
  To compile: 
  ```nvfortran -acc laplace_serial.f90```
  ```nvc -acc laplace_serial.c```
  To run:
  './a.out'

Note: Adding `-Minfo=accel` to your compile command will give you some very useful information about
how well the compiler was able to honor your OpenACC directives.

#### Exercise 1: : Using kernels to parallelize the main loops (on bridge2.psc.edu)
(About 20 minutes)
1. Edit laplace_serial.c/f90
  1). Maybe copy your intended OpenACC version to laplace_acc.c to start
  2). Add directives where it helps
2. Compile with OpenACC parallelization
  1). `nvc -acc –Minfo=accel laplace_acc.c` or
      `nvfortran -acc –Minfo=accel laplace_acc.f90`
  2). Look at your compiler output to make sure you are having an effect
3. Run
  1). `sbatch gpu.job` (Leave it at 4000 iterations if you want a solution that converges to current tolerance)
  2). Look at output in file that returns (something like `slurm-138555.out`)
  3). Compare the serial and your OpenACC version for performance difference

#### Solution 
C: Add `#pragma acc kernels` before for loop
``` c
#pragma acc kernels
for(i = 1; i <= ROWS; i++) {
  for(j = 1; j <= COLUMNS; j++) {
    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                Temperature_last[i][j+1] + Temperature_last[i][j-1]);
  }
}
dt = 0.0; // reset largest temperature change
#pragma acc kernels
for(i = 1; i <= ROWS; i++){
  for(j = 1; j <= COLUMNS; j++){
    dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
    Temperature_last[i][j] = Temperature[i][j];
  }
}
if((iteration % 100) == 0) {
  track_progress(iteration);
}
iteration++;
}
```

##### Examine the compiler's output:
```sh
main:
     62, Loop is parallelizable
         Generating implicit copyin(Temperature_last[:][:]) [if not already present]
         Generating implicit copyout(Temperature[1:1000][1:1000]) [if not already present]
     63, Loop is parallelizable
         Generating Tesla code
         62, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x */
         63,   /* blockIdx.x threadIdx.x auto-collapsed */
     72, Loop is parallelizable
         Generating implicit copyin(Temperature[1:1000][1:1000]) [if not already present]
         Generating implicit copy(dt,Temperature_last[1:1000][1:1000]) [if not already present]
     73, Loop is parallelizable
         Generating Tesla code
         72, #pragma acc loop gang, vector(128) collapse(2) /* blockIdx.x threadIdx.x */
             Generating implicit reduction(max:dt)
         73,   /* blockIdx.x threadIdx.x auto-collapsed */
```
Run time: OpenACC 32.77 seconds  > 5.72 seconds (CPU serial) ==> disapointing!

Reduction:
```c
dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
```
(check it back later)

### 3. Memory management
In the previous example, if we check the output file (enabling `export PGI_ACC_TIME=1`), wee
```sh
Accelerator Kernel Timing data
/jet/home/ptz/Exercises/OpenACC/laplace_acc.c
  main  NVIDIA  devicenum=0
    time(us): 10,852,003
    62: compute region reached 3372 times
        63: kernel launched 3372 times
            grid: [7813]  block: [128]
            elapsed time(us): total=145,530 max=74 min=41 avg=43
    62: data region reached 6744 times
        62: data copyin transfers: 3372
             device time(us): total=2,217,267 max=683 min=656 avg=657
        67: data copyout transfers: 3372
             device time(us): total=2,079,843 max=626 min=615 avg=616
    72: compute region reached 3372 times
        73: kernel launched 3372 times
            grid: [7813]  block: [128]
            elapsed time(us): total=170,110 max=84 min=49 avg=50
        73: reduction kernel launched 3372 times
            grid: [1]  block: [256]
            elapsed time(us): total=78,491 max=41 min=22 avg=23
    72: data region reached 6744 times
        72: data copyin transfers: 10116
             device time(us): total=4,447,145 max=686 min=5 avg=439
        77: data copyout transfers: 6744
             device time(us): total=2,107,748 max=631 min=7 avg=312
```
See last few lines about "data copyin/copyout transfers", there is a bottleneck effect between the CPU/memory and GPU limited by the PCI bus. (CPU/Memory about 200GB/s, but PCI-e 3.0x16: 16GB/s or 4.0 x16: 32GB/s).



