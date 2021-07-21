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
```f90
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
```f90
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
```f90
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

### 3. Memory/Data management

The First, Most Important, and Possibly Only OpenACC Optimization


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


#### 3.1 Scoped Data Construct Syntax
in F:
```f90
!$acc data [clause …]
structured block
!$acc end data
```
In C:
```c
#pragma acc data [clause …]
{
structured block
}
```

#### 3.2 Data clauses

|  Clauses  | Description |
| ------------- | ------------- |
| `copy(list)`   | **Allocates memory on GPU and copies data from host to GPU when entering region and copies data to the host when exiting region.** <br> Principal use: For many important data structures in your code, this is a logical default to input, modify and return the data. </br> |
| `copyin(list)`   | **Allocates memory on GPU and copies data from host to GPU when entering region.** <br> Principal use: Think of this like an array that you would use as just an input to a subroutine. </br> |
| `copyout(list)`   |**Allocates memory on GPU and copies data to the host when exiting region.** <br> Principal use: A result that isn’t overwriting the input data structure. </br> |
| `create(list)`   | **Allocates memory on GPU but does not copy.** <br> Principal use: Temporary arrays. </br> |

#### 3.3 Array sharing
* Compilers sometimes cannot determine the size of arrays, so we must specify explicitly using data clauses with an array “shape”. The compiler will let you know if you need to do this. Sometimes, you will want to for your own efficiency reasons.

* C:
```c
#pragma acc data copyin(a[0:size]), copyout(b[s/4:3*s/4])
```

* F:
```f90
!$acc data copyin(a(1:size)), copyout(b(s/4:3*s/4))

```
* Fortran uses start:end and C uses start:length

* Data clauses can be used on data, kernels or parallel

Recall the ACC kernel, now we distinguish the code into **data region** and **compute (kernel) region**.
```c
int main(int argc, char** argv){
float A[1000];
#pragma acc data copy(A)
{
  #pragma acc kernels
  for( int iter = 1; iter < 1000 ; iter++){
    A[iter] = 1.0;
    }

A[10] = 2.0;
}
printf("A[10] = %f", A[10]);
}
```

When the compiler sees `#pragma acc data copy(A)`, it copies `A` to the GPU. Then, once it enters the kernel indicated by `#pragma acc kernels`, it runs the kernel on the GPU until the end of the kernel (hence here it's *compute region*). Outside the "compute region", even though it's still in the *data region*, the line in the above segment `A[10] = 2.0;` is still run on the **host**. 

However, data that were copied to the GPU will be copied back to the host only at the end of the *data region*. Hence, whatever operations done earlier (even on the host) will be **overwritten**. Therefore, the result of the above segment is 
```sh
A[10] = 1.0
```

Now, we compare the above segment with the below one, which is the simplest kernel without using any data clauses.
```c
int main(int argc, char** argv){
float A[1000];

#pragma acc kernels
for( int iter = 1; iter < 1000 ; iter++){
  A[iter] = 1.0;
}

A[10] = 2.0;
printf("A[10] = %f", A[10]);
}
```
In this case, we will get the result as 
```sh
A[10] = 2.0 
```

#### 3.3 Data Movement Decisions
* Much like loop data dependencies, sometime the compiler needs your human intelligence to make high-level decisions about data movement. Otherwise, it must remain conservative – sometimes at great cost.

* You must think about when data truly needs to migrate, and see if that is better than the default.

* Besides the scope-based data clauses, there are OpenACC options to let us manage data movement more intensely or asyn

(My take: run minimize data transfer on the GPU, and keep as much data as possible on the GPU (always right?))

#### Exercise 2: Use acc data to minimize transfers
Q: What speedup can you get with data + kernels directives? 

Start with your Exercise 1 solution or grab laplace_bad_acc.c/f90 from the Solutions subdirectory. This is just the solution of the last exercise.

Add data directives where it helps.
* Think: when should I move data between host and GPU? Think how you would do it by hand, then determine which data clauses will implement that plan.
* Hint: you may find it helpful to ignore the output at first and just concentrate on getting the solution to converge quickly (at 3372 steps). Then worry about updating the printout.


Solution:

Performance: 0.58 seconds (bumped up 30x)

### 4. Targeting Architecture hirearchies

The OpenACC spec has methods to target architecture hierarchies, and not just GPUs (think Intel MIC). Let’s see how they map to what we know about GPUs.

Example of Turing 
![image](https://user-images.githubusercontent.com/19921248/126533351-9ecb0ada-3721-4cee-977a-8c1f66222a8f.png)


#### 4.1 Warp
Quantum of independence: Warps

* All the threads in a warp execute the same instruction.
* A thread block consists of one or more warps
* When a warp executes an instruction that accesses global memory it coalesces the memory accesses of the threads within the warp into as few transactions as possible


#### 4.2 Parallel region + kernels
As in OpenMP, the OpenACC parallel construct creates a number of parallel gangs that immediately begin executing the body of the construct *redundantly*. When a gang reaches a work-sharing `loop`, that gang will execute a subset of the loop iterations. One major difference between the OpenACC parallel construct and OpenMP is that there is no barrier at the end of a work-sharing loop in a parallel construct. 

But we need the loop directive!
![image](https://user-images.githubusercontent.com/19921248/126536861-c3de9038-1b94-4a11-8107-4d4d6ddd7344.png)

**Sequential loop**
```c
#pragma acc kernels
{
for (i=0; i<n; i++)
  a(i) = b(i)*c(i)
  for (i=1; i<n-1; i++)
    d(i) = a(i-1) + a(i+1)
}
```
**Parallel loop** (Undesirable)
```c
#pragma acc parallel
{
#pragma acc loop
  for (i=0; i<n; i++)
    a(i) = b(i)*c(i)
#pragma acc loop
  for (i=1; i<n-1; i++)
    d(i) = a(i-1) + a(i+1)
}
```


* The compiler will start some number of gangs and then work-share the iterations of the first loop across those gangs, and work-share the iterations of the second loop across the same gangs.

* There is no synchronization between the first and second loop, so there's no guarantee that the assignment to a(i) from the first loop will be complete before its value is fetched by some other gang for the assignment in the second loop. This will result in incorrect results.

* But the most common reason we use parallel regions is because we want to eliminate these wasted blocking cycles. So we just need some means of controlling them…

#### 4.3 Controlling waiting

We can allow workers, or our CPU, to continue ahead while a loop is executing as we wait at the appropriate times (so we don’t get ahead of our data arriving or a calculation finishing. We do this with `asynch` and `wait` statements.

```c
#pragma acc parallel loop async(1) //We are combining the parallel region and loop directives together. A common idiom.
for (i = 0; i < n; ++i)
  c[i] += a[i];
#pragma acc parallel loop async(2)
 for (i = 0; i < n; ++i)
  b[i] = expf(b[i]);
#pragma acc wait
// host waits here for all async activities to complete
```
* Note that there is no sync available within a parallel region (or kernel)!

#### 4.4 Using Separate Queues
We have up to 16 queues that we can use to manage completion dependencies.

```c
#pragma acc parallel loop async(1) // on queue 1
for (i = 0; i < n; ++i)
  c[i] += a[i];
#pragma acc parallel loop async(2) // on queue 2
for (i = 0; i < n; ++i)
  b[i] = expf(b[i]);
#pragma acc parallel loop async(1) wait(2) // waits for both
for (i = 0; i < n; ++i)
  d[i] = c[i] + b[i];
// host continues executing while GPU is busy
```

#### 4.5 Dependencies
```c
#pragma acc kernels loop independent async(1) // we can use these "async" for kernels too
for (i = 1; i < n-1; ++i) {
#pragma acc cache(b[i-1:3], c[i-1:3])
  a[i] = c[i-1]*b[i+1] + c[i]*b[i] + c[i+1]*b[i-1];
}
#pragma acc parallel loop async(2) wait(1) // start queue 2
for (i = 0; i < n; ++i) // after 1
  c[i] += a[i]; // need a to finish
#pragma acc parallel loop async(3) wait(1) // start queue 3
for (i = 0; i < n; ++i) // after 1
  b[i] = expf(b[i]); // don’t mess with b
// host continues executing while GPU is busy
```

#### 4.6 Private Variables

* One other important consideration for parallel regions is what happens with scaler (non-array) variables inside loops. Unlike arrays, which are divided up amongst the cores, the variables are shared by default. This is often not what you want.

* If you have a scaler inside a parallel loop that is being changed, you probably want each core to have a private copy. This is similar to what we saw earlier with a reduction variable.

```f90
integer nsteps, i
double precision step, sum, x
nsteps = ...
sum = 0
step = 1.0d0 / nsteps
!$acc parallel loop private(x) reduction(+:sum)
do i = 1, nsteps
  x = (i + 0.5d0) * step
  sum = sum + 1.0 / (1.0 + x*x)
enddo
pi = 4.0 * step * sum
```

#### 4.7 Loop clauses
|  Clauses  | Description |
| ------------- | ------------- |
| `private (list)`   | Each thread gets it own copy (implied for index variable). |
| `reduction (operator:list)`   | Also private, but combine at end. **Your responsibility now!**. |
| `gang/worker/vector( )`   | workers |
| `independent`   | Independent. Ignore any suspicions |
| `seq`   | Opposite. Sequential, don’t parallelize. |
| `auto`   | Compiler’s call. |
| `collapse()`   | Says how many nested levels apply to this loop. Unrolls. Good for small inner loops.|
| `tile(,)`   | Opposite. Splits each specified level of nested loop into two. Good for locality.|
| `device_type()` | For multiple devices.|

#### 4.8 Summary: Parallel vs. kernels
|  Advantages of kernels  |  Advantages of parallel |
| ------------- |  ------------- |
| * compiler autoparallelizes <br> * best with nested loops and no procedure calls <br> * one construct around many loop nests can create many device kernels | * some compilers are bad at parallelization <br> * more user control, esp. withprocedure calls.<br>* one construct generates one device kernel <br> * similar to OpenMP |


* To put it simply, **kernels leave more decision making up to the compiler**. There is nothing wrong with trusting the compiler (“trust but verify”), and that is probably a reasonable place to start.
* If you are an OpenMP programmer, you will notice a strong similarity between the tradeoffs of kernels and regions and that of OpenMP parallel for/do versus parallel regions. We will discuss this later when we talk about OpenMP 4.0.
* As you gain experience, you may find that the parallel construct allows you to apply your understanding more explicitly. On the other hand, as the compilers mature, they will also be smarter about just doing the right thing. History tends to favor this second path heavily.

### 5. Data management
