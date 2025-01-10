#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

// Host function to initialize bodies
void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

// GPU Kernel to compute forces
__global__ void bodyForce(Body *p, float dt, int n, int blockSize, int blockStride) {
    // 动态共享内存
    extern __shared__ float3 spos[];

    // 计算要处理的数据 index
    int i = threadIdx.x + (int)(blockIdx.x / blockStride) * blockDim.x;
    int start_block = blockIdx.x % blockStride;

    if (i < n) {
        int cycle_times = n / blockSize;
        Body ptemp = p[i];
        Body temp;
        float dx, dy, dz, distSqr, invDist, invDist3;
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int block_num = start_block; block_num < cycle_times; block_num += blockStride) {
            temp = p[block_num * blockSize + threadIdx.x];
            spos[threadIdx.x] = make_float3(temp.x, temp.y, temp.z);

            __syncthreads();

#pragma unroll
            for (int j = 0; j < blockSize; j++) {
                dx = spos[j].x - ptemp.x;
                dy = spos[j].y - ptemp.y;
                dz = spos[j].z - ptemp.z;
                distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                invDist = rsqrtf(distSqr);
                invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            __syncthreads();
        }

        atomicAdd(&p[i].vx, dt * Fx);
        atomicAdd(&p[i].vy, dt * Fy);
        atomicAdd(&p[i].vz, dt * Fz);
    }
}

// GPU Kernel to integrate positions
__global__ void integrate_position(Body *p, float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// Function to print GPU information
void printGPUInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU Name: %s\n", prop.name);
    printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Shared Memory Per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
}

int main(int argc, char **argv) {
    // Default parameters
    int nBodies = 2 << 11; // Default number of bodies
    int BLOCK_SIZE = 32;   // Default block size
    int BLOCK_STRIDE = 1;  // Default block stride

    // Parse command-line arguments
    if (argc > 1) nBodies = atoi(argv[1]);
    if (argc > 2) BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3) BLOCK_STRIDE = atoi(argv[3]);

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(Body);
    float *buf;
    cudaMallocManaged(&buf, bytes);
    Body *p = (Body *)buf;

    size_t threadsPerBlock = BLOCK_SIZE;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock * BLOCK_STRIDE;

    // Initialize data
    randomizeBodies(buf, 6 * nBodies);

    // Print GPU information
    printGPUInfo();

    double totalTime = 0.0;
    float kernelTime = 0.0f;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // Measure kernel execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        bodyForce<<<numberOfBlocks, threadsPerBlock, BLOCK_SIZE * sizeof(float3)>>>(p, dt, nBodies, BLOCK_SIZE, BLOCK_STRIDE);
        integrate_position<<<(nBodies + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(p, dt, nBodies);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        kernelTime += milliseconds;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (iter == nIters - 1) {
            cudaDeviceSynchronize();
        }

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / nIters;
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    float billionsOfOpsPerKernelSecond = 1e-6 * nBodies * nBodies / kernelTime;

    printf("Average Kernel Execution Time: %.3f ms\n", kernelTime / nIters);
    printf("Total Simulation Time (including CPU operations): %.3f ms\n", totalTime * 1000.0);
    printf("Bodies: %d\n", nBodies);
    printf("Block Size: %d\n", BLOCK_SIZE);
    printf("Block Stride: %d\n", BLOCK_STRIDE);
    printf("Interactions Per Second: %.3f Billion\n", billionsOfOpsPerSecond);
    printf("Interactions Per Kernel Second: %.3f Billion\n", billionsOfOpsPerKernelSecond);

    cudaFree(buf);

    return 0;
}