#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

// Host function to initialize bodies
void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

// GPU Kernel to compute forces
__global__ void bodyForce(Body *p, float dt, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

// GPU Kernel to integrate positions
__global__ void integrate_position(Body *p, float dt, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// Function to print GPU information
void printGPUInfo()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU Name: %s\n", prop.name);
    printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Shared Memory Per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
}

int main(int argc, char **argv)
{
    // Default parameters
    int nBodies = 2 << 11; // Default number of bodies
    int BLOCK_SIZE = 32;   // Default block size

    // Parse command-line arguments
    if (argc > 1)
        nBodies = atoi(argv[1]);
    if (argc > 2)
        BLOCK_SIZE = atoi(argv[2]);

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(Body);
    float *buf;
    cudaMallocManaged(&buf, bytes);
    Body *p = (Body *)buf;

    size_t threadsPerBlock = BLOCK_SIZE;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize data
    randomizeBodies(buf, 6 * nBodies);

    // Print GPU information
    printGPUInfo();

    double totalTime = 0.0;
    float kernelTime = 0.0f;

    for (int iter = 0; iter < nIters; iter++)
    {
        StartTimer();

        // Measure kernel execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies);
        integrate_position<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        kernelTime += milliseconds;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (iter == nIters - 1)
        {
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
    printf("Interactions Per Second: %.3f Billion\n", billionsOfOpsPerSecond);
    printf("Interactions Per Kernel Second: %.3f Billion\n", billionsOfOpsPerKernelSecond);

    cudaFree(buf);

    return 0;
}