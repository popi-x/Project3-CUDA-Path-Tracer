#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

static int blockSize = 128;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int* d_a, int* d_b, int offset) {
			int index = threadIdx.x + blockIdx.x * blockDim.x;
			if (index >= n) return;

                if (index >= offset) {
					d_b[index] = d_a[index] + d_a[index - offset];
                }
                else {
					d_b[index] = d_a[index];
                }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = n * sizeof(int);
            int* d_a, * d_b;
            cudaMalloc((void**)&d_a, size);
            cudaMalloc((void**)&d_b, size);

            cudaMemcpy(d_a, idata, size, cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);


            timer().startGpuTimer();
            // TODO
		    for (int d = 1; d <= ilog2ceil(n); d++) {
				int offset = 1 << (d - 1);
                kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, d_a, d_b, offset);
                std::swap(d_a, d_b);
			}

            int* temp = new int[n];
             
            if (ilog2ceil(n) % 2 == 0) {
				cudaMemcpy(temp, d_a, size, cudaMemcpyDeviceToHost);
            }
            else {
				cudaMemcpy(temp, d_b, size, cudaMemcpyDeviceToHost);
            }

			odata[0] = 0;
            for (int i = 0; i < n; i++) {
                odata[i + 1] = temp[i];
            }
            
            timer().endGpuTimer();
        }
    }
}
