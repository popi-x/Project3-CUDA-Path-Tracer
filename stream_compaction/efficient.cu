#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

static int blockSize = 128;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
		using StreamCompaction::Common::kernScatter;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* x, int step) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            
            if (index % step == 0) {
				x[index + step - 1] += x[index + (step / 2) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int* d_down, int step) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            
            if (index % step == 0) {
                int temp = d_down[index + (step / 2) - 1];
                d_down[index + (step / 2) - 1] = d_down[index + step - 1];
                d_down[index + step - 1] += temp;
			}

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int paddedSize = 1 << ilog2ceil(n);
            int size = paddedSize * sizeof(int);
            int* d_up;
            int step;

            dim3 fullBlocksPerGrid((paddedSize + blockSize - 1) / blockSize);

            cudaMalloc((void**)&d_up, size);
            cudaMemcpy(d_up, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_up + n, 0, (paddedSize - n) * sizeof(int));


            timer().startGpuTimer();

            // TODO
		    for (int i = 0; i < ilog2ceil(n); i++) {
                step = 1 << (i + 1);
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedSize, d_up, step);
            }
            cudaMemset(d_up + paddedSize - 1, 0, sizeof(int));

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
				step = 1 << (i + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedSize, d_up, step);
			}
            
		    timer().endGpuTimer();

            cudaMemcpy(odata, d_up, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_up);

        }



        void scanWithoutTimer(int n, int* odata, const int* idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int size = paddedSize * sizeof(int);
            int* d_up;
            int step;

            dim3 fullBlocksPerGrid((paddedSize + blockSize - 1) / blockSize);

            cudaMalloc((void**)&d_up, size);
            cudaMemcpy(d_up, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_up + n, 0, (paddedSize - n) * sizeof(int));

            for (int i = 0; i < ilog2ceil(n); i++) {
                step = 1 << (i + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedSize, d_up, step);
            }
            cudaMemset(d_up + paddedSize - 1, 0, sizeof(int));

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                step = 1 << (i + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedSize, d_up, step);
            }

            cudaMemcpy(odata, d_up, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_up);
        }


        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            int* dev_idata, * dev_bools, * dev_indices, * dev_odata;
            int* bools = new int[n];
            int* indices = new int[n];

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
           
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
            cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

            scanWithoutTimer(n, indices, bools);

            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMemcpy(dev_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);

            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

                    
            timer().endGpuTimer();

            int count = indices[n - 1] + bools[n - 1];

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            delete[] bools;
            delete[] indices;

            return count;
        }
    }
}
