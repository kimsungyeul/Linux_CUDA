#include <stdio.h>

#define NUM_BLOCKS 2
#define NUM_THREAD 5

__global__ void kernel()
{
	printf("Hello CUDA!(%d Thread in %d Block) from Device\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char **argv)
{
	kernel<<<NUM_BLOCKS, NUM_THREAD>>>();

	cudaDeviceSynchronize();

	printf("Hello CUDA! from Host\n");
	return 0;
}
