#include <stdio.h>

#define NUM_BLOCKS 2
#define NUN_THREAD 1

__global__ void kernel()
{
	printf("Hello CUDA!(%d Thread in %d Block) from Device\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char **argv)
{
	kernel<<<NUM_BLOCKS,NUN_THREAD>>>();

	cudaDeviceSynchronize();

	printf("Hello CUDA! from Host\n");

	return 0;
}
