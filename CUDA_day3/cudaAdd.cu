#include <stdio.h>

__global__ void add(int *a, int *b, int *sum){
	*sum = *a + *b;
}

int main(int argc, char** argv)
{
	int a = 2, b = 4, sum;
	int *dev_a,*dev_b,*dev_sum;
	int size = sizeof(int);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_sum, size);

	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1, 1>>>(dev_a, dev_b, dev_sum);

	cudaMemcpy(&sum, dev_sum, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_sum);

	printf("2 + 4 = %d from CUDA\n", sum);

	return 0;
}
