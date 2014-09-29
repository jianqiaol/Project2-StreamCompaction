#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include "cuda_runtime.h"
#include "CPU_StreamCompaction.h"
#include "device_launch_parameters.h"

using namespace std;

const int n_input=200;
const int iters=1000;
const int threadsPerBlock=512;

#define CPU_SCAN			1;
#define GPU_NAIVE			1;
#define GPU_SHARED_NAIVE	1;
#define GPU_SHARED_OP		1;
#define CPU_SCATTER			1;


__global__ void scan_GPU_naive(int *input,int *output,int n)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n && idx>0)
		output[idx]=input[idx-1];
	__syncthreads();
	for(int d=1;d<n;d*=2)
	{
		if(idx>=d && idx<n)
			output[idx]=output[idx]+output[idx-d];
		__syncthreads();
	}
}

//modified from NVIDIA prefix sum slides
__global__ void scan_GPU_shared_naive(int *input, int *output, int n)
{
	extern __shared__ int sdata[];
	int idx=threadIdx.x;
	int out=0,in=1;
	if(idx<n)
	{
		sdata[idx]=(idx>0)?input[idx-1]:0;
		sdata[n+idx]=0;
	}
	__syncthreads();
    for(int d=1;d<n;d*=2)
	{
		out=1-out;
		in=1-out;
		__syncthreads();
		if(idx>=d && idx<n)
			sdata[out*n+idx]=sdata[in*n+idx]+sdata[in*n+idx-d];
		else
			sdata[out*n+idx]=sdata[in*n+idx];
	}
	output[idx]=sdata[out*n+idx];
}

__global__ void scan_GPU_shared_LG()
//
//__global__ void scan_GPU_shared_op(int *input,int *output,int n)
//{
//	extern __shared__ int sdata[];
//	int idx=threadIdx.x;
//	int offset=1;
//	if(2*idx<n)
//		sdata[2*idx]=input[2*idx];
//	if(2*idx+1<n)	
//		sdata[2*idx+1]=input[2*idx+1];
//	__syncthreads();
//	for(int d=n>>1;d>0;d>>=1)
//	{
//		if(idx<d)
//		{
//			int ai=offset*(2*idx+1)-1;
//			int bi=offset*(2*idx+2)-1;
//		}
//		__syncthreads();
//		
//
//	}
//
//
//
//}




int main()
{
	int *a=new int[n_input];
	int *aux=new int[n_input];
	int *scan=new int[n_input];
	int *scatter=new int[n_input];
	int *d_a,*d_scan,*d_aux;
	float time=0.0f;
	cudaMalloc(&d_a,n_input*sizeof(int));
	cudaMalloc(&d_scan,n_input*sizeof(int));
	cudaMalloc(&d_aux,n_input*sizeof(int));

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	int *answer_scan=new int[n_input];
	int *answer_scatter=new int[n_input];
	int num_nonzero=0;
	int num_zero=0;
	for(int i=0;i<n_input;i++)
	{
		//a[i]=i;
		a[i]=rand()%5;
		if(a[i]!=0)
		{
			answer_scatter[num_nonzero]=a[i];
			num_nonzero+=1;
		}
		else
		{
			answer_scatter[n_input-1-num_zero]=0;
			num_zero+=1;
		}
		scan[i]=0;
		scatter[i]=0;
		aux[i]=0;
	}
	cudaMemcpy(d_a,a,n_input*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_scan,scan,n_input*sizeof(int),cudaMemcpyHostToDevice);

	answer_scan[0]=0;
	for(int i=1;i<n_input;i++)
		answer_scan[i]=answer_scan[i-1]+a[i-1];


#if CPU_SCAN
	//CPU scan
	cout<<"----------------------"<<endl;
	cout<<"CPU scan test"<<endl;
	clock_t begin=clock();
	
	for(int k=0;k<iters;k++)
		scan_CPU(a,scan,n_input);
	clock_t end=clock();
	cout<<"Runtime for "<<iters<<" iters="<<end-begin<<" ms"<<endl;
	postprocess(answer_scan,scan,n_input);
#endif



#if GPU_NAIVE
	//GPU naive scan
	cout<<"----------------------"<<endl;
	cout<<"GPU Naive scan"<<endl;
	int dimBlock=threadsPerBlock;
	int dimGrid=int((n_input+dimBlock-1)/dimBlock);
	cudaEventRecord(start, 0);
	for(int i=0;i<iters;i++)
	{
	scan_GPU_naive<<<dimGrid,dimBlock>>>(d_a,d_scan,n_input);
	}
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop );
    cudaEventElapsedTime( &time, start, stop);
	cout << "Runtime for " << iters << " iters=" << time << " ms" << endl;
	cudaMemcpy(scan,d_scan,n_input*sizeof(int),cudaMemcpyDeviceToHost);
	postprocess(answer_scan,scan,n_input);
#endif


#if GPU_SHARED_NAIVE
	//GPU scan with shared memory
	cout<<"----------------------"<<endl;
	cout<<"GPU scan with shared memory"<<endl;
	dimBlock=threadsPerBlock;
	dimGrid=int((n_input+dimBlock-1)/dimBlock);
	cudaEventRecord(start, 0);
	for(int i=0;i<iters;i++)
	{
	scan_GPU_shared_naive<<<dimGrid,dimBlock,2*n_input*sizeof(int)>>>(d_a,d_scan,n_input);
	}
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop );
	time=0.0f;
    cudaEventElapsedTime( &time, start, stop);
	cout << "Runtime for " << iters << " iters=" << time << " ms" << endl;
	cudaMemcpy(scan,d_scan,n_input*sizeof(int),cudaMemcpyDeviceToHost);
	postprocess(answer_scan,scan,n_input);
#endif

#if CPU_SCATTER
	//CPU scan
	cout<<"----------------------"<<endl;
	cout<<"CPU scatter test"<<endl;
	begin=clock();
	for(int k=0;k<iters;k++)
		scatter_CPU(a,scatter,n_input);
	end=clock();
	cout<<"Runtime for "<<iters<<" iters="<<end-begin<<" ms"<<endl;
	postprocess(answer_scatter,scatter,n_input);
#endif






	free(a);
	free(scan);
	free(answer_scan);
	cudaFree(d_a);
	cudaFree(d_scan);
	getchar();
	return 0;
}