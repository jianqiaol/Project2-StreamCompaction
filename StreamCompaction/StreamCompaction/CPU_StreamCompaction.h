#ifndef CPU_STREAMCOMPACTION_H_
#define CPU_STREAMCOMPACTION_H_
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <cstdlib>
using namespace std;

//Check the result, this function is from shehzan's code in profiling and debugging lab
void postprocess(const int *ref, const int *res, int n)
{
    bool passed = true;
    for (int i = 0; i < n; i++)
    {
        if (res[i] != ref[i])
        {
            printf("ID:%d \t Res:%d \t Ref:%d\n", i, res[i], ref[i]);
			for(int j=0;j<n;j++)
				cout<<ref[j]<<" "<<res[j]<<endl;
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if(passed)
        printf("Post process check passed!!\n");
}


void scan_CPU(int *input,int *output,int n)
{
	output[0]=0;
	for(int i=1;i<n;i++)
		output[i]=output[i-1]+input[i-1];
}

void scatter_CPU(int *input,int *output, int n)
{
	int k=0;
	for(int i=0;i<n;i++)
	{
		if(input[i]!=0)
		{
			output[k]=input[i];
			k+=1;
		}
	}

}
#endif