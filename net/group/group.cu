#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cassert>
__device__ inline void swapf(float & a, float & b)
{   
    float tmp = a;
    a = b;
    b = tmp;
}

__device__ inline void swap(int & a, int & b)
{
    int tmp = a;
    a = b ;
    b = tmp;
}

__global__ void KnnKernel(int b,const int n,const int d,const float * xyz,const int k,float * result,int * result_i){
    const int size = 4096;
    __shared__ float dist[size];
    __shared__ int idx[size];
    assert( n <= size );
    for ( int bi = blockIdx.x ; bi < b ; bi += gridDim.x )
    {
        for ( int i = blockIdx.y ;  i < n  ; i += gridDim.y )
        {
            for ( int j = threadIdx.x ; j < n ; j += blockDim.x )
            {
                if( i == j ){
                    dist[j] = 0;
                    idx[j]  = j;
                    continue;
                }
                float d = 0.0;
                for ( int di = 0 ; di < d ; ++di )
                {
                    float dif = xyz[(bi*n+i)*3+di] - xyz[(bi*n+j)*3+di];
                    d += dif*dif;
                }
                dist[j] = d;
                idx[j] = j;
            }
            __syncthreads();
            //odd-even sort
	    int pownum = int(log2(float(n)));
	    if ( n != pow(2, pownum) ){
            for ( int cnt = 0 ; cnt < ( n + 1 ) / 2 ; ++cnt )
            {
                for ( int j = 2*threadIdx.x + 1 ; j < n ; j += 2*blockDim.x )
                {
                    if ( dist[j] < dist[ j - 1 ] )
                    {
                        swapf(dist[j], dist[j-1]);
                        swap(idx[j], idx[j-1]);
                    }
                }
                __syncthreads();
                for ( int j = 2*threadIdx.x + 2 ; j < n ; j += 2*blockDim.x )
                {
                    if ( dist[j] < dist[ j - 1 ] )
                    {
                        swapf(dist[j], dist[j-1]);
                        swap(idx[j], idx[j-1]);
                    }
                }
                __syncthreads();
            }
	    }else{	
            //Bitonic Sort
            for (unsigned int t = 2; t <= n ; t *= 2)
            {
                // Bitonic merge:
                for (unsigned int j = t / 2; j>0; j /= 2)
                {	
			for (unsigned int tid = threadIdx.x ; tid < n ; tid += blockDim.x )
                    	{
				unsigned int ixj = tid ^ j;
                    		if (ixj > tid)
                    		{
                        		if ((tid & t) == 0)
                        		{
                            			if (dist[tid] > dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                        		else
                        		{
                            			if (dist[tid] < dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                    		}
                    		
			}
			__syncthreads();	
                }
            }
	    }
            __syncthreads();
            //copy result
            for ( int j = threadIdx.x ; j < k  ; j += blockDim.x )
            {
                result[(bi*n+i)*k+j] = dist[j+1];
                result_i[ ((bi*n+i)*k+j)*2+0 ] = bi;
                result_i[ ((bi*n+i)*k+j)*2+1 ] = idx[j+1];
            }
            
        }
    }
}
void KnnKernelLauncher(int b,const int n,const int d,const float * xyz,const int k,float * result,int * result_i){
    KnnKernel<<<dim3(b,16,1),512>>>(b,n,d,xyz,k,result,result_i);
}
#endif