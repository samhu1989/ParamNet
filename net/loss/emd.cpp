#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <limits>
using namespace tensorflow;
REGISTER_OP("emdindex")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    
static void emd(int bi,int n_p,int n_o,const float * xyz1,const float * xyz2,int * ri)
{
    const float eps = 1e-4;
    std::vector<float> price(n_o,10.0);
    std::vector<int> idx_p(n_p,-1); 
    std::vector<int> idx_o(n_o,-1);
    #assigned object number
    int a_n_o = 0;
    while(a_n_o < n_o)
    {
        #biding phase
        std::vector<float> cost_1st(n_p,std::numeric_limits<float>::max());
        std::vector<int> cost_1st_idx(n_p,-1);
        std::vector<float> cost_2nd(n_p,std::numeric_limits<float>::max());
        std::vector<int> cost_2nd_idx(n_p,-1);
        std::vector<bool> isbid_o(n_o,false);
        b = dok_matrix((n_p,n_o),dtype=np.float32);
        #reset cost record;
        for(int i=0 ; i < n_p ; i++)
        {
            if idx_p[i] == -1
            {
                c = price - np.sum(np.square(xyz2[:,:] - xyz1[i,:]),axis=1);
                cost_1st_idx[i] = np.argmin(c);
                cost_1st[i] = c[cost_1st_idx[i]];
                c[cost_1st_idx[i]] = 10*m;
                cost_2nd_idx[i] = np.argmin(c);
                cost_2nd[i] = c[cost_2nd_idx[i]];
                if cost_2nd_idx[i] == -1:
                    r = eps;
                else:
                    r = eps + ( cost_2nd[i] - cost_1st[i] );
                b[i,cost_1st_idx[i]] = price[cost_1st_idx[i]] + r;
                isbid_o[cost_1st_idx[i]] = true;
            }
        }
        #assign
        idx = b.asformat('csc').argmax(axis=0);
        for j in range(n_o):
            if isbid_o[j]==1:
                price[j] = b[idx[0,j],j];
                #the original assignment is released
                if idx_o[j]!=-1:
                    idx_p[idx_o[j]]=-1;
                idx_o[j] = idx[0,j];
                idx_p[idx[0,j]] = j;
        a_n_o = np.sum(idx_o!=-1);
    }

}

static void emd_batch(int b,int n,int m,const float * xyz1,const float * xyz2,int * ri)
{
    for(int bi=0 ; bi < b ; bi++)
    {
        emd(bi,n,m,&(xyz1[bi*n*3]),&(xyz2[bi*m*3]),&(ri[bi*n*2]));
    }
}

class emdindexOp : public OpKernel{
	public:
		explicit emdindexOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("emdindex requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("emdindex only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("emdindex requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("emdindex only accepts 3d point set xyz2"));
			int m=xyz2_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("emdindex expects xyz1 and xyz2 have same batch size"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			Tensor * ri_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,2},&ri_tensor));
			auto ri_flat=ri_tensor->flat<float>();
			int* ri=&(ri_flat(0));
			emd_batch(b,n,m,xyz1,xyz2,ri);
		}
};
REGISTER_KERNEL_BUILDER(Name("emdindex").Device(DEVICE_CPU),emdindexOp);
/*
void emdKernelLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,int * ri);
class emdindexGpuOp : public OpKernel{
	public:
		explicit NnDistanceGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("emdindex requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("emdindex only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("emdindex requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("emdindex only accepts 3d point set xyz2"));
			int m=xyz2_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("emdindex expects xyz1 and xyz2 have same batch size"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			Tensor * ri_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,2},&ri_tensor));
			auto ri_flat=ri_tensor->flat<float>();
			int* ri=&(ri_flat(0));
			emdKernelLauncher(b,n,m,xyz1,xyz2,ri);
		}
};
REGISTER_KERNEL_BUILDER(Name("emdindex").Device(DEVICE_GPU),emdindexGpuOp);
*/