#include <mma.h>

#include <torch/torch.h>

struct mmaOutputTile_fp16_gen_gnn{

    int warpin_id;
    const int *values_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_fp16_gen_gnn(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        half* output_matrix, const int *values, half* output_fragment_, int id)
        {
        // 按8x8输出
        //output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*8) + warpin_id/4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*8) + warpin_id/4;        

   

        //结果累加从c0,c1; c2,c3
        for(int i=0;i<2;i++)
        {   
            int temp = *(values_+i*64);
            if(temp!=-1)
            *(output_matrix+ temp)= output_fragment_[2*i];
            
            temp = *(values_+8+i*64);
            if(temp!=-1 )
            *(output_matrix+ temp) = output_fragment_[2*i+1];
        }
       
    
    }
};




/*
TF32 saved as 8x4
*/

struct mmaOutputTile_tf32_gen_gnn{

    int warpin_id;
    const int *values_;


    // Constructor
    __device__ __forceinline__ mmaOutputTile_tf32_gen_gnn(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        float* output_matrix, const int *values, float* output_fragment_, int id)
        {
        //按8x4输出
        //output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*4) + warpin_id/4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*4) + warpin_id/4;       
        if(warpin_id >15) {
            values_ += (32-4);
        }

        //结果累加c0,c1; c2,c3
        for(int i=0;i<2;i++)
        {
            int temp = *(values_+i*64);
            if(temp!=-1)
            *(output_matrix+temp)= output_fragment_[2*i];
        
            temp = *(values_+4+i*64);
            if(temp!=-1)
            *(output_matrix+temp) =output_fragment_[2*i+1];

        }
   
        
    
    }
};



//attention
struct mmaOutputTile_fp16_csr{

    int warpin_id;
    const int *values_;
    // at::Half* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_fp16_csr(
        int lane_id)
    {
        warpin_id=lane_id&31;
    }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, int id,
        half* output_matrix, const int *values, half* output_fragment0_,half* output_fragment1_)
        {
        //output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7)+ ((warpin_id%4)*2*8) + warpin_id/4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*8) + warpin_id/4;        
        //结果累加
        int temp = 0;
        #pragma unroll
        for(int i=0;i<2;i++)
        {
            temp = *(values_+i*64);
            if(temp!=-1)
            *(output_matrix + temp)= __hadd(output_fragment0_[2*i],output_fragment1_[(warpin_id%4)*2]);
            
            temp = *(values_+8+i*64);
            if(temp!=-1)
            *(output_matrix + temp) =__hadd(output_fragment0_[2*i+1],output_fragment1_[(warpin_id%4)*2+1]);
            //  if(warpin_id==4&bid==1&id==0)
            // printf("thread_id:%d ,%d, %d, %d, value is:%f, %f, %f\n",warpin_id,bid,id,(2*i+1), __half2float(*(output_matrix_+16+ i*8)), __half2float(output_fragment0_[2*i+1]),__half2float(output_fragment1_[(warpin_id%4)*2+1]));
        }
        // if(col1==-1){
        // *(output_matrix_)=0.0;
        // *(output_matrix_+ 16)=0.0;
        // }
        // if((col1+8)==-1){
        // *(output_matrix_ + 8 )=0.0;
        // *(output_matrix_+ 24)=0.0;
        // }
        // //输出w1的计算结果，方便后续反向传播
        // if(id==-1 && warpin_id%4==0)
        // {
        //     *(output_w1 +  warpin_id/4) = output_fragment1_[0];
        //     *(output_w1 +  warpin_id/4 +1) = output_fragment1_[2];
        // }

    
    }
};


struct mmaOutputTile_tf32_csr{

    int warpin_id;
    const int *values_;
    // float* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_tf32_csr(
        int lane_id)
    {
        warpin_id=lane_id&31;
    }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, int id,
        float* output_matrix, const int *values, float* output_fragment0_,float* output_fragment1_)
        {
        //k用来做8x4输出的，如果k超过3，即thread16开始的列索引都要偏移一个8x4
        // int k=0;
        // if((warpin_id/4)>=4) k=1;

        //id<<7=id*128, 128为16*8
        //((warpin_id%4)*2*16) thread所在行乘16
        //warpin_id/4为列偏移
       //output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7)+ ((warpin_id%4)*2*4) + warpin_id/4 + k*32 - k*4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*4) + warpin_id/4;       
        if(warpin_id >15) {
            values_ += (32-4);
        }
        //结果累加
        int temp = 0;
        #pragma unroll
        for(int i=0;i<2;i++)
        {
            temp = *(values_+i*64);
            if(temp!=-1)
            *(output_matrix + temp)= output_fragment0_[2*i]+output_fragment1_[(warpin_id%4)*2];
            
            temp = *(values_+4+i*64);
            if(temp!=-1)
            *(output_matrix + temp) = output_fragment0_[2*i+1]+output_fragment1_[(warpin_id%4)*2+1];
        }
        // if(col1==-1){
        // *(output_matrix_)=0.0;
        // *(output_matrix_+ 4)=0.0;
        // }
        // if((col1+8)==-1){
        // *(output_matrix_ + 64)=0.0;
        // *(output_matrix_+ 68)=0.0;
        // }
        // //输出w1的计算结果，方便后续反向传播
        // if(id==-1 && warpin_id%4==0)
        // {
        //     *(output_w1 +  warpin_id/4) = output_fragment1_[0];
        //     *(output_w1 +  warpin_id/4 +1) = output_fragment1_[2];
        // }

    
    }
};
