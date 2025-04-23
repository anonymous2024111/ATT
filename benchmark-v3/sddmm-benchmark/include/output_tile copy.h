#include <mma.h>

#include <torch/torch.h>

/*
FP16 saved as 8x8
*/
struct mmaOutputTile_fp16_gen{

    int warpin_id;
    const int *values_;
    at::Half* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_fp16_gen(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        at::Half* output_matrix, const int *values,at::Half* output_fragment_, int id, int col, int col1, int nonvectors)
        {
        //按8x8输出
        output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*8) + warpin_id/4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*16) + warpin_id/4;        

        if((id+1)*16 <= nonvectors){
            //结果累加
            for(int i=0;i<2;i++)
            {
                if(*(values_+i*8)!=0)
                *(output_matrix_+ i*64)= output_fragment_[2*i];
                
                if(*(values_+16+i*8)!=0 )
                *(output_matrix_+8+ i*64) = output_fragment_[2*i+1];
    
            }
        }else{
            int residue = nonvectors - (id*16);
            int pad = 16 - residue;

            values_ -= (warpin_id%4)*2*pad;
            if(residue>8)
            {
                if(col>=0){
                    if(*(values_)!=0)
                    *(output_matrix_)= output_fragment_[0];
                    if(*(values_+ residue)!=0 )
                    *(output_matrix_+8) = output_fragment_[1];
                }
                if(col1>=0){
                    output_matrix_ -= (warpin_id%4)*2*pad;
                    if(*(values_+8)!=0)
                    *(output_matrix_ + 64 )= output_fragment_[2];
                    if(*(values_+ 8 + residue)!=0 )
                    *(output_matrix_+ 64 +residue ) = output_fragment_[3];
                }
            }else{
                if(col>=0){
                    output_matrix_ -= (warpin_id%4)*2*(pad-8);
                    if(*(values_)!=0)
                    *(output_matrix_)= output_fragment_[0];
                    if(*(values_+ residue)!=0 )
                    *(output_matrix_+residue) = output_fragment_[1];
                }
            }

        }


        //8x16输出
        // output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7)+ ((warpin_id%4)*2*16) + warpin_id/4;
        // values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*16) + warpin_id/4;       
        
        // //结果累加
        // //8x16按8x4输出

        // #pragma unroll
        // for(int i=0;i<2;i++)
        // {
        //     if(*(values_+i*8)!=0)
        //     *(output_matrix_+ i*8)= output_fragment_[2*i];
            
        //     if(*(values_+16+i*8)!=0)
        //     *(output_matrix_+16+ i*8) =output_fragment_[2*i+1];
  
        // }


    
    }
};

struct mmaOutputTile_fp16_gen_16{

    int warpin_id;
    const at::Half *values_;
    at::Half* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_fp16_gen_16(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        at::Half* output_matrix, const at::Half *values,at::Half* output_fragment_ , int id )
        {
        output_matrix_ = output_matrix + row_offset_vec*16 + (id<<7) +  (warpin_id/4)*8 + (warpin_id%4)*2;
        values_ = values+ row_offset_vec*16 + (id<<7) + (warpin_id/4)*8 + (warpin_id%4)*2;   
        
        //结果累加
        #pragma unroll
        for(int i=0;i<2;i++)
        {
            if(*(values_+i*64)!=0)
            *(output_matrix_+ i*64)= output_fragment_[2*i];
              
            if(*(values_+1+i*64)!=0)
            *(output_matrix_+ 1 + i*64) = output_fragment_[2*i+1];
  
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


/*
TF32 saved as 8x4
*/

struct mmaOutputTile_tf32_gen{

    int warpin_id;
    const int *values_;
    float* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_tf32_gen(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        float* output_matrix, const int *values, float* output_fragment_, int id, int col, int col1, int nonvectors)
        {
        //k用来做8x4输出的，如果k超过3，即thread16开始的列索引都要偏移一个8x4
        int k=0;
        if((warpin_id/4)>=4) k=1;
        //row_offset_vec为vector数量
        //((warpin_id%4)*2*8)行偏移
        //warpin_id/4列偏移
        // int row = m_index_vec*8 + (warpin_id%4)*2;
        // int col = warpin_id/4;  
     
        output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7);
        // output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7)+ ((warpin_id%4)*2*4) + warpin_id/4 + k*32 - k*4;
        values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*16) + warpin_id/4;       
        
        //结果累加
        //8x16按8x4输出
        if((id+1)*16 <= nonvectors){
            output_matrix_ += ((warpin_id%4)*2*4) + warpin_id/4 + k*32 - k*4;
            for(int i=0;i<2;i++)
            {
                if(*(values_+i*8)!=0)
                *(output_matrix_+ i*64)= output_fragment_[2*i];
                
                if(*(values_+16+i*8)!=0)
                *(output_matrix_+4+ i*64) =output_fragment_[2*i+1];
    
            }
        }else{
            int residue = nonvectors - (id*16);
            int pad = 16 - residue;
            values_ -= (warpin_id%4)*2*pad;

            if(warpin_id<16){
                //先写c0, c1
                if(col>=0){
                    int offset = 4;
                    if(residue<4) offset = residue;
                    output_matrix_ += ((warpin_id%4)*2*offset)+warpin_id/4;

                    if(*(values_)!=0)
                    *(output_matrix_)= output_fragment_[0];
                    if(*(values_+ residue)!=0 )
                    *(output_matrix_+ offset) = output_fragment_[1];

                    // if(blockIdx.x==0 and blockIdx.y==0 and threadIdx.x==0)
                    // {
                    //     printf("%d\n",(*(values_+ residue)));
                    //     printf("%d\n",(*(values_)));
                    // }
                }
                if(col1>=0){
                    int offset = 4;
                    if(residue<12) offset = residue-8;
                    output_matrix_ += 64 + ((warpin_id%4)*2*offset)+warpin_id/4;

                    if(*(values_ + 8)!=0)
                    *(output_matrix_)= output_fragment_[2];
                    if(*(values_+ 8 + residue)!=0 )
                    *(output_matrix_+ offset) = output_fragment_[3];
                }

            }
            else{
                //先写c0, c1
                if(col>=0){
                    int offset = 4;
                    if(residue<8) offset = residue-4;
                    output_matrix_ += ((warpin_id%4)*2*offset) + 32 + (warpin_id-16)/4;

                    if(*(values_)!=0)
                    *(output_matrix_)= output_fragment_[0];
                    if(*(values_+ residue)!=0 )
                    *(output_matrix_+ offset) = output_fragment_[1];
                }
                if(col1>=0){
                    int offset = 4;
                    if(residue<16) offset = residue-12;
                    output_matrix_ += 64 + ((warpin_id%4)*2*offset) + 32 +  (warpin_id-16)/4;

                    if(*(values_ + 8)!=0)
                    *(output_matrix_)= output_fragment_[2];
                    if(*(values_+ 8 + residue)!=0 )
                    *(output_matrix_+ offset) = output_fragment_[3];
                }
            }


        }
      

        // output_matrix_ = output_matrix + row_offset_vec*8 + (id<<7)+ ((warpin_id%4)*2*16) + warpin_id/4;
        // values_ = values+ row_offset_vec*8 + (id<<7) + ((warpin_id%4)*2*16) + warpin_id/4;       
        
        // //结果累加
        // //8x16

        // #pragma unroll
        // for(int i=0;i<2;i++)
        // {
        //     if(*(values_+i*8)!=0)
        //     *(output_matrix_+ i*8)= output_fragment_[2*i];
            
        //     if(*(values_+16+i*8)!=0)
        //     *(output_matrix_+16+ i*8) =output_fragment_[2*i+1];
  
        // }

    
    }
};

//16x1

struct mmaOutputTile_tf32_gen_16{

    int warpin_id;
    const float *values_;
    float* output_matrix_;

    // Constructor
    __device__ __forceinline__ mmaOutputTile_tf32_gen_16(int lane_id)
    {  warpin_id=lane_id&31; }

    // Store
    __device__ __forceinline__ void Store(long row_offset_vec, 
        float* output_matrix, const float *values, float* output_fragment_, int id)
        {

        output_matrix_ = output_matrix + row_offset_vec*16 + (id<<7) +  (warpin_id/4)*8 + (warpin_id%4)*2;
        values_ = values+ row_offset_vec*16 + (id<<7) + (warpin_id/4)*8 + (warpin_id%4)*2;   
        
        
        //结果累加
        #pragma unroll
        for(int i=0;i<2;i++)
        {
            if(*(values_+i*64)!=0)
            *(output_matrix_+ i*64)= output_fragment_[2*i];
              
            if(*(values_+1+i*64)!=0)
            *(output_matrix_+ 1 + i*64) = output_fragment_[2*i+1];
  
        }
    //   if(col1==-1){
    //     *(output_matrix_)=0.0;
    //     *(output_matrix_+ 4)=0.0;
    //     }
    //     if((col1+8)==-1){
    //     *(output_matrix_ + 64)=0.0;
    //     *(output_matrix_+ 68)=0.0;
    //     }
        // //输出w1的计算结果，方便后续反向传播
        // if(id==-1 && warpin_id%4==0)
        // {
        //     *(output_w1 +  warpin_id/4) = output_fragment1_[0];
        //     *(output_w1 +  warpin_id/4 +1) = output_fragment1_[2];
        // }

    
    }
};