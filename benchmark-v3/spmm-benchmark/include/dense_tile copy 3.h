
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
// using namespace nvcuda;


    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>2)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 8;
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<4;i++)
            {
                if((dimN_index + global_offset+i)<colEdge)
                dense_tile_half[i]=__ldg(matrix_base_ +offset+ i);
            }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            int xid=(warpin_id>>2)&1;  //列0，1，0，1
            //定义临时数组tmp
            float tmp[2];
            at::Half *p=reinterpret_cast<at::Half *>(tmp);
            if(xid==0)
            {
                p[0]=dense_tile_half_temp[2];
                p[1]=dense_tile_half_temp[3];
            }
            else
            {
                p[0]=dense_tile_half_temp[0];
                p[1]=dense_tile_half_temp[1];
            }
            //第一次shuflle
            
            tmp[0] = __shfl_xor_sync(0xffffffff, tmp[0], 4,32);
            //交换后赋值
            if(xid==0)
            {
            p[3]=p[1];
            p[1]=p[0];
            p[0]=dense_tile_half_temp[0];
            p[2]=dense_tile_half_temp[1];
            }else
            {
            p[0]=p[0];
            p[2]=p[1];
            p[1]=dense_tile_half_temp[2];
            p[3]=dense_tile_half_temp[3];
            }
            
            //tmp交替写入dense_tile_
            int warpin_offset=((warpin_id&3)<<2) + (xid<<1);
            int k=(warpin_id&3)>>1; //行 0，0，1，1
            if(k==0){
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2)+ (warpin_id>>3))=tmp[0];
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2)+(warpin_id>>3))=tmp[1];
            }
            else{
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if(row_offsets_ >= 0){
                 const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                for(int i=0;i<4;i++)
                {
                    if((dimN_index + global_offset+i)<colEdge)
                    dense_tile_half[i]=__ldg(matrix_base_ + offset+i);
                    // if(blockIdx.y==1 && threadIdx.x==32){
                    // printf("11111 ");
                    // printf("%.1f  ",__half2float(dense_tile_half[i]));
                    //  printf("%d  ", row_offsets_);
                    //  printf("%d  ", global_offset);
                    //  printf("\n");
                    //  }

                }
            }
            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            int xid=(warpin_id>>2)&1;  //列0，1，0，1
            //定义临时数组tmp
            float tmp[2];
            at::Half *p=reinterpret_cast<at::Half *>(tmp);
            if(xid==0)
            {
                p[0]=dense_tile_half_temp[2];
                p[1]=dense_tile_half_temp[3];
            }
            else
            {
                p[0]=dense_tile_half_temp[0];
                p[1]=dense_tile_half_temp[1];
            }
            //第一次shuflle
            tmp[0] = __shfl_xor_sync(0xffffffff, tmp[0], 4,32);
            //交换后赋值
            if(xid==0)
            {
            p[3]=p[1];
            p[1]=p[0];
            p[0]=dense_tile_half_temp[0];
            p[2]=dense_tile_half_temp[1];
            }else
            {
            p[0]=p[0];
            p[2]=p[1];
            p[1]=dense_tile_half_temp[2];
            p[3]=dense_tile_half_temp[3];
            }

            //tmp交替写入dense_tile_
            int warpin_offset=((warpin_id&3)<<2) + (xid<<1);
            int k=(warpin_id&3)>>1; //行 0，0，1，1
            if(k==0){
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
            }
            else{
                *(dense_tile_+(warp_id<<6) + ((warpin_offset+1)<<2) +(warpin_id>>3))=tmp[1];
                *(dense_tile_+(warp_id<<6) + (warpin_offset<<2) + (warpin_id>>3))=tmp[0];
            }
            
        }
    };

    //fp16 16
    struct mmaDenseTile_fp16_16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_16(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*4) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            for(int i=0;i<2;i++){
                sparse_fragment_[i]=__ldg(values_);
                values_ += 32;
            }
            at::Half dense_tile_half_temp[2]={0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            const int global_offset = (warp_id<<3) + (warpin_id/4);

            if((global_offset+dimN_index)<colEdge)
            {
                for(int i=0;i<2;i++)
                {
                    const long row_offsets_ = __ldg(column_idxs_ + i);
                    const long offset = (row_offsets_*rhs_cols_) + global_offset;
                    dense_tile_half[i]=__ldg(matrix_base_ +offset);

                }
            }
            for(int i=0;i<2;i++)
            *(dense_tile_ + i)=dense_tile_half[i];

            column_idxs_ += 8;
                //     if( blockIdx.x==0 and blockIdx.y==1 and blockIdx.z==0 and threadIdx.x==0)
                // {
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                // }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            for(int i=0;i<2;i++){
                sparse_fragment_[i]=__ldg(values_);
                values_ += 32;
            }
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            at::Half dense_tile_half_temp[2]={0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if((global_offset+dimN_index)<colEdge)
            {
                for(int i=0;i<2;i++){
                    const long row_offsets_ = __ldg(column_idxs_ + i);
                    if(row_offsets_ >= 0)
                    {
                        const long offset = (row_offsets_*rhs_cols_) + global_offset;
                        dense_tile_half[i]=__ldg(matrix_base_ +offset);
                    }
                
                }
            }
            for(int i=0;i<2;i++)
            *(dense_tile_ + i)=dense_tile_half[i];
                //        if( blockIdx.x==0 and blockIdx.y==0 and blockIdx.z==0 and threadIdx.x==0)
                // {
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                //     half * l =reinterpret_cast< half *>(sparse_fragment_);
                //     printf("%f %f \n", __half2float( dense_tile_[0]), __half2float(dense_tile_[1]));
                // }

            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
           
            
        }
    };

    //fp16 ns
    struct mmaDenseTile_fp16_ns{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_ns(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>2)),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 8;
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            for(int i=0;i<4;i++)
            {
                if((dimN_index + global_offset+i)<colEdge)
                dense_tile_half[i]=*(matrix_base_ +offset+ i);
            }
        
            // const int global_offset = (warp_id<<2) + (warpin_id&3);
            // dense_tile_half =reinterpret_cast<const half *>(matrix_base_ + (row_offsets_)*rhs_cols_ + global_offset);
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            
            //tmp交替写入dense_tile_
            for(int i=0; i<4; i++)
                *(dense_tile_+(warp_id<<7) + (warpin_id/4) + ((warpin_id%4)*32) + i*8)=dense_tile_half[i];
 
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id&3)<<2);
            at::Half dense_tile_half_temp[4]={0.0,0.0,0.0,0.0};
            half * dense_tile_half = reinterpret_cast<half *>(dense_tile_half_temp);
            if(row_offsets_ >= 0){
                 const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                for(int i=0;i<4;i++)
                {
                    if((dimN_index + global_offset+i)<colEdge)
                    dense_tile_half[i]=*(matrix_base_ + offset+i);
                    // if(blockIdx.y==1 && threadIdx.x==32){
                    // printf("11111 ");
                    // printf("%.1f  ",__half2float(dense_tile_half[i]));
                    //  printf("%d  ", row_offsets_);
                    //  printf("%d  ", global_offset);
                    //  printf("\n");
                    //  }

                }
            }
            // if(blockIdx.y==1 && threadIdx.x==32){
            // printf("666666 ");
            // for(int i=0;i<4;i++)
            // printf("%.1f  ",__half2float(dense_tile_half[i]));
            // printf("\n");}
            //warp内部开始shuffle
            //shuffle需要先打包需要交换的值
            for(int i=0; i<4; i++)
                *(dense_tile_+(warp_id<<7) + (warpin_id/4) + ((warpin_id%4)*32) + i*8)=dense_tile_half[i];
            
        }
    };    




        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>3)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
            }     
            //    if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     printf("%d\n",warp_id);
            //     printf("%d\n",((warpin_id%8)*2)*4);
            //     printf("%d\n",(((warpin_id%8)*2)+1)*4);
            //     printf("%d\n",warpin_id/8);
            //     printf("%d\n",dense_tile_fp32[0]);
            //     printf("%d\n",dense_tile_fp32[1]);
            //     printf("%d\n",row_offsets_*rhs_cols_);
            //     printf("%d\n",global_offset);
            // }
            int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            if(k==0){
                *(dense_tile_ + warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4  + warpin_id/8) = dense_tile_fp32[1];
            }
            else{
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) =dense_tile_fp32[0];
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i)<colEdge)
                    dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
                 }   
            }
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            if(k==0){
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            }
            else{
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            }
        }
    };
    

    //16
    struct mmaDenseTile_tf32_16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_16(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*16) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){
            // float sparse[2];
            // float a = 1.0;
            // for(int i=0; i<2; i++)
            // {
            //     sparse[i]=__ldg(values_);
            //     //sparse_fragment_[i]=1.0;
            //     values_ += 32;
            // }
            // for(int i =0; i<2; i++)
            // sparse_fragment_[i]=a;
            for(int i =0; i<2; i++){
            sparse_fragment_[i]= __ldg(values_);
            // if(blockIdx.x==0 and blockIdx.y==1220 and blockIdx.z==0 and threadIdx.x==0)
            // printf("%f ", sparse_fragment_[i]);
            values_ += 32;}
            const long row_offsets_ = __ldg(column_idxs_);
            // values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            const long offset = (row_offsets_*rhs_cols_) + global_offset;

            float dense_tile_fp32[1]={0.0};
            if((dimN_index+global_offset)<colEdge)
            dense_tile_fp32[0] =__ldg(matrix_base_ +offset);
             
             *(dense_tile_)=dense_tile_fp32[0];
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            // float sparse[2];
            //             float a = 1.0;
            // for(int i=0; i<2; i++)
            // {
            //     // sparse_fragment_[i]=__ldg(values_);
            //     sparse[i]=__ldg(values_);
            //     values_ += 32;
            // }
            // for(int i =0; i<2; i++)
            // sparse_fragment_[i]=a;
            for(int i =0; i<2; i++){
            sparse_fragment_[i]= __ldg(values_);
            values_ += 32;}
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            // float dense_tile_fp32[2]={0.0,0.0};
            float dense_tile_fp32[1]={0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
 
                if((dimN_index+global_offset)<colEdge)
               dense_tile_fp32[0] = __ldg(matrix_base_ +offset);
                
            }
            *(dense_tile_)=dense_tile_fp32[0];
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            // int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            // if(k==0){
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            // }
            // else{
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            // }
        }
    };

// tf32 - ns
    struct mmaDenseTile_tf32_ns{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_ns(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)>>3)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
            }     
            //    if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     printf("%d\n",warp_id);
            //     printf("%d\n",((warpin_id%8)*2)*4);
            //     printf("%d\n",(((warpin_id%8)*2)+1)*4);
            //     printf("%d\n",warpin_id/8);
            //     printf("%d\n",dense_tile_fp32[0]);
            //     printf("%d\n",dense_tile_fp32[1]);
            //     printf("%d\n",row_offsets_*rhs_cols_);
            //     printf("%d\n",global_offset);
            // }
     
                *(dense_tile_ + warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4  + warpin_id/8) = dense_tile_fp32[1];


        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + ((warpin_id%8)*2);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i)<colEdge)
                    dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i);
                 }   
            }
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            // int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            // if(k==0){
                *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
                *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            // }
            // else{
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            // }
        }
    };


        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half2 *matrix_base_;
        half2 *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_v2(
        long row_offset_vec,
        const double * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const double*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(reinterpret_cast<const half2 *>(matrix + offset)),
            // row_offsets_base_(row_offsets),
            values_(reinterpret_cast<const float *>(values + row_offset_vec*2) + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec + (((lane_id & 31)%4)*2)),
            dense_tile_(reinterpret_cast< half2 *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            int temp = 0;
            if((warpin_id%8)>=4) temp = 1;
            long col_temp = __ldg(column_idxs_ + temp);
            for(int i=0;i<2;i++)
            {
                 int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                 long offset = (col_temp*rhs_cols_/2) + global_offset;
                dense_tile_[i]=__ldg(matrix_base_ +offset);
            }
            //temp=0: 分别取第1,3个数； temp=1: 分别取第0,2个数；
            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[0].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4,32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }
            column_idxs_ += 8;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);

            int temp = 0;
            if((warpin_id%8)>4) temp = 1;
            long col_temp = __ldg(column_idxs_ + temp);

            for(int i=0;i<2;i++)
            {
                if(col_temp >= 0)
                {
                     int global_offset = (warp_id<<3) + (warpin_id/8) + (i*4);
                     long offset = (col_temp*rhs_cols_/2) + global_offset;
                    dense_tile_[i]=__ldg(matrix_base_ +offset);
                }else{
                    // dense_tile_[i].x = __float2half(0.0f);
                    // dense_tile_[i].y = __float2half(0.0f);
                    dense_tile_[i] = __floats2half2_rn(0.0f, 0.0f);
                }
            }
            half2 ex;
            if(temp == 0){
                ex.x =  dense_tile_[0].y;
                ex.y =  dense_tile_[1].y;
            }else{
                ex.x =  dense_tile_[0].x;
                ex.y =  dense_tile_[0].x;
            }
            //做shuffle
            ex = __shfl_xor_sync(0xffffffff, ex, 4, 32);
            //shuffle完，更新dense_tile_
            if(temp == 0){
                dense_tile_[0].y = ex.x;
                dense_tile_[1].y = ex.y;
            }else{
                dense_tile_[0].x = ex.x;
                dense_tile_[1].x = ex.y;
            }

        }
    };

        //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_v2(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ + offset + i*8);
            } 
            for(int i=0;i<2;i++)
                dense_tile_[i]=dense_tile_fp32[i];    
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i*8)<colEdge)
                        dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i*8);
                 }   
            }
            for(int i=0;i<2;i++)
                dense_tile_[i]=dense_tile_fp32[i];
        }
    };