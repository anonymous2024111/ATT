#include <torch/torch.h>
#include <omp.h>
#include <chrono>

struct Value_csr {
    //8x16
    std::vector<int> value_csr;
    //8x8
    std::vector<int> value_templete_csr;
    //8x4
    std::vector<int> value_templete_csr2;
    std::vector<int> colum;
    int row;
    int pad;
};







//balance
struct Value_balance_fp16 {
    std::vector<at::Half> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_balance_tf32 {
    std::vector<float> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_balance_sddmm {
    std::vector<int> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

std::vector<torch::Tensor> blockProcess_fp16_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t)
{
    // auto start = std::chrono::high_resolution_clock::now();
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<at::Half, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_fp16> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_fp16 v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        int r=v_size;
        while(pad-r>0)
        {
        v.colum.push_back(-1);
        r++;
        }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<at::Half> demo(pad*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];

                demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks  > 0){
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            v.row.push_back(pad-v_size);
            v.window.push_back(i);
            v.atomic.push_back(0);
        }else{
             // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    v.row.push_back(0);
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                v.row.push_back(pad - v_size);
                v.window.push_back(i);
                v.atomic.push_back(1);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        v.row.push_back(0);
                    }
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
            }
        }
    }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<at::Half> valueNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
    }
    // auto end = std::chrono::high_resolution_clock::now();

    // // 计算时间差并转换为毫秒
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << duration.count() << " ms" << std::endl;
    
    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat16);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor,t_atomicTensor};
}


std::vector<torch::Tensor> blockProcess_tf32_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_tf32> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_tf32 v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        int r=v_size;
        while(pad-r>0)
        {
        v.colum.push_back(-1);
        r++;
        }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<float> demo(pad*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
   
                    demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];

            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            v.row.push_back(pad-v_size);
            v.window.push_back(i);
            v.atomic.push_back(0);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    v.row.push_back(0);
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                v.row.push_back(pad - v_size);
                v.window.push_back(i);
                v.atomic.push_back(1);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        v.row.push_back(0);
                    }
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
            }
        }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<float> valueNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor,t_atomicTensor};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("blockProcess_fp16_balance", &blockProcess_fp16_balance, "Block for FP16 with any shape");
    m.def("blockProcess_tf32_balance", &blockProcess_tf32_balance, "Block for TF32 with any shape");
}