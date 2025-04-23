from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import torch

data_list = ['ogbn-arxiv', 'ogbn-proteins']
save_path = "/public/home/shijinliang/gnns/"

for data in data_list:
    dataset = DglNodePropPredDataset(name=data, root=save_path)

    # 获取训练、验证、测试索引
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    # 获取 DGL 图和标签
    graph, labels = dataset[0]  

    # 获取节点数
    num_nodes = graph.num_nodes()

    # 获取边的起点和终点
    src_list, dst_list = graph.edges()
    src_list = src_list.numpy()
    dst_list = dst_list.numpy()

    # 获取特征和标签
    features = graph.ndata["feat"].numpy() if "feat" in graph.ndata else np.zeros((num_nodes, 1))  # 如果没有特征，填充 0
    labels = labels.numpy()

    # 获取 mask（转换为 NumPy 数组）
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx.numpy()] = True

    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx.numpy()] = True

    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_idx.numpy()] = True

    # 计算输入和输出大小
    in_size = features.shape[1]  
    out_size = labels.max().item() + 1 if labels.ndim == 1 else labels.shape[1]  # 处理多分类情况

    # 保存到 npz
    np.savez(
        '/home/shijinliang/module/AD/Magicsphere-cmake/dgl_dataset/accuracy/' + data + '.npz',
        num_nodes=num_nodes,
        src_li=src_list,
        dst_li=dst_list,
        features=features,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        in_size=in_size,
        out_size=out_size
    )

    print(f"Saved {data}.npz successfully!")
    
    
    
    
