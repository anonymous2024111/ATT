#!/bin/bash
python ./ATT/end2end_ori/gcn_final/eva_gcn_100.py &&
python ./ATT/end2end_ori/gcn_final/eva_gcn_100_dgl.py &&
python ./ATT/end2end_ori/gcn_final/eva_gcn_100_pyg.py &&
python ./ATT/end2end_ori/gcn_final/plot_h100.py