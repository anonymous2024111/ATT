#!/bin/bash
python ./ATT/end2end_ori/gatv2_final/eva_gatv2_100.py &&
python ./ATT/end2end_ori/gatv2_final/eva_gatv2_100_dgl.py &&
python ./ATT/end2end_ori/gatv2_final/eva_gatv2_100_pyg.py &&
python ./ATT/end2end_ori/gatv2_final/plot_h100.py