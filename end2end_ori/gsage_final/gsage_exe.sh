#!/bin/bash
python ./ATT/end2end_ori/gsage_final/eva_gsage_100.py &&
python ./ATT/end2end_ori/gsage_final/eva_gsage_100_dgl.py &&
python ./ATT/end2end_ori/gsage_final/eva_gsage_100_pyg.py &&
python ./ATT/end2end_ori/gsage_final/plot_h100.py