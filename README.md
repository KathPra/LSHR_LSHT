# Local Spherical Harmonics Improve Skeleton-Based Hand Action Recognition
This repo contains the official implementation for Local Spherical Harmonics Improve Skeleton-Based Hand Action Recognition. The paper is accepted to DAGM / GCPR 2023.

## Main Idea
![LSH_Vis](https://github.com/KathPra/LSHR_LSHT/blob/main/LSH_Vis.pdf)

# Prerequisites
We use the same prerequisites as CTR-GCN

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

Further dependencies:
- Run `pip install -r requirements.txt`
- Run `pip install -e torchlight`

### Training

- Select the config file depending on which dataset and modality you are interested in.
- Select the model that you are interested in running. You can create your own model files under `./model`.

```
# Example: training model LSHT_4 on NTU RGB+D 120 on the cross-subject dataset using the joint modality with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.LSHT_4.Model --work-dir work_dir/ntu120/csub/LSHT --device 0
```

### Testing

- To test a trained model saved in <work_dir>, run this command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/csub/LSHT_4 --bone-dir work_dir/ntu120/csub/LSHT_4_bone --joint-motion-dir work_dir/ntu120/csub/LSHT_4_vel --bone-motion-dir work_dir/ntu120/csub/LSHT_4_bone_vel
```

- To calculate hand accuracy, run 
```
python hand_action_acc.py --datasets ntu120/xsub --acc-dir work_dir/ntu120/csub/LSHT_4 --best_ep 64
```

## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!

## Citation


# Contact
For any questions, feel free to contact: `katharina.prasse@uni-siegen.de`
