# MSR-GCN

Official implementation of [MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction](www.baidu.com) (ICCV 2021 paper)

![MSR-GCN](figs/overview.png)

## Dependencies

* Pytorch 1.7.0+cu110
* Python 3.8.5
* Nvidia RTX 3090

## Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

## About datasets

Human3.6M

+ A pose in h3.6m has 32 joints, from which we choose 22, and build the multi-scale by 22 -> 12 -> 7 -> 4 dividing manner.
+ We use S5 / S11 as test / valid dataset, and the rest as train dataset, testing is done on the 15 actions separately, on each we use all data instead of the randomly selected 8 samples.
+ Some joints of the origin 32 have the same position
+ The input / output length is 10 / 25

CMU Mocap dataset

+ A pose in cmu has 38 joints, from which we choose 25, and build the multi-scale by 25 -> 12 -> 7 -> 4 dividing manner.
+ CMU does not have valid dataset, testing is done on the 8 actions separately, on each we use all data instead of the random selected 8 samples.
+ Some joints of the origin 38 have the same position
+ The input / output length is 10 / 25

## train

+ train on Human3.6M: 
  `python main.py --expname=h36m --is_train=1 --output_n=25 --dct_n=35`
+ train on CMU Mocap: 
  `python main.py --expname=cmu --is_train=1 --output_n=25 --dct_n=35`

## evaluate and visualize results

+ evaluate on Human3.6M: 
  `python main.py --expname=h36m --is_load=1 --model_path=ckpt/pretrained/ --output_n=25 --dct_n=35 --test_manner=all`
+ evaluate on CMU Mocap: 
  `python main.py --expname=cmu --is_load=1 --model_path=ckpt/pretrained/ --output_n=25 --dct_n=35 --test_manner=all`

## Results

![img.png](figs/img.png)


## Citing

If you use our code, please cite our work

```
@inproceedings{lingwei2021msrgcn,
  title={MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction},
  author={Lingwei, Dang and Yongwei, Nie and Chengjiang, Long and Qing, Zhang and Guiqing Li},
  booktitle={ICCV},
  year={2021}
}
```

## Acknowledgments

Some of our evaluation code and data process code was adapted/ported from [LearnTrajDep](https://github.com/wei-mao-2019/LearnTrajDep) by [Wei Mao](https://github.com/wei-mao-2019). 

## Licence
MIT
