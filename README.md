# MSR-GCN

Official implementation of [MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction](https://openaccess.thecvf.com/content/ICCV2021/html/Dang_MSR-GCN_Multi-Scale_Residual_Graph_Convolution_Networks_for_Human_Motion_Prediction_ICCV_2021_paper.html) (ICCV 2021 paper)

[\[Paper\]](https://openaccess.thecvf.com/content/ICCV2021/papers/Dang_MSR-GCN_Multi-Scale_Residual_Graph_Convolution_Networks_for_Human_Motion_Prediction_ICCV_2021_paper.pdf)
[\[Supp\]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Dang_MSR-GCN_Multi-Scale_Residual_ICCV_2021_supplemental.pdf)
[\[Poster\]](./assets/7627-poster.pdf)
[\[Slides\]](./assets/7627-slides-v4.5.pptx)
[\[Video\]](./assets/7627-slides-v4.5.mp4)

**2022.09.06: Note that for stochastic long-term human motion prediction which aims to producing future sequences with high plausibility and diversity, our new work [Diverse Human Motion Prediction via Gumbel-Softmax Sampling from an Auxiliary Space (ACMMM 2022)](https://github.com/Droliven/diverse_sampling) is available.**


## Authors

<!--   <div style="display:flex;flex-direction:row;flex-wrap:wrap;justify-content:space-around;align-items:center;">
    <div style="display:flex;flex-direction:column;flex-wrap:wrap;justify-content:center;align-items:center;">
        <a href="https://github.com/Droliven" style="text-align: center;"><img src="./assets/lingweidang.png" width="40%"></a>
        <p>
          <a href="https://github.com/Droliven">[Lingwei Dang]</a>
        </p>
      </div>
      <div style="display:flex;flex-direction:column;flex-wrap:wrap;justify-content:center;align-items:center;">
        <a href="https://nieyongwei.net" style="text-align: center;"><img src="./assets/yongweinie.png" width="40%"></a>
        <p>
          <a href="https://nieyongwei.net">[Yongwei Nie]</a>
        </p>
      </div>
      <div style="display:flex;flex-direction:column;flex-wrap:wrap;justify-content:center;align-items:center;">
        <a href="http://www.chengjianglong.com" style="text-align: center;"><img src="./assets/chengjianglong.png" width="60%"></a>
        <p>
          <a href="http://www.chengjianglong.com">[Chengjiang Long]</a>
        </p>
      </div>
      <div style="display:flex;flex-direction:column;flex-wrap:wrap;justify-content:center;align-items:center;">
        <a href="http://zhangqing-home.net/" style="text-align: center;"><img src="./assets/qingzhang.png" width="40%"></a>
        <p>
          <a href="http://zhangqing-home.net/">[Qing Zhang]</a>
        </p>
      </div>
      <div style="display:flex;flex-direction:column;flex-wrap:wrap;justify-content:center;align-items:center;">
        <a href="http://www2.scut.edu.cn/cs/2017/0629/c22284a328097/page.htm" style="text-align: center;"><img src="./assets/guiqingli.png" width="40%"></a>
        <p>
          <a href="http://www2.scut.edu.cn/cs/2017/0629/c22284a328097/page.htm">[Guiqing Li]</a>
        </p>
      </div>
  </div>
 -->
1. [Lingwei Dang](https://github.com/Droliven), School of Computer Science and Engineering, South China University of Technology, China, [levondang@163.com](mailto:levondang@163.com)
2. [Yongwei Nie](https://nieyongwei.net), School of Computer Science and Engineering, South China University of Technology, China, [nieyongwei@scut.edu.cn](mailto:nieyongwei@scut.edu.cn)
3. [Chengjiang Long](http://www.chengjianglong.com), JD Finance America Corporation, USA, [cjfykx@gmail.com](mailto:cjfykx@gmail.com)
4. [Qing Zhang](http://zhangqing-home.net/), School of Computer Science and Engineering, Sun Yat-sen University, China, [zhangqing.whu.cs@gmail.com](mailto:zhangqing.whu.cs@gmail.com)
5. [Guiqing Li](http://www2.scut.edu.cn/cs/2017/0629/c22284a328097/page.htm), School of Computer Science and Engineering, South China University of Technology, China, [ligq@scut.edu.cn](mailto:ligq@scut.edu.cn)


## Overview


<a href="./assets/7627-poster.pdf">
  <img src="./assets/7627-poster.png" />
</a>


  &nbsp;&nbsp;&nbsp;  Human motion prediction is a challenging task due to the stochasticity and aperiodicity of future poses. Recently, graph convolutional network (GCN) has been proven to be very effective to learn dynamic relations among pose joints, which is helpful for pose prediction. On the other hand, one can abstract a human pose recursively to obtain a set of poses at multiple scales. With the increase of the abstraction level, the motion of the pose becomes more stable, which benefits pose prediction too. In this paper, we propose a novel multi-scale residual Graph Convolution Network (MSR-GCN) for human pose prediction task in the manner of end-to-end. The GCNs are used to extract features from fine to coarse scale and then from coarse to fine scale. The extracted features at each scale are then combined and decoded to obtain the residuals between the input and target poses. Intermediate supervisions are imposed on all the predicted poses, which enforces the network to learn more representative features. Our proposed approach is evaluated on two standard benchmark datasets, i.e., the Human3.6M dataset and the CMU Mocap dataset. Experimental results demonstrate that our method outperforms the state-of-the-art approaches.


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

## Train

+ train on Human3.6M:

  `python main.py --exp_name=h36m --is_train=1 --output_n=25 --dct_n=35 --test_manner=all`

+ train on CMU Mocap:

  `python main.py --exp_name=cmu --is_train=1 --output_n=25 --dct_n=35 --test_manner=all`


## Evaluate and visualize results

+ evaluate on Human3.6M:

  `python main.py --exp_name=h36m --is_load=1 --model_path=ckpt/pretrained/h36m_in10out25dctn35_best_err57.9256.pth --output_n=25 --dct_n=35 --test_manner=all`

+ evaluate on CMU Mocap: 
  
  `python main.py --exp_name=cmu --is_load=1 --model_path=ckpt/pretrained/cmu_in10out25dctn35_best_err37.2310.pth --output_n=25 --dct_n=35 --test_manner=all`

## Results

H3.6M-10/25/35-all | 80 | 160 | 320 | 400 | 560 | 1000 | -
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:
walking | 12.16 | 22.65 | 38.65 | 45.24 | 52.72 | 63.05 | -
eating | 8.39 | 17.05 | 33.03 | 40.44 | 52.54 | 77.11 | -
smoking | 8.02 | 16.27 | 31.32 | 38.15 | 49.45 | 71.64 | -
discussion | 11.98 | 26.76 | 57.08 | 69.74 | 88.59 | 117.59 | -
directions | 8.61 | 19.65 | 43.28 | 53.82 | 71.18 | 100.59 | -
greeting | 16.48 | 36.95 | 77.32 | 93.38 | 116.24 | 147.23 | -
phoning | 10.10 | 20.74 | 41.51 | 51.26 | 68.28 | 104.36 | -
posing | 12.79 | 29.38 | 66.95 | 85.01 | 116.26 | 174.33 | -
purchases | 14.75 | 32.39 | 66.13 | 79.63 | 101.63 | 139.15 | -
sitting | 10.53 | 21.99 | 46.26 | 57.80 | 78.19 | 120.02 | -
sittingdown | 16.10 | 31.63 | 62.45 | 76.84 | 102.83 | 155.45 | -
takingphoto | 9.89 | 21.01 | 44.56 | 56.30 | 77.94 | 121.87 | -
waiting | 10.68 | 23.06 | 48.25 | 59.23 | 76.33 | 106.25 | -
walkingdog | 20.65 | 42.88 | 80.35 | 93.31 | 111.87 | 148.21 | -
walkingtogether | 10.56 | 20.92 | 37.40 | 43.85 | 52.93 | 65.91 | -
Average | 12.11 | 25.56 | 51.64 | 62.93 | 81.13 | 114.18 | 57.93 

****

CMU-10/25/35-all | 80 | 160 | 320 | 400 | 560 | 1000 | -
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:
basketball | 10.24 | 18.64 | 36.94 | 45.96 | 61.12 | 86.24 | -
basketball_signal | 3.04 | 5.62 | 12.49 | 16.60 | 25.43 | 49.99 | -
directing_traffic | 6.13 | 12.60 | 29.37 | 39.22 | 60.46 | 114.56 | -
jumping | 15.19 | 28.85 | 55.97 | 69.11 | 92.38 | 126.16 | -
running | 13.17 | 20.91 | 29.88 | 33.37 | 38.26 | 43.62 | - 
soccer | 10.92 | 19.40 | 37.41 | 47.00 | 65.25 | 101.85 | -
walking | 6.38 | 10.25 | 16.88 | 20.05 | 25.48 | 36.78 | - 
washwindow | 5.41 | 10.93 | 24.51 | 31.79 | 45.13 | 70.16 | -
Average | 8.81 | 15.90 | 30.43 | 37.89 | 51.69 | 78.67 | 37.23 

[comment]: <> (****)

[comment]: <> (H3.6M-10/10/15-8 | 80 | 160 | 320 | 400 | -)

[comment]: <> (:----: | :----: | :----: | :----: | :----: | :----:)

[comment]: <> (walking | 8.72 | 15.52 | 28.37 | 32.36 | )

[comment]: <> (eating | 8.29 | 17.67 | 36.30 | 43.66 | )

[comment]: <> (smoking | 7.51 | 15.43 | 27.42 | 31.52 | )

[comment]: <> (discussion | 9.33 | 22.14 | 40.55 | 45.55 | )

[comment]: <> (directions | 11.41 | 21.90 | 45.78 | 56.15 | )

[comment]: <> (greeting | 13.51 | 26.51 | 68.80 | 86.15 | )

[comment]: <> (phoning | 11.78 | 20.59 | 37.46 | 41.72 | )

[comment]: <> (posing | 8.49 | 21.79 | 61.24 | 76.44 | )

[comment]: <> (purchases | 18.95 | 38.70 | 64.54 | 72.59 | )

[comment]: <> (sitting | 11.31 | 26.52 | 56.15 | 69.17 | )

[comment]: <> (sittingdown | 11.06 | 28.22 | 56.14 | 66.77 | )

[comment]: <> (takingphoto | 6.59 | 15.80 | 40.75 | 53.09 | )

[comment]: <> (waiting | 8.89 | 20.89 | 53.61 | 69.78 | )

[comment]: <> (walkingdog | 24.39 | 53.58 | 95.64 | 110.43 | )

[comment]: <> (walkingtogether | 8.69 | 18.52 | 35.37 | 45.59 | )

[comment]: <> (Average | 11.26 | 24.25 | 49.87 | 60.06 | 36.36 )

[comment]: <> (****)

[comment]: <> (CMU-10/10/15-8 | 80 | 160 | 320 | 400 | -)

[comment]: <> (:----: | :----: | :----: | :----: | :----: | :----: )

[comment]: <> (basketball | 12.18 | 22.01 | 45.51 | 57.96 | )

[comment]: <> (basketball_signal | 2.63 | 5.37 | 13.96 | 18.74 | )

[comment]: <> (directing_traffic | 6.48 | 13.49 | 29.59 | 38.28 | )

[comment]: <> (jumping | 14.02 | 29.77 | 75.12 | 98.06 | )

[comment]: <> (running | 17.68 | 21.01 | 19.17 | 21.95 | )

[comment]: <> (soccer | 8.11 | 14.90 | 33.73 | 41.98 | )

[comment]: <> (walking | 5.70 | 8.81 | 16.36 | 19.87 | )

[comment]: <> (washwindow | 5.01 | 10.35 | 28.38 | 37.69 | )

[comment]: <> (Average | 8.98 | 15.71 | 32.73 | 41.82 | 24.81 )

  
## Citation

If you use our code, please cite our work

```
@InProceedings{Dang_2021_ICCV,
    author    = {Dang, Lingwei and Nie, Yongwei and Long, Chengjiang and Zhang, Qing and Li, Guiqing},
    title     = {MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11467-11476}
}
```

## Acknowledgments

Some of our evaluation code and data process code was adapted/ported from [LearnTrajDep](https://github.com/wei-mao-2019/LearnTrajDep) by [Wei Mao](https://github.com/wei-mao-2019). 

## Licence
MIT
