# Enhancing Joint Multiple Intent Detection and Slot Filling with Global Intent-Slot Co-occurrence

Requirements:

-   numpy==1.19.1
-   tqdm==4.50.0
-   pytorch==1.2.0
-   python==3.6.12
-   cudatoolkit==9.2
-   fitlog==0.9.13
-   ordered-set==4.0.2


## Run 
The **train.py** acts as a main function to the project, you can run it by the following commands.
```Shell
# MixATIS_clean dataset (ON Tesla T4)
python train.py -g -bs=16 -dd=./data/MixATIS_clean -sd=./save/MixATIS_best -ne=200 -wd=1e-6 -sddr=0.0 -gdr=0.4 -lalpha=0.8 -salpha=0.2 -wed=128 -ied=384 -sed=384 -ehd=256 -sahd=1024 -saod=128 -god=384

# MixSNIPS_clean dataset (ON Tesla T4)
python train.py -g -bs=16 -dd=./data/MixSNIPS_clean -sd=./save/MixSNIPS_best -ne=150 -wd=5e-4 -sddr=0.6 -gdr=0.0 -lalpha=0.7 -salpha=0.3 -wed=128 -ied=384 -sed=384 -ehd=256 -sahd=1024 -saod=128 -god=384
```
## Notes and Acknowledgments
The implementation is based on  https://github.com/yizhen20133868/GL-GIN
