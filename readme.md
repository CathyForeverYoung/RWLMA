# RWLMA


This repo contains code for our SIGIR 2019 paper: Xuejiao Yang and Bang Wang. [“Local Matrix Approximation based on Graph Random Walk” ](https://dl.acm.org/doi/10.1145/3331184.3331338) Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2019.


Please cite this paper if you use our codes. Thanks!


# How to use

- You should inherit the class Template with your data set, and make sure the input of training data and test data are lists with tuples of (user, item, rating).

- Then run it:  
```
python main_RWLMA.py
```