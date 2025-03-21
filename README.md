# TIFAE

This repo implements the TIFAE for the following paper:
"Temporal Invariant Feature Combined with Arbitrary Enhancement for Missing Modality Emotion Recognition" 

# Environment

``` 
python 3.7.0
pytorch >= 1.0.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

The preprocess of feature was done handcrafted in several steps, we will make it a automatical running script in the next update. You can download the preprocessed feature to run the code.

+ For Training TIFAE on IEMOCAP:

    First training a model fusion model with all audio, visual and lexical modality as the pretrained encoder.

    ```bash
    bash scripts/CAP_pretrained_TIFAE.sh AVL [num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer]
    ```

    Then

    ```bash
    bash scripts/CAP_TIFAE.sh [num_of_expr] [pretrained_num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer]
    ```

+ For Training TIFAE on MSP-IMPROV:

    ```bash
    bash scripts/MSP_pretrained_TIFAE.sh AVL [num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer] 
    ```

    ```bash
    bash scripts/MSP_TIFAE.sh [num_of_expr] [pretrained_num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer]
    ```
    
+ For Training TIFAE on CMU-MOSI: 

    ```bash
    bash scripts/MOSI_pretrained_TIFAE.sh AVL [num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer] 
    ```

    ```bash
    bash scripts/MOSI_TIFAE.sh [num_of_expr] [pretrained_num_of_expr] [GPU_index] [Transformer_head] [Transformer_layer]
    ```



Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get_opt.py and the ```modify_commandline_options``` method of each model you choose.

# Download the features
Baidu Yun Link
IEMOCAP A V L modality Features
链接: https://pan.baidu.com/s/1y0ZFoVV9X8viVabz8HQmAw?pwd=zsc9 提取码: zsc9

Baidu Yun Link
MSP A V L modality Features
链接: https://pan.baidu.com/s/1grE7N_PUJxTWHSl70IijAg?pwd=uhrp 提取码: uhrp

Baidu Yun Link
MOSI A V L modality Features
链接: https://pan.baidu.com/s/1K7IpbhBHUvRkHTGp8SqU7g?pwd=uyx8 提取码: uyx8 

# License
MIT license. 

Copyright (c) 2024 MOE Research Center of Software/Hardware Co-Design Engineering, East China Normal University.
