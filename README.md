# Post Training Quantization on Mamba State-Space Models: Theory and Practice

This is the associated github repo for Post-Training Quantization for State-Space Models and Transformers . For more details, see the paper.

Author: Mike Qu

### Installing Dependencies
```
pip3 install -r requirements.txt
```
## Weight-Only Quantization

### Naive Weight-Only Quantization

`naive_quantization.ipynb` contains experiments where mamba is quantized using naive weight quantization only. 

This notebook may be run with any mamba-based methods. Options include 
`mamba-130m-hf`, `mamba-370m-hf`, `mamba-790m-hf`, `mamba-1.4b-hf`, `mamba-2.8b-hf`

To select a quantization scheme, simply uncomment one (and only one) of the methods below.
```
# cur_model = "w8_all"          # per-tensor all submodules absmax
# cur_model = "w8_pc_all"       # per-channel all submodules absmax
# cur_model = "w8_tanh_all"     # per-tensor all submodules tanh quantization (not included in the paper)
cur_model = "w8_inout"          # per-tensor in/out projection only absmax
# cur_model = "w8_pc_inout"     # per-channel in/out projection only absmax
# cur_model = "w8_tanh_inout"   # per-tensor in/out projection only tanh quantization
```
Evaluations will be performed on the original model and the quantized model. This includes testing accuracy on LAMBADA, and perplexity on WikiText. The model footprint is also calculated based on which weights were quantized.

### GPTQ Weight-Only Quantization

We adapt GPTQ to Mamba models and show that advanced weight-only quantization methods for transformer models are also effective on SSMs.

This is contained in the `gptq` directory, adapted from https://github.com/IST-DASLab/gptq.

We implement `mamba.py`, which applies the GPTQ methodology to mamba models. We also fix some deprecated functions that previously prevented the code from running. The resulting file performs GPTQ quantization, then performs an accuracy evalulation on LAMBADA and a perplexity evaluation on WikiText, identical to the experiments from the Naive Weight-Only Quantization notebook.

Example usage:
```
CUDA_VISIBLE_DEVICES=0 python mamba.py state-spaces/mamba-1.4b-hf c4 --wbits 8
```
where the first argument is the particular model of interest, the second parameter is the calibration dataset, and the third parameter wbits refers to the quantization bitwidth.

## Weight and Activation Quantization

### Naive weight-and-activation quantization and Smoothing 

We adapt Smoothquant (https://github.com/mit-han-lab/smoothquant) to work with Mamba models in the `smoothquant` directory.

First, the smoothquant package must be installed using the below command:
```
python setup.py install
```

To perform W8A8 quantization on Mamba or Llama 3.2 models, navigate to `smoothquant/examples`.

`smoothquant_mamba.ipynb` contains all mamba weight-and-activation quantization experiments.

`smoothquant_llama3.ipynb` contains all Llama 3.2 weight-and-activation quantization experiments.

We use the same evaluation schemes as previous quantization methods.

### Hadamard-based quantization (Quamba)

We use Quamba to perform experiments on Hadamard-based quantization directly. The source repo can be found here: https://github.com/enyac-group/Quamba.

To perform experiments, you will need to create a new conda environment specifically for Quamba
```
cd Quamba
conda create -n quamba python=3.10
conda activate quamba
pip install -r requirements.txt
```

Installing 3rd Party Libraries for Quamba
```
# set force build to include 12N, 40N from the newer commit
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
pip install 3rdparty/fast-hadamard-transform

# lm_eval-0.4.2 word2number-1.1
pip install 3rdparty/lm-evaluation-harness

export MAMBA_FORCE_BUILD=TRUE
pip install 3rdparty/mamba

# cmake version >= 3.22.1
bash build_cutlass.sh
```

Build and install the package:
```
pip install .
```
To run experiments in the Quamba directory
```
python expts.py state-spaces/mamba-790m fake --do_hadamard --do_percentile_u --batch_size 16        # Quamba 
python expts.py state-spaces/mamba-790m fake --do_smoothing --do_percentile_u --batch_size 16       # Smoothing expts
```


