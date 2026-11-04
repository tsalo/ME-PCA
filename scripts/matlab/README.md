# MATLAB reference script

`run_TensorMPPCA.m` is the original MATLAB driver script that the `mepca` command
line interface replaces. It is kept for reference only; it expects the MATLAB
[Tensor-MP-PCA](https://github.com/sunenj/Tensor-MP-PCA) toolbox on the MATLAB path.

The equivalent Python call is:

```bash
mepca sub-01_ses-1_task-rest_echo-{1,2,3,4}_part-mag_bold.nii.gz \
    --output-dir func_preproc_tmppca --window 4 4
```
