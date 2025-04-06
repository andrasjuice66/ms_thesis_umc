# SynthSegContainer

Containerized [SynthSeg](https://github.com/BBillot/SynthSeg).

[Containers on
DockerHub](https://hub.docker.com/repository/docker/cookpa/synthseg/general).

## Building with docker

Before building, download the models as instructed in the [SynthSeg
installation instructions](https://github.com/BBillot/SynthSeg#installation) and place
them in `synthseg_models/`.

Then, build with one of the available Dockerfiles.


## Dockerfile.conda

Installs tensorflow-gpu from conda, which includes GPU support. This works for Singularity
containers with `singularity run --nv`.

This also runs without the GPU. MKL support is disabled, which extends execution
time (to about 8-10 min) but massively reduces the memory requirements.


## Dockerfile.conda-cuda

As above, but use a cuda runtime layer. Possibly useful for use with the GPU in Docker,
rather than Singularity.


## Dockerfile.conda.mkl

Installs tensorflow-mkl, which uses MKL optimizations. This is 2-3x faster than
the non-MKL tensorflow, but it requires MUCH more memory. The memory use can be reduced
with the environment variable
[MKL_DISABLE_FAST_MM](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/support-functions/memory-management/mkl-disable-fast-mm.html)
but even still, with the default crops, memory use peaks at around 40 Gb.


## Dockerfile.tfsrc

This builds without CUDA, but enhances CPU optimization by building tensorflow from
source. This takes longer and by default optimizes the tensorflow binary for "native"
architecture. This is customizable by changing the configuration options in
`tf_configure.bazelrc.in`.

I'm not sure how much this helps, it doesn't seem much faster than the conda
tensorflow-mkl version.

This dockerfile has not been updated to reflect the simplified requirements
provided by SynthSeg developers in March 2023.


## Usage

The entrypoint to the container is the SynthSeg prediction script
`SynthSeg_predict.py`. Run without args to see usage.


## Docker containers

Available on
[DockerHub](https://hub.docker.com/repository/docker/cookpa/synthseg/general).


## Singularity

For GPU support, build from synthseg:conda-latest and run with `singularity --nv`.


## Licensing and citations

[SynthSeg license](https://github.com/BBillot/SynthSeg/blob/master/LICENSE.txt).

The authors ask that if it is used in published research, to cite:

SynthSeg: Domain Randomisation for Segmentation of Brain MRI Scans of any
Contrast and Resolution
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V.
Dalca, J.E. Iglesias (https://pubmed.ncbi.nlm.nih.gov/36857946/)

For cortical parcellation, automated QC, or robust fitting, please also cite the following
paper:

Robust Segmentation of Brain MRI in the Wild with Hierarchical CNNs and no Retraining
B. Billot, M. Colin, S.E. Arnold, S. Das, J.E. Iglesias [MICCAI
2022](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_52)
