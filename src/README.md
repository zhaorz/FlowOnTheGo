# FlowOnTheGo

GPU accelerated optical flow using Dense Inverse Search.

## Compiling

On an x86 machine

```
$ mkdir build && cd build
$ cmake -DARCH=x86 ..
$ make -j
```

On a NVIDIA Jetson TX2 (ARM)

```
$ mkdir build && cd build
$ cmake -DARCH=ARM -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
$ make -j
```

## Citations

### Original Dense Inverse Search

`@inproceedings{kroegerECCV2016,
   Author    = {Till Kroeger and Radu Timofte and Dengxin Dai and Luc Van Gool},
   Title     = {Fast Optical Flow using Dense Inverse Search},
   Booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
   Year      = {2016}} `

### Variational Refinement

` @inproceedings{weinzaepfelICCV2013,
    TITLE = {{DeepFlow: Large displacement optical flow with deep matching}},
    AUTHOR = {Weinzaepfel, Philippe and Revaud, J{\'e}r{\^o}me and Harchaoui, Zaid and Schmid, Cordelia},
    BOOKTITLE = {{ICCV 2013 - IEEE International Conference on Computer Vision}},
    YEAR = {2013}} `

