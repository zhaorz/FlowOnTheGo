---
layout: default
title: Home
---

# Flow on the Go

### {{ site.description }}

Preliminary Results (Updated May 10)

[Ashwin Sekar](mailto:asekar@andrew.cmu.edu) (asekar)
and [Richard Zhao](mailto:richardz@andrew.cmu.edu) (richardz)

![frames]({{ site.baseurl }}/public/img/temple_3.gif)
![flow]({{ site.baseurl }}/public/img/flow.gif)

The top gif is an input frame sequence, from which we calculate an optical flow (bottom) which
represents movement of subjects in the input.

## Summary

We implement super-realtime (>30fps), high resolution optical flows on a mobile GPU platform. Fast
optical flows allow a realtime video processing pipeline to use the flow in other algorithms such
as object detection or image stabilization.

## Challenges

The main technical challenges associated with this project involve optimizing the algorithm to run
on the NVIDIA Jetson, which has a less powerful CPU and GPU than traditional desktop machines.

Since copying memory between the device and host is the main performance bottleneck, we designed
the architecture as a pipeline which essentially performs copies at just the beginning and end of
the pipeline.

The most significant computational bottleneck in the original implementation was the construction
of image pyramids (a series of downsampled images, and their gradients). We used CUDA kernels to
significantly improve the performance of this step.

Additionally, during the gradient descent phase of the algorithm, which acts on local patches of
the image, careful management of thread blocks is required to hide the system's memory latency.

Finally, all of our optimizations are done while preserving the accuracy of the computed flow. This
makes our approach both fast and accurate enough for realtime use.

## Preliminary Results

All results are from our code running on an NVIDIA Jetson TX2.

### Optical Flow (total)

Using a hybrid GPU-CPU implementation, we achieve an end-to-end latency of roughly 10ms. This
is a speedup of around **10x**.

### Image pyramid construction

The image pyramid construction step was optimized to run in just 3 ms, which is a speedup of
**30x** over our optimized CPU version, which takes 90 ms.

## Remaining Work

Before the deadline, we stil have some final tuning to do on the gradient descent algorithm,
and finalizing the video processing pipeline (currently, our pipeline operates on two images at a
time). The performance figures should remain, however, roughly the same.

