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

## Summary

We implement super-realtime (>30fps), high resolution optical flows on a mobile GPU platform. Fast
optical flows allow a realtime video processing pipeline to use the flow in other algorithms such
as object detection or image stabilization.

## Challenges

Image pyramids

Number of patches increases

Maintaining accuracy of the flow

Memory management

Optimizing to Jetson (which has a lackluster CPU)

## Preliminary Results

All results are from our code running on an NVIDIA Jetson TX2.

### Optical Flow (total)

Using a hybrid GPU-CPU implementation, we achieve an end-to-end latency of roughly 10ms. This
is a speedup of roughly 10x.

### Image pyramid construction

90 ms => 3 ms (30x speedup)
