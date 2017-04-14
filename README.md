# Flow on the Go

[Ashwin Sekar](mailto:asekar@andrew.cmu.edu) (asekar)
and [Richard Zhao](mailto:richardz@andrew.cmu.edu) (richardz)

## Summary

We implement real time optical flows on a mobile GPU platform using the dense inverse search method.

## Background

A common problem in computer vision is detecting moving objects on a background. With an increasing
amount of cameras mounted on moving vehicles, stabilization of the video feed is a crucial
preprocessing task.

Optical flows present an elegant solution to a wide class of problems such as the above. An optical
flow is a vector field that describes per-pixel displacements between two consecutive video frames
in a video feed.

In recent years, there has been increased interest in algorithms for computing optical flows,
especially ones that achieve a mix of efficiency and accuracy. Kroeger et. al. propose a method with
very low time complexity and competitive accuracy for computing dense optical flow<sup>[1]</sup>.

The algorithm is highly parallelizable, which gives it the potential to achieve super-real-time
(faster than 30 Hz) performance on GPUs.

## Build

### Reference

```shell
make flow_ref
```

## Resources

[1] Tim Kroeger, et. al *Fast Optical Flow using Dense Inverse Search* (2016)

## Schedule

| Date     | Milestone                                                     | Done |
| -------- | ------------------------------------------------------------- | ---- |
| April 11 | Complete understanding of the algorithm                       |  ✔️ |
| April 15 | Working implementation in C++                                 |      |
| April 19 | Critical CUDA kernels written and working on desktop GTX 1080 |      |
| April 25 | **[Checkpoint]** Optimized CUDA version working on a Jetson   |      |
| April 30 | Interfacing with drone camera and preliminary evaluation      |      |
| May 2    | Achieve performance as published in [1]                       |      |
| May 10   | All testing done                                              |      |
| May 11   | Final writeup and demo preparation                            |      |
| May 12   | Final presentation                                            |      |
