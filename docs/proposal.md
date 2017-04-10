---
layout: page
title: Proposal
date: April 10, 2017
---

---

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

## The Challenge

The main challenge is achieving real-time (30 Hz) performance on the limited compute resources of a
mobile GPU while maintaining accuracy comparable to that of state-of-the-art neural nets.

Additionally, managing memory is going to be a significant challenge, as we hope to eventually
target 1920x1080 footage. The algorithm requires two frames to be in memory at all times, and memory
traffic between the CPU and GPU will be high.

Finally, a use case would be a real-time system that provides a video stream as input. In such a
scenario, we would need to design an usable interface for a stream of optical flows.

## Resources

Our starting point will be the method described in

[1] Tim Kroeger, et. al *Fast Optical Flow using Dense Inverse Search* (2016)

which we hope to implement on a GPU (there exists an implemenation for CPUs) and improve by
introducing RGB channels.

Additionally, there is [starter code](https://github.com/tikroeger/OF_DIS) provided by the
author of the paper, which we plan to use as a guide for our own implementation. This is
because a cursory reading of the code indicates significant architectural changes for our
use case.

## Goals and Deliverables

### Metrics

We will use at least 2 widely accepted benchmarks for optical flows:

* [MPI Sintel](http://sintel.is.tue.mpg.de/) a sequence of computer generated videos of various
  'difficulties'
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) a dataset
  consisting of pairs of frames taken from an autonomous testing vehicle.

We will say our implemenation is 'acceptably accurate' if we achieve accuracy that matches the
published results in [1].

### Baseline

Attain acceptable accuracy and achieve the runtime detailed in [1] on a mobile GPU platform.

### Reach

Attain acceptable accuracy for an input stream in full HD, full color in less than 3 ms on a mobile
platform. With a traditional 30fps camera, there is a total of 33.3 ms between consecutive frames.
Computing optical flows in 3 ms allows ample time to apply other algorithms which might require
postprocessing the optical flow.

Implement various use cases of optical flows:
* Video stream stabilization
* Fast object tracking
* Video compression

### Demo

We plan to prepare videos of the optical flow computed from drone footage. If possible, it would be
cool to implement video stabilization and object tracking using our optical flow. Also, it would be
very cool to bring the drone to our presentation for a live demo.

## Platform

We hope to gain access to the NVIDIA Jetson platform to develop the final deliverable. We wish to
connect the board to a mobile video camera, such as one found on a drone or a car.

The Jetson is an intriguing platform, as its relatively compact size allows it to be integrated into
systems such as drones, autonomous vehicles, remote sensors, etc. An increasing amount of video
streams are captured in settings where a traditional computing environment (desktop CPUs and GPUs
and network access) is impossible.

## Schedule

| Date     | Milestone                                                     |
| -------- | ------------------------------------------------------------- |
| April 11 | Complete understanding of the algorithm                       |
| April 15 | Working implementation in C++                                 |
| April 19 | Critical CUDA kernels written and working on desktop GTX 1080 |
| April 25 | **[Checkpoint]** Optimized CUDA version working on a Jetson   |
| April 30 | Interfacing with drone camera and preliminary evaluation      |
| May 2    | Achieve performance as published in [1]                       |
| May 10   | All testing done                                              |
| May 11   | Final writeup and demo preparation                            |
| May 12   | Final presentation                                            |
