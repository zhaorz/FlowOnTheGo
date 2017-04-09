# Proposal

##### Ashwin Sekar <asekar@andrew.cmu.edu> and Richard Zhao <richardz@andrew.cmu.edu>

## Summary

We implement real time phase-based optical flows on mobile GPU platforms for video stabilization and
object tracking.

## Background

A common problem in computer vision is detecting moving objects on a background. With an increasing
amount of cameras mounted on moving vehicles, stabilization of the video feed is a crucial
preprocessing task.

Optical flows present an elegant solution to a wide class of problems such as the above. An optical
flow is a vector field that describes per-pixel displacements between two consecutive video frames
in a video feed.

In recent years, there has been increased interest in algorithms for computing optical flows,
especially ones that achieve a mix of efficiency and accuracy. The phase-based method, initially
proposed by Fleet and Jepson in 1990, presents a method for computing these flows by finding the
corresponding component velocity vectors of the input transformed into the phase space. The approach
is more robust than traditional techniques to changes in contrast, scale, orientation, and speed.

The price paid for higher accuracy is a higher computational cost, mainly associated with the
filtering operations required by the transformation into phase space. It turns out that the
algorithm operates exclusively on localized data, making it a good candidate for parallelization on
GPUs.

## The Challenge

The main challenge is achieving realtime (30 Hz) performance on the limited compute resources of a
mobile GPU.

## Resources

[1] Karl Pauwels, et. al *Realtime Phase-based Optical Flow on the GPU* (2008)

## Goals and Deliverables

## Platform

## Schedule

| -------- | ------------------------------------------------------------- |
| April 11 | Complete understanding of the algorithm                       |
| -------- | ------------------------------------------------------------- |
| April 15 | Working implementation in C++                                 |
| -------- | ------------------------------------------------------------- |
| April 19 | Critical CUDA kernels written and working on desktop GTX 1080 |
| -------- | ------------------------------------------------------------- |
| April 25 | **[Checkpoint]** Optimized CUDA version working on Kepler K1  |
| -------- | ------------------------------------------------------------- |
| May 12   | Final presentation                                            |
| -------- | ------------------------------------------------------------- |
