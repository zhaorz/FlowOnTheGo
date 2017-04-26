---
layout: page
title: Checkpoint
date: April 25, 2017
---

---

[Ashwin Sekar](mailto:asekar@andrew.cmu.edu) (asekar)
and [Richard Zhao](mailto:richardz@andrew.cmu.edu) (richardz)

## Progress

Most of our time was spent doing background reading on the subject and rederiving the math behind
the algorithm we plan to use. This provided us with a solid foundation and outline of our
algorithm. We created a reference program that uses OpenCV builtins and created a testing harness
to run it.

The next step was to create the baseline C++ implementation that we would eventually port to CUDA.
We took a good deal of time exploring existing implementations such as
[`OF_DIS`](https://github.com/tikroeger/OF_DIS) and [`image-align`](https://github.com/cheind/image-align)
and decided to iterate on `OF_DIS`, whose author is the same as [1]. We studied the existing
implementation, cleaned parts of it up, and generally modified it to better suit our needs.

The updated schedule reflects a general shift back of roughly a week as both team members are
part of an organization that is heavily involved in Spring Carnival's Booth. This was accounted
for in the original schedule, which had a relatively light workload planned for the final week.

## Updates on Goals and Deliverables

As a result of the schedule shift, we plan to demo our algorithm operating on previously recorded
drone _footage_, with the reach goal being a demo on live drone footage.

## Preliminary results

OpenCV renders a 1024 x 436 frame in 67ms, whereas our baseline C++ implementation renders the same
frame, with smoother output flow, in **15ms**.

## Updated Schedule

| Date     | Milestone                                                     | Done |
| -------- | ------------------------------------------------------------- | ---- |
| April 11 | Complete understanding of the algorithm                       |  ✔️   |
| April 14 | Working OpenCV reference and testing harness                  |  ✔️   |
| April 25 | Working implementation in C++                                 |  ✔️   |
| April 27 | Cleaned up and optimized C++ version                          |      |
| May 1    | Working implementation in CUDA                                |      |
| May 5    | CUDA implementation with same performance as C++ version      |      |
| May 8    | Achieve performance better than published in [1]              |      |
| May 9    | Running on example drone footage                              |      |
| May 11   | Final writeup and demo preparation                            |      |
| May 11   | (_Reach_) Hardware hooked up to drone                         |      |
| May 12   | Final presentation                                            |      |
