/**
 * Helpful timers
 */

#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>
#include <chrono>

namespace timer {

  typedef std::chrono::high_resolution_clock Clock;

  /**
   * General purpose high resolution timer.
   *
   * Usage:
   *
   *  auto start = now();
   *
   *  [...] // Compute something
   *
   *  double duration = calc_print_elapsed("myComputation", start);
   *
   */
  inline std::chrono::time_point<Clock> now() {
    return Clock::now();
  }

  inline double time_diff(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> end) {
    auto dt = end - start;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(dt);
    return us.count() / 1000.0;
  }

  inline double calc_print_elapsed(const char* name, std::chrono::time_point<Clock> start) {
#if (VERBOSE)
    double duration = time_diff(start, now());
    std::cout << "[time] " << duration << " (ms) : " << name  << std::endl;
    return duration;
#else
    return 0.0;
#endif
  }

}

#endif // end __TIMER_H__

