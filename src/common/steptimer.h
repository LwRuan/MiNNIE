#pragma once

#include <chrono>

namespace Rain {
class StepTimer {
 public:
  using duration = std::chrono::duration<double>;
  using steady_clock = std::chrono::steady_clock;
  using time_point =
      std::chrono::time_point<std::chrono::steady_clock, duration>;

  duration delta_time_ = duration::zero();
  duration paused_time_ = duration::zero();

  time_point base_time_;
  time_point stop_time_;
  time_point prev_time_;
  time_point curr_time_;

  bool stopped_ = false;

  duration TotalTime() {
    return (stopped_ ? stop_time_ : curr_time_) - base_time_ - paused_time_;
  };
  duration DeltaTime() { return delta_time_; }

  void Reset() {
    base_time_ = steady_clock::now();
    prev_time_ = base_time_;
    paused_time_ = duration::zero();
    stopped_ = false;
  }

  void Stop() {
    if (!stopped_) {
      stop_time_ = steady_clock::now();
      stopped_ = true;
    }
  }

  void Tick() {
    if (stopped_) {
      delta_time_ = duration::zero();
      return;
    }
    curr_time_ = steady_clock::now();
    delta_time_ = curr_time_ - prev_time_;
    prev_time_ = curr_time_;
    if (delta_time_ < duration::zero()) delta_time_ = duration::zero();
  }
};
};  // namespace Rain