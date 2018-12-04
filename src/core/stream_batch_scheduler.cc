// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/stream_batch_scheduler.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/infer.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

StreamBatchScheduler::StreamBatchScheduler(
  const ModelConfig& config, uint32_t runner_cnt, StandardRunFunc OnSchedule)
    : OnSchedule_(OnSchedule), scheduler_thread_cnt_(runner_cnt),
      idle_scheduler_thread_cnt_(0)
{
  scheduler_threads_exit_.store(false);

  // Create one scheduler thread for each requested runner. Associate
  // each scheduler thread with a runner.
  const int nice = GetPriorityNiceLevel(config);
  for (uint32_t c = 0; c < scheduler_thread_cnt_; ++c) {
    scheduler_threads_.emplace_back(
      new std::thread([this, c, nice]() { SchedulerThread(c, nice); }));
  }
}

StreamBatchScheduler::~StreamBatchScheduler()
{
  // Signal the scheduler threads to exit and then wait for them...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_threads_exit_.store(true);
    cv_.notify_all();
  }

  for (auto& thd : scheduler_threads_) {
    thd->join();
  }
}

void
StreamBatchScheduler::Enqueue(
  std::shared_ptr<ModelInferStats> stats,
  std::shared_ptr<InferRequestProvider> request_provider,
  std::shared_ptr<InferResponseProvider> response_provider,
  std::function<void(tensorflow::Status)> OnComplete)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  bool wake_runner = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.emplace_back(
      now, stats, request_provider, response_provider, OnComplete);

    // If there are any idle runners then wake one up to service this
    // request. We do the actual wake outside of the lock to avoid
    // having the woken thread immediately block on the lock
    wake_runner = (idle_scheduler_thread_cnt_ > 0);
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

void
StreamBatchScheduler::SchedulerThread(const uint32_t runner_id, const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting stream-batch scheduler thread " << runner_id
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting stream-batch scheduler thread " << runner_id
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // For debugging delay start of threads until the queue contains the
  // specified number of entries.
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying scheduler thread " << runner_id << " until "
             << delay_cnt << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_threads_exit_.load()) {
    std::shared_ptr<std::vector<Scheduler::Payload>> payloads;
    bool wake_thread = false;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        // Debugging... wait until queue contains 'delay_cnt' items...
        wait_microseconds = 10 * 1000;
        if (queue_.size() >= delay_cnt) {
          delay_cnt = 0;
        }
      } else if (queue_.empty()) {
        wait_microseconds = default_wait_microseconds;
      } else {
        // No batching... execute next request payload
        payloads = std::make_shared<std::vector<Scheduler::Payload>>();
        payloads->emplace_back(queue_.front());
        queue_.pop_front();
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        idle_scheduler_thread_cnt_++;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        idle_scheduler_thread_cnt_--;
      }
    }

    if (wake_thread) {
      cv_.notify_one();
    }

    if ((payloads != nullptr) && !payloads->empty()) {
      auto OnCompleteQueuedPayloads = [payloads](tensorflow::Status status) {
        bool found_success = false;
        for (auto& payload : *payloads) {
          tensorflow::Status final_status =
            status.ok() ? (payload.status_.ok() ? payload.compute_status_
                                                : payload.status_)
                        : status;

          // All the payloads executed together, so count 1 execution in
          // the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.ok()) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }
          payload.complete_function_(final_status);
        }
      };

      OnSchedule_(runner_id, payloads.get(), OnCompleteQueuedPayloads);
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping stream-batch scheduler thread " << runner_id
                 << "...";
}

}}  // namespace nvidia::inferenceserver
