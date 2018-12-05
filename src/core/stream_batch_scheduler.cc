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
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/infer.h"
#include "src/core/logging.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

StreamBatchScheduler::StreamBatchScheduler(
  const ModelConfig& config, const uint32_t runner_cnt,
  StandardRunFunc OnSchedule)
    : OnSchedule_(OnSchedule), scheduler_thread_cnt_(runner_cnt),
      max_batch_size_(config.max_batch_size()), idle_scheduler_thread_cnt_(0)
{
  scheduler_threads_exit_.store(false);

  // Create one scheduler thread for each requested runner. Associate
  // each scheduler thread with a runner. Also create and associate a
  // StreamBatch object with each thread that manages the queues of
  // requests for that thread.
  const int nice = GetPriorityNiceLevel(config);
  for (uint32_t c = 0; c < scheduler_thread_cnt_; ++c) {
    std::shared_ptr<StreamBatch> sb =
      std::make_shared<StreamBatch>(max_batch_size_);
    batches_.push_back(sb);
    scheduler_threads_.emplace_back(
      new std::thread([this, sb, c, nice]() { SchedulerThread(sb, c, nice); }));
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
  const std::shared_ptr<ModelInferStats>& stats,
  const std::shared_ptr<InferRequestProvider>& request_provider,
  const std::shared_ptr<InferResponseProvider>& response_provider,
  std::function<void(tensorflow::Status)> OnComplete)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  const auto& request_header = request_provider->RequestHeader();
  const StreamID stream_id = request_header.stream_id();

  // A request must have a stream ID to be processed correctly by this
  // scheduler. A value of 0 (zero) indicates that the request doesn't
  // have a stream ID.
  if (stream_id == 0) {
    OnComplete(tensorflow::errors::InvalidArgument(
      "inference request to model '", request_provider->ModelName(),
      "' must specify a stream ID"));
    return;
  }

  StreamTarget* target = nullptr;

  std::unique_lock<std::mutex> lock(mu_);

  // If the request's stream_id is new, then attempt to find a free
  // slot to use for that stream. If one doesn't exist then put the
  // request onto the backlog queue where it must wait for a slot to
  // come free. If a free slot is found assign this and subsequent
  // requests in this stream to that StreamBatch+slot.
  auto sb_itr = stream_to_target_map_.find(stream_id);
  if (sb_itr == stream_to_target_map_.end()) {
    bool found_slot = false;
    std::shared_ptr<StreamBatch> isb;
    uint32_t islot;
    for (const std::shared_ptr<StreamBatch>& bsb : batches_) {
      found_slot = bsb->GetFreeSlot(&islot);
      if (found_slot) {
        isb = bsb;
        break;
      }
    }

    target = &stream_to_target_map_[stream_id];
    if (found_slot) {
      target->stream_batch_ = isb;
      target->slot_ = islot;
    } else {
      backlog_stream_ids_.push_back(stream_id);
    }
  } else {
    target = &sb_itr->second;
  }

  if (target->IsBacklog()) {
    target->backlog_.emplace_back(
      now, stats, request_provider, response_provider, OnComplete);
    return;
  }

  std::shared_ptr<StreamBatch> sb = target->stream_batch_;
  uint32_t slot = target->slot_;

  lock.unlock();

  sb->Enqueue(
    slot, now, stats, request_provider, response_provider, OnComplete);
}

void
StreamBatchScheduler::SchedulerThread(
  std::shared_ptr<StreamBatch> sb, const uint32_t runner_id, const int nice)
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
    uint64_t wait_microseconds = 0;

// Hold the lock for as short a time as possible.
#if 0
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
#endif

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

StreamBatchScheduler::StreamBatch::StreamBatch(uint32_t max_batch_size)
    : max_batch_size_(max_batch_size), queues_(max_batch_size)
{
}

bool
StreamBatchScheduler::StreamBatch::GetFreeSlot(uint32_t* slot)
{
  std::unique_lock<std::mutex> lock(mu_);

  // FIXME, empty queue doesn't mean it is free!!!

  for (size_t i = 0; i < queues_.size(); ++i) {
    if (queues_[i].empty()) {
      *slot = i;
      return true;
    }
  }

  return false;
}

}}  // namespace nvidia::inferenceserver
