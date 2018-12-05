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
#pragma once

#include <sys/time.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

// Scheduler that implements batching for stream inferences.
class StreamBatchScheduler : public Scheduler {
 public:
  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  StreamBatchScheduler(
    const ModelConfig& config, const uint32_t runner_cnt,
    StandardRunFunc OnSchedule);
  ~StreamBatchScheduler();

  // \see Scheduler::Enqueue()
  void Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete) override;

 private:
  // Scheduler payload for each request.
  struct StreamPayload : public Scheduler::Payload {
    StreamPayload() = default;
    StreamPayload(const StreamPayload& payload) = default;
    StreamPayload(
      const struct timespec queued_timestamp,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      const std::function<void(tensorflow::Status)> complete_function)
        : Payload(
            queued_timestamp, stats, request_provider, response_provider,
            complete_function)
    {
    }
  };

  // Queued requests for a model instance that will be sent through
  // that instance together in a batch.
  class StreamBatch {
   public:
    StreamBatch(uint32_t max_batch_size);

    // Return the index within the batch that has no queued
    // requests. If there are multiple such indices return the lowest
    // numbered one.
    bool GetFreeSlot(uint32_t* slot);

    void Enqueue(
      const uint32_t slot, const struct timespec queue_timestamp,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(tensorflow::Status)> OnComplete);

   private:
    // The maximum batch size requests can be grouped into (model
    // must support at least this size).
    const uint32_t max_batch_size_;

    // Mutex protecting queues.
    std::mutex mu_;
    std::condition_variable cv_;

    // Queues holding inference requests. There are 'max_batch_size_'
    // queues, one for each batch "slot" where requests assigned to
    // that slot are enqueued to wait for inferencing.
    std::vector<std::deque<StreamPayload>> queues_;
  };

 private:
  void SchedulerThread(
    std::shared_ptr<StreamBatch> sb, const uint32_t runner_id, const int nice);

  // Function the scheduler will call to schedule a payload(s) for
  // execution.
  const StandardRunFunc OnSchedule_;

  // The number of scheduler threads.
  const uint32_t scheduler_thread_cnt_;

  // The maximum size of the batch to create for each runner.
  const uint32_t max_batch_size_;

  // The number of scheduler threads currently idle.
  uint32_t idle_scheduler_thread_cnt_;

  std::vector<std::unique_ptr<std::thread>> scheduler_threads_;
  std::atomic<bool> scheduler_threads_exit_;

  // The StreamBatch for each scheduler thread.
  std::vector<std::shared_ptr<StreamBatch>> batches_;

  // The target location for requests for a given stream ID. The
  // target is either a StreamBatch or a backlog queue.
  struct StreamTarget {
    // Return true if this target is a backlog queue, false if this
    // target is a StreamBatch_slot.
    bool IsBacklog() const { return stream_batch_ == nullptr; }

    // If 'stream_batch_' is non-null then the target is 'slot_'
    // within 'stream_batch_'.
    std::shared_ptr<StreamBatch> stream_batch_;
    uint32_t slot_;

    // If 'stream_batch_' is null then the target is a backlog queue.
    std::deque<StreamPayload> backlog_;
  };

  // Map from a request's stream ID to the StreamBatch+slot or backlog
  // queue assigned to that stream ID.
  using StreamTargetMap = std::unordered_map<StreamID, StreamTarget>;
  StreamTargetMap stream_to_target_map_;

  // Ordered list of stream IDs in the backlog. When a slot becomes
  // available the first item from the backlog, if any, is used to
  // fill that slot.
  std::deque<StreamID> backlog_stream_ids_;

  // Mutex/cv for communicating with scheduling threads and for
  // protecting stream IDs -> StreamBatch map.
  std::mutex mu_;
  std::condition_variable cv_;
};

}}  // namespace nvidia::inferenceserver
