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

#include <stdint.h>
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver {

// The type for the stream_id in an inference request. This must match
// the protobuf type used in InferRequestHeader.stream_id.
using StreamID = uint64_t;

using DimsList = ::google::protobuf::RepeatedField<::google::protobuf::int64>;

// Enumeration for the different platform types
enum Platform {
  PLATFORM_UNKNOWN = 0,
  PLATFORM_TENSORRT_PLAN = 1,
  PLATFORM_TENSORFLOW_GRAPHDEF = 2,
  PLATFORM_TENSORFLOW_SAVEDMODEL = 3,
  PLATFORM_CAFFE2_NETDEF = 4
};

// Get the size of a datatype in bytes. Return 0 if unable to
// determine the size of the data type.
size_t GetDataTypeByteSize(const DataType dtype);

// Get the size of a tensor based on datatype and dimensions. Return 0
// if unable to determine the size of the data type.
uint64_t GetSize(const DataType& dtype, const DimsList& dims);

// Get the size of a tensor based on ModelInput. Return 0 if unable to
// determine the size of the data type.
uint64_t GetSize(const ModelInput& mio);

// Get the size of a tensor based on ModelOutput. Return 0 if unable
// to determine the size of the data type.
uint64_t GetSize(const ModelOutput& mio);

// Get the Platform value for a platform string or Platform::UNKNOWN
// if the platform string is not recognized.
Platform GetPlatform(const std::string& platform_str);

// Get the CPU thread nice level associate with a model configurations
// priority.
int GetPriorityNiceLevel(const ModelConfig& config);

}}  // namespace nvidia::inferenceserver
