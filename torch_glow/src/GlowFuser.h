/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H
#define GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H

#include <torch/csrc/jit/ir.h>

namespace glow {
typedef std::function<bool(torch::jit::Node *)> isSupportFunc;

void GlowCustomFuse(std::shared_ptr<torch::jit::Graph> graph, isSupportFunc fn,
                    at::Symbol kind);
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H
