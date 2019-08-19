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

#include "GlowFuser.h"

#include <llvm/Support/raw_ostream.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace glow {

// This is mainly copied from pytorch/tvm
// This pass fuse the addmm or matmul + add generated by JIT back to linear
// to allow direct support with Glow integration with Glow IR
// This pass can be deleted once the JIT can emit the aten::linear in the future
void FuseLinear(std::shared_ptr<torch::jit::Graph> &graph) {
  std::string addmm_pattern = R"IR(
graph(%input, %weight, %bias, %4):
  %weight_t = aten::t(%weight)
  %res = aten::addmm(%bias, %input, %weight_t, %4, %4)
  return (%res))IR";
  std::string matmul_add_pattern = R"IR(
graph(%input, %weight, %bias, %4):
  %weight_t = aten::t(%weight)
  %output = aten::matmul(%input, %weight_t)
  %res = aten::add_(%output, %bias, %4)
  return (%res))IR";
  std::string mm_add_pattern = R"IR(
graph(%input, %weight, %bias, %4):
  %weight_t = aten::t(%weight)
  %output = aten::mm(%input, %weight_t)
  %res = aten::add_(%output, %bias, %4)
  return (%res))IR";
  std::string fused_linear = R"IR(
graph(%input, %weight, %bias, %4):
  %res = aten::linear(%input, %weight, %bias)
  return (%res))IR";

  std::string matmul_pattern = R"IR(
graph(%input, %weight):
  %weight_t = aten::t(%weight)
  %output = aten::matmul(%input, %weight_t)
  return (%output))IR";
  std::string mm_pattern = R"IR(
graph(%input, %weight):
  %weight_t = aten::t(%weight)
  %output = aten::mm(%input, %weight_t)
  return (%output))IR";
  std::string fused_linear_bias_none = R"IR(
graph(%input, %weight):
  %bias: Tensor? = prim::Constant()
  %res = aten::linear(%input, %weight, %bias)
  return (%res))IR";

  // replace addmm pattern to linear
  torch::jit::SubgraphRewriter addmm_to_linear;
  addmm_to_linear.RegisterRewritePattern(addmm_pattern, fused_linear);
  addmm_to_linear.runOnGraph(graph);

  // replace matmul + add pattern to linear
  torch::jit::SubgraphRewriter matmuladd_to_linear;
  matmuladd_to_linear.RegisterRewritePattern(matmul_add_pattern, fused_linear);
  matmuladd_to_linear.runOnGraph(graph);

  // replace mm + add pattern to linear
  torch::jit::SubgraphRewriter mmadd_to_linear;
  mmadd_to_linear.RegisterRewritePattern(mm_add_pattern, fused_linear);
  mmadd_to_linear.runOnGraph(graph);

  // replace matmul with bias=None pattern to linear
  torch::jit::SubgraphRewriter matmul_to_linear;
  matmul_to_linear.RegisterRewritePattern(matmul_pattern,
                                          fused_linear_bias_none);
  matmul_to_linear.runOnGraph(graph);

  // replace mm with bias=None pattern to linear
  torch::jit::SubgraphRewriter mm_to_linear;
  mm_to_linear.RegisterRewritePattern(mm_pattern, fused_linear_bias_none);
  mm_to_linear.runOnGraph(graph);
}

torch::jit::value_list
sortReverseTopological(at::ArrayRef<torch::jit::Value *> inputs,
                       torch::jit::Block *block) {
  torch::jit::value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }

  std::sort(result.begin(), result.end(),
            [&](torch::jit::Value *a, torch::jit::Value *b) {
              return a->node()->isAfter(b->node());
            });
  return result;
}

bool canMerge(torch::jit::Node *node, isSupportFunc fn) {
  return node->kind() == torch::jit::prim::Constant || fn(node);
}

bool canMerge(torch::jit::Block *block, isSupportFunc fn) {
  for (torch::jit::Node *node : block->nodes()) {
    if (!canMerge(node, fn)) {
      return false;
    }
  }
  return true;
}

#define REQ(cond, log_info)                                                    \
  if (!(cond)) {                                                               \
    llvm::errs() << log_info;                                                  \
    return c10::nullopt;                                                       \
  }

c10::optional<torch::jit::Node *> tryMerge(torch::jit::Node *consumer,
                                           torch::jit::Node *producer,
                                           torch::jit::AliasDb &aliasDb,
                                           isSupportFunc fn, at::Symbol kind) {

  std::string symbol_name_producer = producer->kind().toQualString();
  std::string symbol_name_consumer = consumer->kind().toQualString();
  REQ(canMerge(producer, fn),
      "Detected unknown node: " + symbol_name_producer + ".\n")
  REQ(consumer->kind() == kind || canMerge(consumer, fn),
      "Detected unknown node: " + symbol_name_consumer + ".\n")

  // Alias checks
  // Requirement:
  // - moveAfterTopologicallyValid(consumer, producer)
  // - One of:
  //   1) Both are in-place ops
  //   2) Consumer is in-place, producer !hasInputWriters
  //   3) Producer is in-place, consumer !hasOutputWriters
  REQ(aliasDb.moveAfterTopologicallyValid(consumer, producer),
      "Unable to move after topologically valid.");

  // 1)
  if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer))) {
    // 2)
    if (aliasDb.isMutable(consumer)) {
      REQ(!aliasDb.hasInputWriters(producer),
          "Producer does not have input writer when merging.");
      // 3)
    } else if (aliasDb.isMutable(producer)) {
      REQ(!aliasDb.hasOutputWriters(consumer),
          "Consumer does not have output writer when merging.");
    }
  }

  if (!consumer->hasAttribute(torch::jit::attr::Subgraph) &&
      consumer->kind() != kind) {
    consumer =
        torch::jit::SubgraphUtils::createSingletonSubgraph(consumer, kind);
  }
  if (producer->kind() == torch::jit::prim::Constant) {
    auto &subgraph = consumer->g(torch::jit::attr::Subgraph);
    torch::jit::Node *in_const = subgraph->createClone(
        producer, [](torch::jit::Value *) -> torch::jit::Value * {
          throw std::runtime_error("unexpected input");
        });
    subgraph->insertNode(in_const);
  } else {
    torch::jit::SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }
  return consumer;
}
#undef REQ

std::pair<torch::jit::graph_node_list::iterator, bool>
getNewNode(torch::jit::Node *node, torch::jit::AliasDb &aliasDb,
           torch::jit::Block *block, isSupportFunc fn, at::Symbol kind) {
  auto node_inputs = sortReverseTopological(node->inputs(), block);
  for (auto input : node_inputs) {
    if (auto group = tryMerge(node, input->node(), aliasDb, fn, kind)) {
      return {group.value()->reverseIterator(), true};
    }
  }
  return {++node->reverseIterator(), false};
}

void GlowCustomFuse(std::shared_ptr<torch::jit::Graph> graph, isSupportFunc fn,
                    at::Symbol kind) {
  torch::jit::AliasDb aliasDb(graph);
  auto block = graph->block();

  bool is_changed;
  do {
    is_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool is_changed_thisnode;
      std::tie(it, is_changed_thisnode) =
          getNewNode(*it, aliasDb, block, fn, kind);
      is_changed |= is_changed_thisnode;
    }
  } while (is_changed);
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}

} // namespace glow
