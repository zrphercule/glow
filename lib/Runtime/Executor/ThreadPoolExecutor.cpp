/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "ExecutionState.h"

#include "glow/Backends/DeviceManager.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"

#include <queue>
#include <unordered_set>

#include "llvm/Support/FormatVariadic.h"
#include <glog/logging.h>

namespace glow {
namespace runtime {

void InflightBarrier::decrement(unsigned decr) {
  std::unique_lock<std::mutex> lock(mtx_);
  DCHECK_GE(count_, decr) << "Barrier decrement cannot be less than count!";
  count_ -= decr;

  // If count_ has hit zero, wake up all threads that are waiting.
  if (count_ == 0) {
    cv_.notify_all();
  }
} // namespace runtime

void InflightBarrier::increment(unsigned incr) {
  std::unique_lock<std::mutex> lock(mtx_);
  count_ += incr;
}

unsigned InflightBarrier::count() {
  std::unique_lock<std::mutex> lock(mtx_);
  return count_;
}

void InflightBarrier::wait() {
  std::unique_lock<std::mutex> lock(mtx_);
  // If count_ is not 0, wait until a signal is received that it is.
  // The second argument below is a predicate that returns true when
  // it is safe to wake up. It preserves correctness in the case of
  // spurious wakeups.
  cv_.wait(lock, [&] { return count_ == 0; });
}

ThreadPoolExecutor::ThreadPoolExecutor(const DeviceManagerMapTy &deviceManagers,
                                       unsigned numWorkers)
    : threadPool_(numWorkers), deviceManagers_(deviceManagers) {}

void ThreadPoolExecutor::shutdown() {
  // Prevent more requests from being processed.
  shuttingDown_ = true;

  // Wait for all inflight DeviceManager::runFunction() calls to return and be
  // processed before starting to destroy state that is used in
  // handleDeviceManagerResult().
  inflightBarrier_.wait();
}

void ThreadPoolExecutor::run(const DAGNode *root,
                             std::unique_ptr<ExecutionContext> context,
                             RunIdentifierTy runId, ResultCBTy cb) {
  DCHECK(cb != nullptr);

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::run");

  if (context->getTraceContext()) {
    for (auto id : threadPool_.getThreadIds()) {
      context->getTraceContext()->setThreadName(id, "ThreadPoolExecutor");
    }
  }

  // Don't process new requests if the executor is shutting down.
  if (shuttingDown_) {
    cb(runId,
       MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_REQUEST_REFUSED,
                "ThreadPoolExecutor is shutting down"),
       std::move(context));
    return;
  }

  // If list of roots is empty, there is nothing to do. Give back the
  // bindings so the caller can reuse it.
  if (!root) {
    cb(runId, Error::success(), std::move(context));
    return;
  }

  auto numChildren = (root->children).size();
  // Mark the child nodes as "inflight" (i.e. currently executing). This must
  // be done here instead of inside executeDAGNode() so that a node can be
  // executed while placeholders are being propagated for the next node
  // without the callback for that node deleting the execution state.
  inflightBarrier_.increment(numChildren);
  // Get and bind state.
  auto currentState = states_[root]->getNextNetworkExecutionState();
  currentState->bind(std::move(context), std::move(cb), runId);

  currentState->incrementInflightNodes(numChildren);
  for (auto const &node : root->children) {
    // Run with cached state
    executeDAGNode(currentState, node);
  }
}

void ThreadPoolExecutor::executeDAGNode(NetworkExecutionState *executionState,
                                        DAGNode *node) {
  TRACE_EVENT_SCOPE(executionState->getRawResultContextPtr()->getTraceContext(),
                    TraceLevel::RUNTIME, "ThreadPoolExecutor::executeDAGNode");
  if (executionState->getErrorContainer().containsErr()) {
    // Mark the node as no longer executing.
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }

  // Get the PlaceholderBindings containing all of the inputs for the node.
  std::unique_ptr<ExecutionContext> nodeCtx =
      executionState->getUniqueNodeContextPtr(node);
  auto deviceManagerIt = deviceManagers_.find(node->getNextDevice());
  if (deviceManagerIt == deviceManagers_.end()) {
    // Mark the node as no longer executing.
    executionState->getErrorContainer().set(
        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                 "Cannot find the DeviceManager specified."));
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }
  DeviceManager *deviceManager = deviceManagerIt->second.get();
  // Run the node using the DeviceManager.
  deviceManager->runFunction(
      node->name, std::move(nodeCtx),
      [this, executionState,
       node](RunIdentifierTy id, Error err,
             std::unique_ptr<ExecutionContext> resultCtx) {
        // Immediately move the handling of the result onto this run's executor
        // to avoid doing work on the DeviceManager thread.
        threadPool_.getExecutor()->submit(
            [this, executionState, node, err = std::move(err),
             ctx = std::move(resultCtx)]() mutable {
              this->handleDeviceManagerResult(executionState, std::move(err),
                                              std::move(ctx), node);
            });
      });
}

void ThreadPoolExecutor::handleDeviceManagerResult(
    NetworkExecutionState *executionState, Error err,
    std::unique_ptr<ExecutionContext> ctx, const DAGNode *node) {

  TraceContext *traceContext = ctx->getTraceContext();
  if (traceContext) {
    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                      "ThreadPoolExecutor::handleResult");
  }

  auto runWasSuccess = !err;

  // Set the result code for the run.
  executionState->getErrorContainer().set(std::move(err));

  // If the DeviceManager executed the node, propagate its output Placeholders
  // to its children or the result PlaceholderBindings as appropriate.
  if (runWasSuccess) {
    for (auto &child : node->children) {
      // Execute any child that has no parent nodes left to execute.
      bool childReadyToExecute =
          executionState->incrementNodeParentsDone(child);
      if (childReadyToExecute) {
        // Mark the node as "inflight" (i.e. currently executing).
        executionState->incrementInflightNodes();
        inflightBarrier_.increment();
        executeDAGNode(executionState, child);
      }
    }
  }
  // Return intermediateContext to executionState.
  executionState->returnUniqueNodeContextPtr(node, std::move(ctx));

  // Now, check if all nodes in the graph are done. If so, the callback can be
  // called and all state associated with the run can be erased.
  bool noNodesInflight = executionState->decrementInflightNodes();

  if (traceContext) {
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::handleResult");
    // Lock is not necessary as we only access on this runs executor.
    executionState->insertIntoTraceContext(traceContext);
  }

  if (noNodesInflight) {
    // If there are no nodes inflight, that means all nodes are done. Transfer
    // the outpus. Call the callback and erase the state information.
    // Because we are redirecting inputs and outputs to use the provided tensor
    // we do not have to transfer outputs here. Once we have pinned memory we
    // will transfer. //executionState->transferOutputs();
    ResultCBTy cb = executionState->getCallback();
    DCHECK(cb != nullptr);
    cb(executionState->getRunId(), executionState->getErrorContainer().get(),
       executionState->getUniqueResultContextPtr());
  }

  // Decrement the inflight barrier for the executor keeping track of all
  // outstanding DeviceManager::runFunction() calls. This must be done here
  // instead of right after executionState->decrementInflightNodes() so that
  // ~ThreadPoolExecutor does not delete executor state before this function
  // is done using it (e.g. when erasing the ExecutionState object for a
  // run).
  inflightBarrier_.decrement();
}

void ThreadPoolExecutor::createPool(const DAGNode *root, unsigned poolSize) {
  std::unique_ptr<NetworkExecutionStatePool> pool =
      glow::make_unique<NetworkExecutionStatePool>();
  for (unsigned i = 0; i < poolSize; i++) {
    auto newState = glow::make_unique<NetworkExecutionState>(root);
    newState->init(deviceManagers_);
    pool->addNewState(std::move(newState));
  }
  states_[root] = std::move(pool);
}

void ThreadPoolExecutor::freePool(const DAGNode *root) { states_.erase(root); }

} // namespace runtime
} // namespace glow
