#include "worker.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "prefetcher.hpp"

void PredictWorker::do_one_task_impl(BaseTask *_) {
  // TRACE_EVENT_GURAD(kPredict, "predict thread");
  auto prob = predictor->predict().reshape({metas->num_layer, metas->num_expert});
  auto sorted = prob.sort(-1, true);
  // auto predicted_expert_prob = std::get<0>(sorted).slice(1, 0,
  // metas->num_predict_expert_per_layer);
  auto predicted_expert = std::get<1>(sorted).slice(1, 0, std::min<size_t>(metas->num_predict_expert_per_layer, cache->query_per_layer_cache_len()));

  LOG_BLOCK(DEBUG, logger, {
    for (int l = 0; l < metas->num_layer; l++) {
      logger << "predicted expert" << l << ":" << tensor_to_str(predicted_expert[l]);
    }
  });

  {
    TRACE_EVENT_GURAD(kPredict, "add_multi_layer_task");
    size_t per_layer_num_expert = predicted_expert.size(1);
    for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
      {
        TRACE_EVENT_GURAD(kPredict, "wait for budget " + std::to_string(layer_idx));
        while (sem_trywait(&prefetch_layer_budget) == -1) {
          if (should_exit()) { return; }
        }
      }
      prefetcher->add_one_layer_task(layer_idx, predicted_expert[layer_idx].data_ptr<int64_t>(), per_layer_num_expert);
      sem_post(&prefetch_layer_progress);
    }
  }
  predictor->clear_access_buffer();
}
void ExpertUnlockWorker::do_one_task_impl(ExpertHandler *task) {
  CHECK(task != nullptr);
  task->expert_status.wait(kUsing, kUsing);
  CUDA_CALL(cudaEventSynchronize(task->event));
  task->expert_status.transfer(kUsing, kReady);
}
void FetchWorker::do_one_task_impl(CopyTask *task) {
  TRACE_EVENT_GURAD(kFetcher, "do:" + task->toString());
  LOG(TRACE) << "fetcher: copying " << task->toString();
  {
    task->lambda_wait();
  }
  CUDA_CALL(cudaMemcpyAsync(
      task->dst->mem_buffers[task->mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[task->mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[task->mem_buf_idx].len(),
      cudaMemcpyHostToDevice, this->stream));

  // point model parameter to destination
  auto gpu_tensor = task->dst->mem_buffers[task->mem_buf_idx].get_tensor();
  auto expert_param = task->expert->reference_to_model_param.mem_buffers[task->mem_buf_idx].get_tensor();
  expert_param.set_(gpu_tensor, 0, gpu_tensor.sizes(), gpu_tensor.strides());

  CUDA_CALL(cudaStreamSynchronize(this->stream));
  prefetcher->fetch_schedule_thread->copy_done_task.init(task);
  prefetcher->fetch_schedule_thread->add_one_task(&prefetcher->fetch_schedule_thread->copy_done_task);
}
void CopyTask::init(PrefetchTask *task) {
  this->expert      = task->expert;
  this->mem_buf_idx = task->mem_buf_idx;
  this->is_precise  = task->is_precise;
  this->dst = expert->gpu_data;
}
