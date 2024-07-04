#include "worker.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "prefetcher.hpp"

void PredictWorker::do_one_task_impl() {
  // TRACE_EVENT_GURAD(kPredict, "predict thread");
  auto prob = predictor->predict().reshape({metas->num_layer, -1});
  CHECK(prob.size(1) == metas->num_expert || prob.size(1) == 0);
  if (metas->cache_policy == "nn" && prob.size(1) > 0) {
    cache->update_all_priority(prob);
  }
  auto sorted = prob.sort(-1, true);
  // auto predicted_expert_prob = std::get<0>(sorted).slice(1, 0,
  // metas->num_predict_expert_per_layer);
  auto per_layer_predict_num_expert_in_cur_iter = std::min<size_t>(metas->num_predict_expert_per_layer, cache->query_per_layer_cache_len());
  per_layer_predict_num_expert_in_cur_iter = std::min<size_t>(per_layer_predict_num_expert_in_cur_iter, prob.size(1));
  auto predicted_expert = std::get<1>(sorted).slice(1, 0, per_layer_predict_num_expert_in_cur_iter);

  LOG_BLOCK(DEBUG, logger, {
    for (int l = 0; l < metas->num_layer; l++) {
      logger << "predicted expert" << l << ":" << tensor_to_str(predicted_expert[l]);
    }
  });

  {
    TRACE_EVENT_GURAD(kPredictor, "add_multi_layer_task");
    size_t per_layer_num_expert = predicted_expert.size(1);
    for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
      {
        LOG(DEBUG) << "predict worker: add layer task " << layer_idx << ", wait for budget";
        TRACE_EVENT_GURAD(kPredictor, "wait for budget " + std::to_string(layer_idx));
        while (sem_trywait(&prefetch_layer_budget) == -1) {
          if (should_exit()) { return; }
        }
      }
      LOG(DEBUG) << "predict worker: add layer task now " << layer_idx;
      PrefetchLayerTask task;
      task.layer_idx = layer_idx;
      task.expert_idxs = predicted_expert[layer_idx].data_ptr<int64_t>();
      task.num_expert = per_layer_num_expert;
      auto wait_handler = fetch_schedule_thread->add_one_task(&task);
      fetch_schedule_thread->wait_progress(wait_handler);
      // fetch_schedule_thread->add_one_layer_task(layer_idx, predicted_expert[layer_idx].data_ptr<int64_t>(), per_layer_num_expert);
      sem_post(&prefetch_layer_progress);
    }
  }
  predictor->end_of_one_token_prediction();
  // if (metas->predict_input_mode == kOneToken) {
  //   predictor->clear_access_buffer();
  // }
}
void ExpertUnlockWorker::do_one_task_impl(ExpertHandler *task) {
  TRACE_EVENT_GURAD(kUnlocker, "unlock:" + task->toString());
  task->expert_status.wait(kUsing);
  CUDA_CALL(cudaEventSynchronize(task->event));
  task->expert_status.transfer(kUsing, kReady);
}
void FetchWorker::do_one_task_impl(CopyTask *task) {
  TRACE_EVENT_GURAD(kFetcher, "fetch:" + task->toString());
  LOG(TRACE) << "fetcher: copying " << task->toString();
  {
    task->lambda_wait();
  }
  for (int mem_buf_idx = task->start_mem_buf_idx; mem_buf_idx < task->stop_mem_buf_idx; mem_buf_idx++) {
    CUDA_CALL(cudaMemcpyAsync(
      task->expert->gpu_data->mem_buffers[mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[mem_buf_idx].len(),
      cudaMemcpyHostToDevice, this->stream));

    // point model parameter to destination
    auto gpu_tensor = task->expert->gpu_data->mem_buffers[mem_buf_idx].get_tensor();
    auto expert_param = task->expert->reference_to_model_param.mem_buffers[mem_buf_idx].get_tensor();
    expert_param.set_(gpu_tensor, 0, gpu_tensor.sizes(), gpu_tensor.strides());
  }

  CUDA_CALL(cudaStreamSynchronize(this->stream));
  fetch_schedule_thread->add_one_task(&fetch_schedule_thread->copy_done_task);
}
void PredictWorker::add_prefetch_layer_budget() {
  LOG(DEBUG) << "predict worker: add prefetch layer budget";
  sem_post(&prefetch_layer_budget);
}
void PredictWorker::on_one_iter_done() {
  LOG(DEBUG) << "predict workers, one iter done";
  switch (metas->predict_input_mode) {
    case kOneToken:                { add_one_task(); break; }
    case kDecodeCumsum:            { add_one_task(); break; }
    case kLastUseDistance:         { add_one_task(); break; }
    case kWeighedDecodeCumsum:     { add_one_task(); break; }
    case kFirstMoeAttnInputLogits: { break; }
    default: { CHECK(false) << "Unknown predict input mode"; }
  }
}
void PredictWorker::on_moe_attn_input_logits_recorded(int layer_id) {
  LOG(DEBUG) << "predict workers, on_moe_attn_input_logits_recorded " << layer_id;
  switch (metas->predict_input_mode) {
    case kOneToken:                { break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: {
      if (layer_id == 0) { add_one_task(); }
      break;
    }
    default: { CHECK(false) << "Unknown predict input mode"; }
  }
}
