#include "worker.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "prefetcher.hpp"

void PredictWorker::do_one_task_impl(PredictJob job) {
  TRACE_EVENT_GURAD(kPredictor, "predict thread " + std::to_string(job.input_layer_id));
  const auto & p_m_metas = predictor->predict_model_metas[job.input_layer_id];
  auto prob = predictor->predict(job.input_layer_id);
  LOG_BLOCK(DEBUG, logger, {
    logger << "predict worker: predict " << job.input_layer_id << " " << prob.sizes() << ", slice it with [" << p_m_metas.slice_start << ":" << p_m_metas.slice_stop << "]";
  });
  prob = prob.slice(0, p_m_metas.slice_start, p_m_metas.slice_stop);
  auto num_predicted_layers = prob.size(0);
  CHECK(prob.size(1) == metas->num_expert || prob.size(1) == 0);
  if (metas->cache_policy == "nn" && prob.size(1) > 0) {
    cache->update_priority(prob, p_m_metas.output_layer_start());
  }
  auto sorted = prob.sort(-1, true);
  // auto predicted_expert_prob = std::get<0>(sorted).slice(1, 0,
  // metas->num_predict_expert_per_layer);
  auto per_layer_predict_num_expert_in_cur_iter = std::min<size_t>(metas->num_predict_expert_per_layer, cache->query_per_layer_cache_len());
  per_layer_predict_num_expert_in_cur_iter = std::min<size_t>(per_layer_predict_num_expert_in_cur_iter, prob.size(1));
  auto predicted_expert = std::get<1>(sorted).slice(1, 0, per_layer_predict_num_expert_in_cur_iter);

  LOG_BLOCK(DEBUG, logger, {
    logger << "predicted shape " << predicted_expert.sizes() << "\n";
  });
  LOG_BLOCK(DEBUG, logger, {
    for (int l_in_slice = 0; l_in_slice < num_predicted_layers; l_in_slice++) {
      int layer_idx = p_m_metas.output_layer(l_in_slice);
      logger << "predicted expert" << layer_idx << ":" << tensor_to_str(predicted_expert[l_in_slice]) << "\n";
    }
  });

  {
    TRACE_EVENT_GURAD(kPredictor, "add_multi_layer_task [" + std::to_string(job.input_layer_id) + "," + std::to_string(num_predicted_layers + job.input_layer_id) + ")");
    size_t per_layer_num_expert = predicted_expert.size(1);
    for (int l_in_slice = 0; l_in_slice < num_predicted_layers; l_in_slice++) {
      int layer_idx = p_m_metas.output_layer(l_in_slice);
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
      task.expert_idxs = predicted_expert[l_in_slice].data_ptr<int64_t>();
      task.num_expert = per_layer_num_expert;
      auto wait_handler = fetch_schedule_thread->add_one_task(&task);
      fetch_schedule_thread->wait_progress(wait_handler);
      // fetch_schedule_thread->add_one_layer_task(layer_idx, predicted_expert[layer_idx].data_ptr<int64_t>(), per_layer_num_expert);
      sem_post(&prefetch_layer_progress);
    }
  }
  if (num_predicted_layers + job.input_layer_id == metas->num_layer) {
    predictor->end_of_one_token_prediction();
  }
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
    task->expert->reference_to_model_param.mem_buffers[mem_buf_idx].map_to(task->expert->gpu_data->mem_buffers[mem_buf_idx]);
    CUDA_CALL(cudaMemcpyAsync(
      task->expert->reference_to_model_param.mem_buffers[mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[mem_buf_idx].len(),
      cudaMemcpyHostToDevice, this->stream));
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
    case kOneToken:                { add_one_task(PredictJob()); break; }
    case kDecodeCumsum:            { add_one_task(PredictJob()); break; }
    case kLastUseDistance:         { add_one_task(PredictJob()); break; }
    case kWeighedDecodeCumsum:     { add_one_task(PredictJob()); break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    case kMoeLayerLogits:          { break; }
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
      if (layer_id == 0) { add_one_task(PredictJob()); }
      break;
    }
    case kMoeAttnInputLogits:      {
      if (predictor->layer_predict_enabled[layer_id]) {
      // if (layer_id % metas->layer_predict_interval == 0) {
        add_one_task(PredictJob(layer_id));
      }
      break;
    }
    case kMoeLayerLogits:          { break; }
    default: { CHECK(false) << "Unknown predict input mode"; }
  }
}
void PredictWorker::on_moe_layer_logits_recorded(int layer_id) {
  LOG(DEBUG) << "predict workers, on_moe_layer_logits_recorded " << layer_id;
  switch (metas->predict_input_mode) {
    case kOneToken:                { break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: { break;}
    case kMoeAttnInputLogits:      { break;}
    case kMoeLayerLogits:          {
      if (predictor->layer_predict_enabled[layer_id]) {
        add_one_task(PredictJob(layer_id));
      }
      break;
    }
    default: { CHECK(false) << "Unknown predict input mode in on_moe_layer_logits_recorded"; }
  }
}
