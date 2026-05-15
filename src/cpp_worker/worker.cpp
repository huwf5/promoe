#include "worker.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "prefetcher.hpp"
#include "nvtx_utils.hpp"

void PredictWorker::do_one_task_impl(PredictJob job) {
  const int64_t current_generate_epoch = this->current_generate_epoch.load(std::memory_order_acquire);
  CHECK(!(job.generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << job.generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kPredictor, "predict thread " + std::to_string(job.input_layer_id));
  NVTX_RANGE("predict/thread L" + std::to_string(job.input_layer_id));

  int num_predict_jobs = predictor->query_predict_jobs(job.input_layer_id);

  // auto pred_result = predictor->predict(job.input_layer_id);
  for (int job_idx = 0; job_idx < num_predict_jobs; job_idx++) {
    auto pred_result = predictor->predict_one_job(job.input_layer_id, job_idx);

    predictor->slice_predict_output_layer(pred_result);

    auto num_predicted_layers = pred_result.num_output_layer();
    CHECK(pred_result.num_output_expert() == metas->num_expert || pred_result.num_output_expert() == 0);
    if (metas->cache_policy == "nn" && pred_result.num_output_expert() > 0) {
      cache->update_priority(pred_result.prob, pred_result.start_output_layer_id);
    }
    pred_result.rank_experts(metas->num_predict_expert_per_layer);

    LOG_BLOCK(DEBUG, logger, {
      logger << "predicted shape " << pred_result.experts.sizes() << "\n";
    });
    LOG_BLOCK(DEBUG, logger, {
      for (int l_in_slice = 0; l_in_slice < num_predicted_layers; l_in_slice++) {
        int layer_idx = pred_result.start_output_layer_id + l_in_slice;
        logger << "predicted expert" << layer_idx << ":" << tensor_to_str(pred_result.experts[l_in_slice]) << "\n";
      }
    });

    {
      TRACE_EVENT_GURAD(kPredictor, "add_multi_layer_task [" + std::to_string(pred_result.inner_l_to_outer_l(0)) + "," + std::to_string(pred_result.inner_l_to_outer_l(num_predicted_layers)) + ")");
      size_t per_layer_num_expert = pred_result.num_top_experts();
      for (int inner_l = 0; inner_l < num_predicted_layers; inner_l++) {
        int layer_idx = pred_result.inner_l_to_outer_l(inner_l);
        precision_profiler->record_predicted_experts(layer_idx, pred_result.top_experts(inner_l), per_layer_num_expert);
      }
      for (int inner_l = 0; inner_l < num_predicted_layers; inner_l++) {
        int layer_idx = pred_result.inner_l_to_outer_l(inner_l);
        {
          LOG(INFO) << "predict worker: add layer task " << layer_idx << ", wait for budget";
          TRACE_EVENT_GURAD(kPredictor, "wait for budget " + std::to_string(layer_idx));
          while (true) {
            int budge_remaining = prefetch_layer_budget.try_pop(true);
            if (budge_remaining != -1) {
              LOG(INFO) << "predict worker: add layer task now " << layer_idx << ", wait for budget done, remaining " << budge_remaining;
              break;
            }
            if (should_exit()) { return; }
          }
        }
        LOG(INFO) << "predict worker: add layer task now " << layer_idx;
        PrefetchLayerTask task;
        task.layer_idx   = layer_idx;
        task.forward_epoch = job.forward_epoch;
        task.generate_epoch = job.generate_epoch;
        task.expert_idxs = pred_result.top_experts(inner_l);
        if (layer_idx == 0 && metas->limit_layer_0_num_predict != -1) {
          task.num_expert = std::min<int>(per_layer_num_expert, metas->limit_layer_0_num_predict);
        } else {
          task.num_expert  = per_layer_num_expert;
        }
        std::string nvtx_range_name = "predict/submit_prefetch_layer inL" + std::to_string(job.input_layer_id) +
                                      " outL" + std::to_string(layer_idx) +
                                      " N" + std::to_string(task.num_expert) +
                                      " forward_epoch=" + std::to_string(job.forward_epoch);
        if (NvtxDetailEnabled()) {
          nvtx_range_name += " experts=[" + array_to_str(task.expert_idxs, task.num_expert) + "]";
        }
        {
          NVTX_RANGE(nvtx_range_name);
          auto wait_handler = fetch_schedule_thread->add_one_task(&task);
          fetch_schedule_thread->wait_progress(wait_handler);
          // fetch_schedule_thread->add_one_layer_task(layer_idx, predicted_expert[layer_idx].data_ptr<int64_t>(), per_layer_num_expert);
          prefetch_layer_progress.push(layer_idx);
        }
      }
    }
    if (pred_result.inner_l_to_outer_l(num_predicted_layers) == metas->num_layer) {
      predictor->end_of_one_token_prediction();
    }
  }
  // if (metas->predict_input_mode == kOneToken) {
  //   predictor->clear_access_buffer();
  // }
}
void ExpertUnlockWorker::do_one_task_impl(ExpertHandler *task) {
  TRACE_EVENT_GURAD(kUnlocker, "unlock:" + task->toString());
  NVTX_RANGE("unlock/wait_compute L" + std::to_string(task->layer_idx) +
             " E" + std::to_string(task->expert_idx));
  task->expert_status.wait(kUsing);
  CUDA_CALL(cudaEventSynchronize(task->event));
  task->expert_status.transfer(kUsing, kReady);
}
void FetchWorker::do_one_task_impl(CopyTask *task) {
  TRACE_EVENT_GURAD(kFetcher, "fetch:" + task->toString());
  NVTX_RANGE("fetch/task L" + std::to_string(task->expert->layer_idx) +
             " E" + std::to_string(task->expert->expert_idx) +
             " P" + std::to_string(task->start_mem_buf_idx) +
             "-" + std::to_string(task->stop_mem_buf_idx) +
             (task->is_precise ? " precise" : " prefetch"));
  LOG(TRACE) << "fetcher: copying " << task->toString();
  {
    NVTX_RANGE("fetch/wait_evict L" + std::to_string(task->expert->layer_idx) +
               " E" + std::to_string(task->expert->expert_idx));
    task->lambda_wait();
  }
  {
    size_t total_nbytes = 0;
    for (int mem_buf_idx = task->start_mem_buf_idx; mem_buf_idx < task->stop_mem_buf_idx; mem_buf_idx++) {
      total_nbytes += task->expert->host_data->nbytes(mem_buf_idx);
    }
    NVTX_RANGE(std::string(task->is_precise ? "fetch/demand_h2d " : "fetch/prefetch_h2d ") +
               "L" + std::to_string(task->expert->layer_idx) +
               " E" + std::to_string(task->expert->expert_idx) +
               " P" + std::to_string(task->start_mem_buf_idx) +
               "-" + std::to_string(task->stop_mem_buf_idx) +
               " chunks=" + std::to_string(task->stop_mem_buf_idx - task->start_mem_buf_idx) +
               " bytes=" + std::to_string(total_nbytes));
    for (int mem_buf_idx = task->start_mem_buf_idx; mem_buf_idx < task->stop_mem_buf_idx; mem_buf_idx++) {
      // LOG(ERROR) << "fetcher: copy from " << task->expert->host_data.ptr(mem_buf_idx) << " to " << task->expert->gpu_data->ptr(mem_buf_idx);
      CUDA_CALL(cudaMemcpyAsync(
        task->expert->gpu_data->ptr(mem_buf_idx),
        task->expert->host_data->ptr(mem_buf_idx),
        task->expert->host_data->nbytes(mem_buf_idx),
        cudaMemcpyHostToDevice, this->stream));
    }
  }
  if (task->start_mem_buf_idx == 0) {
    NVTX_RANGE("fetch/remap L" + std::to_string(task->expert->layer_idx) +
               " E" + std::to_string(task->expert->expert_idx));
    task->expert->reference_to_model_param->unmap();
    task->expert->reference_to_model_param->map_to(task->expert->gpu_data, mem_mngr_ctx);
  }

  {
    NVTX_RANGE("fetch/sync L" + std::to_string(task->expert->layer_idx) +
               " E" + std::to_string(task->expert->expert_idx));
    CUDA_CALL(cudaStreamSynchronize(this->stream));
  }
  fetch_schedule_thread->add_one_task(&fetch_schedule_thread->copy_done_task);
}
void PredictWorker::add_prefetch_layer_budget() {
  LOG(DEBUG) << "predict worker: add prefetch layer budget";
  prefetch_layer_budget.push(0);
}
void PredictWorker::on_one_iter_done(int64_t forward_epoch, int64_t generate_epoch) {
  LOG(DEBUG) << "predict workers, one iter done";
  current_generate_epoch.store(generate_epoch, std::memory_order_release);
  switch (metas->predict_input_mode) {
    // case kNoPredict:               { break; }
    case kNoPredict:                { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); break; }
    case kOneToken:                { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); break; }
    case kDecodeCumsum:            { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); break; }
    case kLastUseDistance:         { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); break; }
    case kWeighedDecodeCumsum:     { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    case kMoeLayerLogits:          { break; }
    default: { CHECK(false) << "Unknown predict input mode"; }
  }
}
void PredictWorker::on_moe_attn_input_logits_recorded(int layer_id, int64_t forward_epoch, int64_t generate_epoch) {
  LOG(DEBUG) << "predict workers, on_moe_attn_input_logits_recorded " << layer_id;
  current_generate_epoch.store(generate_epoch, std::memory_order_release);
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: {
      if (layer_id == 0) { add_one_task(PredictJob(0, forward_epoch, generate_epoch)); }
      break;
    }
    case kMoeAttnInputLogits:      {
      if (predictor->layer_predict_enabled(layer_id)) {
      // if (layer_id % metas->layer_predict_interval == 0) {
        add_one_task(PredictJob(layer_id, forward_epoch, generate_epoch));
      }
      break;
    }
    case kMoeLayerLogits:          { break; }
    default: { CHECK(false) << "Unknown predict input mode"; }
  }
}
void PredictWorker::on_moe_layer_logits_recorded(int layer_id, int64_t forward_epoch, int64_t generate_epoch) {
  LOG(DEBUG) << "predict workers, on_moe_layer_logits_recorded " << layer_id;
  current_generate_epoch.store(generate_epoch, std::memory_order_release);
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: { break;}
    case kMoeAttnInputLogits:      { break;}
    case kMoeLayerLogits:          {
      if (predictor->layer_predict_enabled(layer_id)) {
        add_one_task(PredictJob(layer_id, forward_epoch, generate_epoch));
      }
      break;
    }
    default: { CHECK(false) << "Unknown predict input mode in on_moe_layer_logits_recorded"; }
  }
}
