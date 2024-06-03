#include "cache.hpp"
#include "logging.hpp"

ExpertMemHanlder *CacheMngr::allocate_from_free_buffer() {
  ExpertMemHanlder *ret = nullptr;
  unused_mems_lock.lock();
  if (unused_mems.size() > 0) {
    ret = unused_mems.back();
    CHECK(ret != nullptr);
    unused_mems.pop_back();
  } else {
    CHECK(false) << "no remaining mem buffer";
  }
  unused_mems_lock.unlock();
  return ret;
}
void CacheMngr::init_gpu_mem_buffer(size_t num_buffers) {
  unused_mems.resize(num_buffers, nullptr);
  auto &mem_example = model_loader->get_source(0, 0)->host_data.mem_buffers;
  for (int i = 0; i < num_buffers; i++) {
    unused_mems[i] = new ExpertMemHanlder;
    unused_mems[i]->mem_buffers.resize(mem_example.size());
    for (int j = 0; j < mem_example.size(); j++) {
      unused_mems[i]->mem_buffers[j].set_tensor(torch::empty_like(mem_example[j].get_tensor(), torch::TensorOptions().device(torch::kCUDA, 0)));
    }
  }
}
CacheMngr::~CacheMngr() {
  size_t used_mem_cnt = 0;
  for (auto &l : prefetched_experts) {
    used_mem_cnt += l.size();
  }
  LOG(ERROR) << unused_mems.size() << "+" << used_mem_cnt << "=" << unused_mems.size() + used_mem_cnt;
}
CacheMngr::CacheMngr(std::shared_ptr<ModuleMeta> metas,
                     std::shared_ptr<ModelLoader> model_loader)
    : metas(metas), model_loader(model_loader) {
  prefetched_experts.resize(metas->num_layer);
}
