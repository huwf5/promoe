#include "utils.hpp" 
#include "logging.hpp"


void AtomicMultiStatusLock::unlock(ExpertStatus from, ExpertStatus to) {
  int actual_cur_status = lock_.exchange(to);
  CHECK(actual_cur_status == from) << actual_cur_status << "!=" << from;
}
bool AtomicMultiStatusLock::try_unlock(ExpertStatus from, ExpertStatus to) {
  int from_ = from, to_ = to;
  return lock_.compare_exchange_strong(from_, to_);
  // return (actual_cur_status == from);
}
void AtomicMultiStatusLock::lock(ExpertStatus from, ExpertStatus to) {
  int from_ = from, to_ = to;
  while (lock_.compare_exchange_strong(from_, to_) == false) {
    from_ = from;
  };
}
bool AtomicMultiStatusLock::is_locked(ExpertStatus locked_status) {
  return lock_.load() == locked_status;
}

void AtomicLock::unlock() {
  bool assume_cur_status = true;
  CHECK(lock_.compare_exchange_strong(assume_cur_status, false));
  // CHECK(actual_cur_status == true);
}
void AtomicLock::lock() {
  bool cur_status = false;
  while (lock_.compare_exchange_strong(cur_status, true) == false) {
    cur_status = false;
  };
}
bool AtomicLock::is_locked() { return lock_.load(); }

std::string tensor_to_str(torch::Tensor t) {
  std::stringstream ss;
  t = t.flatten();
  for (int i = 0; i < t.numel(); i++) {
    ss << t[i].item() << ",";
  }
  return ss.str();
}
