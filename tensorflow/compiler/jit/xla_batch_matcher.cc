#include "tensorflow/compiler/jit/xla_batch_matcher.h"
#include "xla/debug_options_flags.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

XlaBatchMatcher::XlaBatchMatcher() {
  env_str_ = xla::GetDebugOptionsFromFlags().xla_compile_batch_sizes();
  parse_env_config();
}

// Trim whitespace (spaces/tabs) from both ends of a string
std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t");
  size_t end = s.find_last_not_of(" \t");
  return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

std::vector<int64_t> XlaBatchMatcher::parse_single_item(const std::string& item) {
  std::vector<int64_t> batch_list;
  if (item.empty()) return batch_list;

  // 1. Parse single value (no colon separator)
  if (item.find(':') == std::string::npos) {
    // Validate all characters are digits (reject non-numeric chars)
    for (char c : item) {
      if (!isdigit(c)) {
        throw std::invalid_argument("Non-numeric characters: " + item);
      }
    }
    auto val = static_cast<int64_t>(std::stoll(item));
    if (val <= 0 || val > kMaxBatch) {
      throw std::invalid_argument("Out of valid range (1-" + std::to_string(kMaxBatch) + "): " + item);
    }
    batch_list.push_back(val);
    return batch_list;
  }

  // 2. Parse range format (start:end:step)
  std::stringstream ss(item);
  std::string part;
  std::vector<std::string> parts;
  while (std::getline(ss, part, ':')) {
    parts.push_back(trim(part));
  }
  if (parts.size() > 3) {
    throw std::invalid_argument("Invalid range format (requires start:end:step): " + item);
  }

  // Convert and validate range parameters
  int64_t start, end, step;
  try {
    start = std::stoi(parts[0]);
    end = std::stoi(parts[1]);
    step = parts.size() == 2 ? 1 : std::stoi(parts[2]);
  } catch (...) {
    throw std::invalid_argument("Invalid numeric values in range: " + item);
  }
  if (start <= 0 || end <= 0 || step <= 0) {
    throw std::invalid_argument("Range parameters must be positive integers: " + item);
  }
  if (start > end) {
    throw std::invalid_argument("Start value > end value in range: " + item);
  }
  if (end > kMaxBatch) {
    throw std::invalid_argument("Range exceeds max limit (" + std::to_string(kMaxBatch) + "): " + item);
  }

  // Generate batch list from range
  for (int64_t i = start; i <= end; i += step) {
    batch_list.push_back(i);
  }
  return batch_list;
}

void XlaBatchMatcher::print_all_batches(const std::string& cluster_key,
                                       const std::vector<int64_t>& batches) {
  std::ostringstream oss;
  oss << "[XLA_BATCH_INFO] cluster_key=" << cluster_key
      << " valid batch list update: ";
  for (size_t i = 0; i < batches.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << batches[i];
  }
  LOG(INFO) << oss.str();
}

// Parse environment variable config into a seed batch list used to initialize
// each cluster-key state.
void XlaBatchMatcher::parse_env_config() {
  // If the env var not set or is empty, leave seed empty and let per-cluster
  // logic generate factor-preserving padded batches.
  if (env_str_.empty()) {
    VLOG(2) << "[XLA_BATCH_WARN] Env var " << "--tf_xla_compile_batch_sizes" <<
        "is empty, will use factor-preserving padding by default";
    return;
  }

  // Parse into the default seed state.
  ClusterState& seed = clusters_[""];

  std::stringstream ss(env_str_);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed_item = trim(item);
    if (trimmed_item.empty()) continue;

    try {
      std::vector<int64_t> item_batches = parse_single_item(trimmed_item);
      seed.all_batches.insert(seed.all_batches.end(), item_batches.begin(),
                              item_batches.end());
    } catch (const std::exception& e) {
      LOG(INFO) << "[XLA_BATCH_WARN] Failed to parse config item, skipping: "
                << trimmed_item << " (" << e.what() << ")";
    }
  }

  if (!seed.all_batches.empty()) {
    std::sort(seed.all_batches.begin(), seed.all_batches.end());
    auto last = std::unique(seed.all_batches.begin(), seed.all_batches.end());
    seed.all_batches.erase(last, seed.all_batches.end());
    print_all_batches("<seed>", seed.all_batches);
  }
}

// Calculate the smallest power of two greater than the real batch
static int64_t GetNextPowerOfTwo(int64_t real_batch) {
  // If real_batch is already a power of two, return real_batch directly
  if ((real_batch & (real_batch - 1)) == 0) {
    return real_batch;
  }

  int64_t power = 1;
  while (power < real_batch) {
    power <<= 1;
  }
  return power;
}

// Small/large multipliers for padding are configurable via environment variables:
// TF_XLA_BATCH_SMALL_FACTOR (default 10) and TF_XLA_BATCH_LARGE_FACTOR (default 2).
static int64_t GetEnvFactorOrDefault(const char* name, int64_t def) {
  const char* val = std::getenv(name);
  if (val == nullptr) return def;
  int64_t parsed = 0;
  if (!absl::SimpleAtoi(val, &parsed) || parsed <= 0) {
    LOG(WARNING) << "[XLA_BATCH_WARN] Failed to parse env var " << name
                 << "=\"" << val << "\", using default: " << def;
    return def;
  }
  return parsed;
}

static int64_t GetFactorPreservingBatch(int64_t real_batch) {
  // Cache env values in function-local statics (thread-safe since C++11).
  static const int64_t small_factor =
      GetEnvFactorOrDefault("TF_XLA_BATCH_SMALL_FACTOR", 10);
  static const int64_t large_factor =
      GetEnvFactorOrDefault("TF_XLA_BATCH_LARGE_FACTOR", 2);

  const int64_t k = (real_batch < 10) ? small_factor : large_factor;
  // Guard overflow / max range.
  if (real_batch > kMaxBatch / k) {
    LOG(WARNING) << "[XLA_BATCH_ERR] Out of valid range: " << real_batch;
    return real_batch;
  }
  return real_batch * k;
}

int64_t XlaBatchMatcher::find_min_larger_batch(ClusterState* state,
                                              int64_t real_batch) {
  if (real_batch <= 0 || real_batch > kMaxBatch) {
    LOG(INFO) << "[XLA_BATCH_WARN] Out of valid range: " << real_batch;
    return real_batch;
  }

  // 1) If there is no configured candidate list, do not use power-of-two.
  if (state->all_batches.empty()) {
    const int val = GetFactorPreservingBatch(real_batch);
    state->all_batches.push_back(val);
    state->updated = true;
    return val;
  }

  // 2) Prefer a configured candidate strictly larger than real_batch.
  auto ub = std::upper_bound(state->all_batches.begin(), state->all_batches.end(),
                             real_batch);
  if (ub != state->all_batches.end()) {
    return *ub;
  }

  // 3) real_batch > all_batches_.back(): generate a factor-preserving padded
  // batch, write it back, keep list sorted/unique.
  const int64_t val = GetFactorPreservingBatch(real_batch);
  auto insert_pos =
      std::lower_bound(state->all_batches.begin(), state->all_batches.end(), val);
  if (insert_pos == state->all_batches.end() || *insert_pos != val) {
    state->all_batches.insert(insert_pos, val);
    state->updated = true;
  }
  return val;
}

int64_t XlaBatchMatcher::get_xla_compile_batch(const std::string& cluster_key,
                                              int64_t real_batch) {
  mutex_lock lock(mu_);
  // Lazily initialize per-key state from the seed ("" key) to preserve previous
  // behavior when an env list is provided.
  ClusterState& state = clusters_[cluster_key];
  state.updated = false;
  if (state.all_batches.empty()) {
    auto it = clusters_.find("");
    if (it != clusters_.end()) {
      state.all_batches = it->second.all_batches;
    }
  }

  const int64_t selected = find_min_larger_batch(&state, real_batch);

  if (state.updated) {
    VLOG(2) << "[XLA_BATCH_INFO] cluster_key=" << cluster_key
            << " real batch: " << real_batch
            << " -> selected compile batch: " << selected;
    print_all_batches(cluster_key, state.all_batches);
  }

  return selected;
}

int64_t XlaBatchMatcher::get_xla_compile_batch(int64_t real_batch) {
  return get_xla_compile_batch("", real_batch);
}

std::vector<int64_t> XlaBatchMatcher::get_all_batches(
    const std::string& cluster_key) {
  mutex_lock lock(mu_);
  auto it = clusters_.find(cluster_key);
  if (it == clusters_.end()) return {};
  return it->second.all_batches;
}

}  // namespace tensorflow
