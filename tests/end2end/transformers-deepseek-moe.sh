# # ------------------ normal config ------------------
# python transformers-deepseek-moe.py --num_predict_expert_per_layer 6  --cache_rate 0.1875 --cache_policy lru   --enable_per_layer_cache 
# # ------------------ full cache ---------------------
# python transformers-deepseek-moe.py --num_predict_expert_per_layer 0  --cache_rate 1      --cache_policy lru   --enable_per_layer_cache 
# # ------------------ disable prefetch ---------------
# python transformers-deepseek-moe.py --num_predict_expert_per_layer 0  --cache_rate 0.1875 --cache_policy lru   --enable_per_layer_cache 
# ------------------ corner test 1 ------------------
python transformers-deepseek-moe.py --num_predict_expert_per_layer 6  --cache_len 1       --cache_policy lru   --disable_per_layer_cache --max_prefetch_layer_distance 1 
# ------------------ corner test 2 ------------------
python transformers-deepseek-moe.py --num_predict_expert_per_layer 6  --cache_len 1       --cache_policy fifo  --disable_per_layer_cache --max_prefetch_layer_distance 1 
# ------------------ corner test 3 ------------------
python transformers-deepseek-moe.py --num_predict_expert_per_layer 0  --cache_len 1       --cache_policy fifo  --disable_per_layer_cache --max_prefetch_layer_distance 1 
