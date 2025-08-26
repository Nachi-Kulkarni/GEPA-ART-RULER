🔍 DATASET SUMMARY:" && echo "==================" && echo &&
      echo "📊 Final counts:" && wc -l data/minimal_datasets/*.json && echo
       && echo "📈 Info file:" && cat data/minimal_datasets/info.json)
  ⎿  🔍 DATASET SUMMARY:                                                  
     ==================

     📊 Final counts:
          373 data/minimal_datasets/gepa_train.json
           51 data/minimal_datasets/gepa_val.json
           16 data/minimal_datasets/info.json
         1175 data/minimal_datasets/rl_train.json
          261 data/minimal_datasets/rl_val.json
         1876 total

     📈 Info file:
     {
       "total_problems": 100,
       "splits": {
         "rl_train": 80,
         "rl_val": 20,
         "gepa_train": 25,
         "gepa_val": 5
       },
       "sources": {
         "codeforces": 53,
         "atcoder": 29,
         "usaco": 18
       },
       "overlap_strategy": "30 GEPA problems are subset of 80 RL training
     problems",
       "evaluation": "OJBench (232 problems) - completely separate",
       "note": "Efficient data usage with strategic overlap"
     }

