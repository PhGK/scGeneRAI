# GeneNetworkLRP
Gene networks

# train network:
./train_network
--> starts main_large.py with training flag in slurm


# compute LRP values:
./start_batch_LRP_array
--> starts main_large.py for every cell in slurm to compute LRP, network must be trained before


# compute LRPau scores
./loop_LRP_au 
--> loops through cells for which raw LRP scores are computed 
--> starts SLURM script with create_LRPau.py for every cell to compute LRP_au scores


# pull highest LRP_au scores from every cell
./loop_start_highest_int_with_name
--> loops through all LRP_au files
--> starts start_filter_higehst_interactions --> starts filter_h_i_without_dask.py
--> manually start concat_highinteractions.py --> creates file high_values_concat.csv
