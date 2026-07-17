cd /data1/tianang/Projects/Synergy/DataPrepare/MDLM
NUM_CORES=128
echo "NUM_CORES: ${NUM_CORES}"
python tokenize_SELFIES_descriptors_hf.py -s "010" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "011" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "012" -c $NUM_CORES


