import pickle

with open("/data/fangping/bulleye/Bullseye UPenn dict data/BE_cntrl_merged_counts.DNA.pkl", "rb") as file:
    print('loading...')
    loaded_data = pickle.load(file)

print(loaded_data['CIVYLPD'])