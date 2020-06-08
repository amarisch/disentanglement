bsub -n 4 -J s_enh_now -W 24:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_nogi_all/json --runname normal_weight --savepath result_nogi
bsub -n 4 -J s_enh_now -W 24:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_finall/all3 --runname normal_weight --savepath result_all3


#bsub -n 4 -J s_enh_now -W 24:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_over/json_shuf --runname normal_weight --savepath result_patenhanced_noweight --equalweights

# bsub -n 4 -J s_enhanced -W 24:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_over/json_shuf --runname normal_weight --savepath result_patenhanced

# bsub -n 4 -J search -W 100:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_over/json --runname normal_weight --savepath result_patover

# bsub -n 4 -J search -W 100:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_single/json --runname normal_weight --savepath result_patsingle

# bsub -n 4 -J search -W 100:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search2.py --data patdata_single/all --runname normal_weight --savepath result_patsingleall

# bsub -n 4 -J search -W 100:00 -R "rusage[mem=1500,ngpus_excl_p=1]" python hparam_search.py --data patdata_over/json --runname normal_weight

