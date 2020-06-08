
t=( 0 0.00001 0.0001 0.001 0.01 0.1 0.5 1.0 )

# folders=( test3 test1 test2 )
# for i in "${!folders[@]}"; do
# 	mkdir 5_19/"${folders[$i]}"/prc
# 	for val in "${t[@]}"; do
# 		python3 prcurve.py --t "$val" --loadpath 5_19/"${folders[$i]}" --data patdata_final/all3 
# 	done
# done

# folders=( test48_1_single test48_2_single test48_3_single )
# for i in "${!folders[@]}"; do
# 	mkdir 5_19/"${folders[$i]}"/prc
# 	for val in "${t[@]}"; do
# 		python3 prcurve.py --t "$val" --loadpath 5_19/"${folders[$i]}" --data patdata_single/json
# 	done
# done

# folders=( test24_3 test24_2 test24_1 )
# for i in "${!folders[@]}"; do
# 	mkdir 5_19/"${folders[$i]}"/prc
# 	for val in "${t[@]}"; do
# 		echo "$val"
# 		python3 prcurve.py --t "$val" --loadpath 5_19/"${folders[$i]}" --data patdata24_all/json --seqlen 24
# 	done
# done

# folders=( test24_3_single test24_2_single test24_1_single )

# for i in "${!folders[@]}"; do
# 	mkdir 5_19/"${folders[$i]}"/prc
# 	for val in "${t[@]}"; do
# 		echo "$val"
# 		python3 prcurve.py --t "$val" --loadpath 5_19/"${folders[$i]}" --data patdata24/json --seqlen 24
# 	done
# done

# put plots in the right folders
# folders=( test3 test24_3 test24_2_single test1 test24_1_single test24_3_single test48_1_single test24_2 test48_3_single test24_1 test48_2_single test2 )
# cd 5_19
# for i in "${!folders[@]}"; do
# 	cd "${folders[$i]}"
# 	mv `find . -maxdepth 1 -name "*prc_plot*"` prc
# 	cd prc
# 	mkdir heart
# 	mkdir lung
# 	mkdir liver
# 	mkdir gi
# 	mkdir kidney
# 	mv `find . -maxdepth 1 -name "*heart*"` heart
# 	mv `find . -maxdepth 1 -name "*lung*"` lung
# 	mv `find . -maxdepth 1 -name "*liver*"` liver
# 	mv `find . -maxdepth 1 -name "*gi*"` gi
# 	mv `find . -maxdepth 1 -name "*kidney*"` kidney
# 	cd ../..
# done


# folders=( test3 test1 test2 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.5,1.0" --loadpath 5_19/"${folders[$i]}" --data patdata_final/all3 
# done

# folders=( test48_1_single test48_2_single test48_3_single )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.5,1.0" --loadpath 5_19/"${folders[$i]}" --data patdata_single/json
# done

# folders=( test24_3 test24_2 test24_1 )
# for i in "${!folders[@]}"; do  
# 	echo "${folders[$i]}"
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.5,1.0" --loadpath 5_19/"${folders[$i]}" --data patdata24_all/json --seqlen 24
# done

# folders=( test24_3_single test24_2_single test24_1_single )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.5,1.0" --loadpath 5_19/"${folders[$i]}" --data patdata24/json --seqlen 24
# done


# folders=( test3 test24_3 test24_2_single test1 test24_1_single test24_3_single test48_1_single test24_2 test48_3_single test24_1 test48_2_single test2 )
# cd 5_19
# for i in "${!folders[@]}"; do
# 	mv "${folders[$i]}"/run_prc_out "${folders[$i]}"/run_prc_out.csv
# done


# folders=( 5_19/test24_1 5_19/test24_2 5_19/test24_3 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all/json  --seqlen 24
# done

# folders=( 5_19/test1 5_19/test2 5_19/test3 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata_final/all3
# done

# folders=( 5_19/test24_1 5_19/test24_2 5_19/test24_3 5_19/test1 5_19/test2 5_19/test3 5_21/test24_1_all3 5_21/test24_2_all3 5_21/test24_3_all3 5_21/test24_1_all5 5_21/test24_2_all5 5_21/test24_3_all5 5_21/test48_1_all3 5_21/test48_2_all3 5_21/test48_3_all3  5_21/test48_1_all5 5_21/test48_2_all5 5_21/test48_3_all5 )

# # folders=( 5_21/test48_3_all5 5_21/test48_2_all5 5_21/test48_1_all5 5_21/test24_1_all5 5_21/test24_2_all5 5_21/test24_3_all5 5_21/test24_1_all3 5_21/test24_2_all3 5_21/test24_3_all3 )
# #folders=( 5_21/test48_1_all3 5_21/test48_2_all3 5_21/test48_3_all3 )
# # folders=( 5_19/test24_1 5_19/test24_2 5_19/test24_3 )
# folders=( 5_25/test24_2_all3 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all3_seq10/json --seqlen 24
# done

# folders=( 5_21/test24_1_all3 5_21/test24_2_all3 5_21/test24_3_all3 5_21/test24_1_all5 5_21/test24_2_all5 5_21/test24_3_all5  )
# for i in "${!folders[@]}"; do
# 	cat "${folders[$i]}"/run_prc_out.csv >> prc24.csv	
# done

# folders=( 5_21/test48_1_all3 5_21/test48_2_all3 5_21/test48_3_all3 5_21/test48_1_all5 5_21/test48_2_all5 5_21/test48_3_all5 )
# for i in "${!folders[@]}"; do
# 	cat "${folders[$i]}"/run_prc_out.csv >> prc48.csv	
# done

# folders=( test24_1_all3 test24_2_all3 test24_3_all3 test24_1_all5 test24_2_all5 test24_3_all5 test48_1_all3 test48_2_all3 test48_3_all3 test48_1_all5 test48_2_all5 test48_3_all5 )
# for i in "${!folders[@]}"; do
# 	mkdir 5_25/plots/"${folders[i]}"
# 	cp -r 5_21/"${folders[i]}"/prc 5_25/plots/"${folders[i]}"
# done


# for i in "${!folders[@]}"; do  
# 	#echo "${folders[$i]}"
# 	#mkdir "${folders[$i]}"/prc
# 	#python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all3_seq10/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> prc24_all3_seq5.csv	
# done


# cd 5_19

# folders=( test24_1 test24_2 test24_3 )
# for i in "${!folders[@]}"; do
# 	cat "${folders[$i]}"/run_prc_out.csv >> prc24.csv	
# done

# folders=( test24_1_single test24_2_single test24_3_single )
# for i in "${!folders[@]}"; do
# 	cat "${folders[$i]}"/run_prc_out.csv >> prc24.csv	
# done





# # folders=( 5_21/test24_1_all3 5_21/test24_2_all3 5_21/test24_3_all3 )
# folders=( 5_25/test24_1_all3 5_25/test24_2_all3 5_25/test24_3_all3 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all3_seq10/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_25/prc24.csv	
# done

# # folders=( 5_21/test24_1_all5 5_21/test24_2_all5 5_21/test24_3_all5 )
# folders=( 5_25/test24_1_all5 5_25/test24_2_all5 5_25/test24_3_all5 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq10/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_25/prc24.csv
# done

# # folders=( 5_21/test48_1_all3 5_21/test48_2_all3 5_21/test48_3_all3 )
# folders=( 5_25/test48_1_all3 5_25/test48_2_all3 5_25/test48_3_all3 )
# for i in "${!folders[@]}"; do  
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all3_seq10/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_25/prc48.csv	
# done

# # folders=( 5_21/test48_1_all5 5_21/test48_2_all5 5_21/test48_3_all5 )
# folders=( 5_25/test48_1_all5 5_25/test48_2_all5 5_25/test48_3_all5 )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all5_seq10/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_25/prc48.csv	
# done

# folders=( 5_25/test24_1_all3 5_25/test24_2_all3 5_25/test24_3_all3 5_25/test24_1_all5 5_25/test24_2_all5 5_25/test24_3_all5 5_25/test48_1_all3 5_25/test48_2_all3 5_25/test48_3_all3 5_25/test48_1_all5 5_25/test48_2_all5 5_25/test48_3_all5 )
# folders=( 5_26/test24_1_all5_feat65 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done


# folders=( test24_1_all3 test24_2_all3 test24_3_all3 test24_1_all5 test24_2_all5 test24_3_all5 test48_1_all3 test48_2_all3 test48_3_all3 test48_1_all5 test48_2_all5 test48_3_all5 )
# for i in "${!folders[@]}"; do
# 	mkdir 5_26/plots/"${folders[i]}"
# 	cp -r 5_25/"${folders[i]}"/prc 5_26/plots/"${folders[i]}"
# done

# f=( test24_1_all5_feat65/ test24_2_all5_feat65/ test24_3_all5_feat65/ test48_1_all5_feat65/ test48_2_all5_feat65/ test48_3_all5_feat65/ )
# folders=( test24_1_all3_feat65/ test24_2_all3_feat65/ test24_3_all3_feat65/ test48_1_all3_feat65/ test48_2_all3_feat65/ test48_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	cp 5_26/"${f[i]}"/hparam.json 5_27/"${folders[i]}"
# done

# folders=( 5_26/test24_2_all5_feat65/ 5_26/test24_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq10_feat65/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_26/prc24.csv
# done

# folders=( 5_26/test48_1_all5_feat65/ 5_26/test48_2_all5_feat65/ 5_26/test48_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all5_seq10_feat65/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_26/prc48.csv	
# done

# folders=( 5_26/test24_2_all5_feat65/ 5_26/test24_3_all5_feat65/ 5_26/test48_1_all5_feat65/ 5_26/test48_2_all5_feat65/ 5_26/test48_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done

# folders=( test24_1_all5_feat65/ test24_2_all5_feat65/ test24_3_all5_feat65/ test48_1_all5_feat65/ test48_2_all5_feat65/ test48_3_all5_feat65/ )
# mkdir 5_26/plots2
# for i in "${!folders[@]}"; do
# 	mkdir 5_26/plots2/"${folders[i]}"
# 	cp -r 5_26/"${folders[i]}"/prc 5_26/plots2/"${folders[i]}"
# done

# folders=( 5_27/test24_1_all3_feat65/ 5_27/test24_2_all3_feat65/ 5_27/test24_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all3_seq10_feat65/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_27/prc24.csv
# done

# folders=( 5_27/test48_1_all3_feat65/ 5_27/test48_2_all3_feat65/ 5_27/test48_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all3_seq10_feat65/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_27/prc48.csv	
# done

# folders=( 5_27/test24_1_all3_feat65/ 5_27/test24_2_all3_feat65/ 5_27/test24_3_all3_feat65/ 5_27/test48_1_all3_feat65/ 5_27/test48_2_all3_feat65/ 5_27/test48_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done

# folders=( test24_1_all3_feat65/ test24_2_all3_feat65/ test24_3_all3_feat65/ test48_1_all3_feat65/ test48_2_all3_feat65/ test48_3_all3_feat65/ )
# mkdir 5_27/plots
# for i in "${!folders[@]}"; do
# 	mkdir 5_27/plots/"${folders[i]}"
# 	cp -r 5_27/"${folders[i]}"/prc 5_27/plots/"${folders[i]}"
# done


# f=( test24_1_all5_feat65/ test24_2_all5_feat65/ test24_3_all5_feat65/ test48_1_all5_feat65/ test48_2_all5_feat65/ test48_3_all5_feat65/ )
# folders=( test24_1_all3_feat65/ test24_2_all3_feat65/ test24_3_all3_feat65/ test48_1_all3_feat65/ test48_2_all3_feat65/ test48_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	mkdir 5_28/"${folders[i]}"
# 	mkdir 5_28/"${f[i]}"
# 	cp 5_26/"${f[i]}"/hparam.json 5_28/"${folders[i]}"
# 	cp 5_26/"${f[i]}"/hparam.json 5_28/"${f[i]}"
# done



# folders=( 5_28/test24_1_all3_feat65/ 5_28/test24_2_all3_feat65/ 5_28/test24_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all3_seq5_feat65/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc24.csv
# done

# folders=( 5_28/test48_1_all3_feat65/ 5_28/test48_2_all3_feat65/ 5_28/test48_3_all3_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all3_seq5_feat65/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc48.csv	
# done

# folders=( 5_28/test24_1_all5_feat65/ 5_28/test24_2_all5_feat65/ 5_28/test24_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq5_feat65/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc24.csv
# done

# folders=( 5_28/test48_1_all5_feat65/ 5_28/test48_2_all5_feat65/ 5_28/test48_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all5_seq5_feat65/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc48.csv	
# done

# folders=( 5_28/test24_1_all3_feat65/ 5_28/test24_2_all3_feat65/ 5_28/test24_3_all3_feat65/ 5_28/test48_1_all3_feat65/ 5_28/test48_2_all3_feat65/ 5_28/test48_3_all3_feat65/ 5_28/test24_1_all5_feat65/ 5_28/test24_2_all5_feat65/ 5_28/test24_3_all5_feat65/ 5_28/test48_1_all5_feat65/ 5_28/test48_2_all5_feat65/ 5_28/test48_3_all5_feat65/ )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done

# folders=( test24_1_all3_feat65/ test24_2_all3_feat65/ test24_3_all3_feat65/ test48_1_all3_feat65/ test48_2_all3_feat65/ test48_3_all3_feat65/ test24_1_all5_feat65/ test24_2_all5_feat65/ test24_3_all5_feat65/ test48_1_all5_feat65/ test48_2_all5_feat65/ test48_3_all5_feat65/ )
# mkdir 5_28/plots
# for i in "${!folders[@]}"; do
# 	mkdir 5_28/plots/"${folders[i]}"
# 	cp -r 5_28/"${folders[i]}"/prc 5_28/plots/"${folders[i]}"
# done


# f=( test24_1_all5_feat65/ test24_2_all5_feat65/ test24_3_all5_feat65/ test48_1_all5_feat65/ test48_2_all5_feat65/ test48_3_all5_feat65/ )
# folders=( test24_1_all5_feat13/ test24_2_all5_feat13/ test24_3_all5_feat13/ test48_1_all5_feat13/ test48_2_all5_feat13/ test48_3_all5_feat13/ )
# for i in "${!folders[@]}"; do
# 	mkdir 5_28/"${folders[i]}"
# 	cp 5_28/"${f[i]}"/hparam.json 5_28/"${folders[i]}"
# done


# folders=( 5_28/test24_1_all5_feat13/ 5_28/test24_2_all5_feat13/ 5_28/test24_3_all5_feat13/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq5/json --seqlen 24
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc24_feat13.csv
# done

# folders=( 5_28/test48_1_all5_feat13/ 5_28/test48_2_all5_feat13/ 5_28/test48_3_all5_feat13/ )
# for i in "${!folders[@]}"; do
# 	echo "${folders[$i]}"
# 	mkdir "${folders[$i]}"/prc	
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.003,0.005,0.007,0.01,0.1,0.5,1.0" --loadpath "${folders[$i]}" --data patdata48_all5_seq5/json
# 	cat "${folders[$i]}"/run_prc_out.csv >> 5_28/prc48_feat13.csv	
# done

#folders=( 5_28/test24_1_all5_feat13/ 5_28/test24_2_all5_feat13/ 5_28/test24_3_all5_feat13/ 5_28/test48_1_all5_feat13/ 5_28/test48_2_all5_feat13/ 5_28/test48_3_all5_feat13/ )
#python3 prcurve_cov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath 5_29/test1 --data patdata24_all5_seq10_feat65/json --seqlen 24
#python3 prcurve_cov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath 5_29/test2 --data patdata24_all5_seq10_feat65/json --seqlen 24

# folders=( 6_3/test2 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve_cov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all8_seq20_feat65_cat8/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mkdir "${folders[$i]}"/prc/nervous
# 	mkdir "${folders[$i]}"/prc/endo
# 	mkdir "${folders[$i]}"/prc/blood	
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*nervous*"` "${folders[$i]}"/prc/nervous
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*endo*"` "${folders[$i]}"/prc/endo
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*blood*"` "${folders[$i]}"/prc/blood
# done

# folders=( 6_3/test1 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all8_seq20_feat65_cat8/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mkdir "${folders[$i]}"/prc/nervous
# 	mkdir "${folders[$i]}"/prc/endo
# 	mkdir "${folders[$i]}"/prc/blood	
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*nervous*"` "${folders[$i]}"/prc/nervous
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*endo*"` "${folders[$i]}"/prc/endo
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*blood*"` "${folders[$i]}"/prc/blood
# done


# folders=( 6_3/test3 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve_cov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq20_feat65/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mkdir "${folders[$i]}"/prc/nervous
# 	mkdir "${folders[$i]}"/prc/endo
# 	mkdir "${folders[$i]}"/prc/blood	
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*nervous*"` "${folders[$i]}"/prc/nervous
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*endo*"` "${folders[$i]}"/prc/endo
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*blood*"` "${folders[$i]}"/prc/blood
# done

# folders=( 6_3/test4 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq20_feat65/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mkdir "${folders[$i]}"/prc/nervous
# 	mkdir "${folders[$i]}"/prc/endo
# 	mkdir "${folders[$i]}"/prc/blood	
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*nervous*"` "${folders[$i]}"/prc/nervous
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*endo*"` "${folders[$i]}"/prc/endo
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*blood*"` "${folders[$i]}"/prc/blood
# done

# folders=( 6_4/test7 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve_cov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq20_feat65/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done


folders=( 6_5/test2 )
for i in "${!folders[@]}"; do
	mkdir "${folders[$i]}"/prc
	python3 prcurve_xcov.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq20_feat65/json --seqlen 24
	mkdir "${folders[$i]}"/prc/heart
	mkdir "${folders[$i]}"/prc/lung
	mkdir "${folders[$i]}"/prc/liver
	mkdir "${folders[$i]}"/prc/gi
	mkdir "${folders[$i]}"/prc/kidney
	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
done


# folders=( 6_4/test3 6_4/test3_2 )
# for i in "${!folders[@]}"; do
# 	mkdir "${folders[$i]}"/prc
# 	python3 prcurve2.py --t "0,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.7,0.8,1.0" --loadpath "${folders[$i]}" --data patdata24_all5_seq20_feat65/json --seqlen 24
# 	mkdir "${folders[$i]}"/prc/heart
# 	mkdir "${folders[$i]}"/prc/lung
# 	mkdir "${folders[$i]}"/prc/liver
# 	mkdir "${folders[$i]}"/prc/gi
# 	mkdir "${folders[$i]}"/prc/kidney
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*heart*"` "${folders[$i]}"/prc/heart
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*lung*"` "${folders[$i]}"/prc/lung
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*liver*"` "${folders[$i]}"/prc/liver
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*gi*"` "${folders[$i]}"/prc/gi
# 	mv `find "${folders[$i]}"/prc -maxdepth 1 -name "*kidney*"` "${folders[$i]}"/prc/kidney
# done



# folders=( test24_1_all5_feat13/ test24_2_all5_feat13/ test24_3_all5_feat13/ test48_1_all5_feat13/ test48_2_all5_feat13/ test48_3_all5_feat13/ )
# mkdir 5_28/plots
# for i in "${!folders[@]}"; do
# 	mkdir 5_28/plots/"${folders[i]}"
# 	cp -r 5_28/"${folders[i]}"/prc 5_28/plots/"${folders[i]}"
# done
