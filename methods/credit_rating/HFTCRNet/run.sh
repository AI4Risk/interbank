#!/bin/bash

start_year=2018
start_quarter=1
end_year=2023
end_quarter=1
time_steps=7

offset_1=$((-(time_steps) - 1))
offset_2=$((-(time_steps)))

get_adjacent_quarter() {
    local year=$1
    local quarter=$2
    local offset=$3

    local new_quarter=$((quarter + offset))

    while [ $new_quarter -gt 4 ] || [ $new_quarter -lt 1 ]; do
        if [ $new_quarter -gt 4 ]; then
            year=$((year + 1))
            new_quarter=$((new_quarter - 4))
        elif [ $new_quarter -lt 1 ]; then
            year=$((year - 1))
            new_quarter=$((new_quarter + 4))
        fi
    done

    echo "$year $new_quarter"
}


for ((year=start_year; year<=end_year; year++)); do
    if [ $year -eq $start_year ]; then
        start_q=$start_quarter
    else
        start_q=1
    fi
    
    if [ $year -eq $end_year ]; then
        end_q=$end_quarter
    else
        end_q=4
    fi
    
    for ((quarter=start_q; quarter<=end_q; quarter++)); do
        train_result_from=($(get_adjacent_quarter $year $quarter $offset_1))
        train_year_from=${train_result_from[0]}
        train_quarter_from=${train_result_from[1]}

        train_result_to=($(get_adjacent_quarter $year $quarter -2))
        train_year_to=${train_result_to[0]}
        train_quarter_to=${train_result_to[1]}

        test_result_from=($(get_adjacent_quarter $year $quarter $offset_2))
        test_year_from=${test_result_from[0]}
        test_quarter_from=${test_result_from[1]}

        test_result_to=($(get_adjacent_quarter $year $quarter -1))
        test_year_to=${test_result_to[0]}
        test_quarter_to=${test_result_to[1]} 

        result=($(get_adjacent_quarter $year $quarter -2))
        single_year=${result[0]}
        single_quarter=${result[1]}

        CUDA_VISIBLE_DEVICES=0 python train.py --train_year_from $train_year_from --train_quarter_from $train_quarter_from --train_year_to $train_year_to --train_quarter_to $train_quarter_to --test_year_from $test_year_from --test_quarter_from $test_quarter_from --test_year_to $test_year_to --test_quarter_to $test_quarter_to --d_model 256 --d_k 64 --d_v 64 --n_layers_lt3 3 --n_layers_stcgt 2 --n_layers_arct 2 --n_heads 4 --cross_trans_heads 1 --lr 0.001
    done
done