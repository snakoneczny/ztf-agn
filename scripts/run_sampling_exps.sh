for timespan in 100  # 1800 1500 1250 1000 750 500 250
do
    for frac_n_obs in 1.0 0.8 0.6 0.4 0.2 0.1 0.05
    do
        python  retrain_astromer_tf.py -f g -s --timespan $timespan --frac_n_obs $frac_n_obs
    done
done
