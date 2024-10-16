# missing: 1500 onward and 0.8 onward
for timespan in 1500 1250 1000 750 500 250 100  # 1800
do
    for frac_n_obs in 0.8 0.6 0.4 0.2 0.1 0.05  # 1.0
    do
        python  retrain_astromer_tf.py -f g -s --timespan $timespan --frac_n_obs $frac_n_obs
    done
done
