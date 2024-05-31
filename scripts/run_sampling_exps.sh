for timespan in 100 250 500 750 1000 1250 1500 1800
do
    for frac_n_obs in 1.0 0.8 0.6 0.4 0.2 0.1 0.05
    do
        python  retrain_astromer_tf.py -f g --timespan $timespan --frac_n_obs $frac_n_obs
    done
done
