#!/usr/bin/env bash

python split.py --tag full  --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                            --valid_start "2006-05-18 00:00:00" --valid_end "2006-05-25 00:00:00" \
                            --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00"
