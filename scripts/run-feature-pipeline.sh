#!/bin/bash

set -e

jupyter nbconvert --to notebook --execute Final_Hopsworks_feature_pipeline.ipynb
