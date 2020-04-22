#!/bin/bash

# bash file to be called for executing all figures

rm -f Figure*.md
rm -f SMFigure*.md
jupyter nbconvert --to notebook --inplace --execute -y Figure*.ipynb
jupyter nbconvert --to notebook --inplace --execute -y SMFigure*.ipynb
jupytext --to markdown Figure*.ipynb
jupytext --to markdown SMFigure*.ipynb

cd figdump/
bash figzip.sh
cd ..
