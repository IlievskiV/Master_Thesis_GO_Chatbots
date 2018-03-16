#!/bin/bash

cd docs
python autogen.py
cd ../
mkdocs build
mkdocs serve