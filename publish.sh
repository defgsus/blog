#!/bin/bash
cd blog || exit 1
bundle exec jekyll build -d ../docs
cd ..
