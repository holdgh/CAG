#!/bin/bash

python ./kvcache.py --kvcache file --dataset "squad-train" --similarity bertscore --maxKnowledge 5 --maxParagraph 100 --maxQuestion 1000 --modelname "E:\aiModel\llmModel\Qwen2.5-0.5B-Instruct" --randomSeed 0 --output "./result_kvcache.txt"