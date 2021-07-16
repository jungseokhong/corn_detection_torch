#!/bin/bash
cd results
ffmpeg -r 15 -pattern_type glob -i '*.png' -vcodec libx264 -crf 25 -s 848x480 -pix_fmt yuv420p test.mp4
mv test.mp4 ../out_results.mp4