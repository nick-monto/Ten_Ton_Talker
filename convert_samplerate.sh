#!/bin/bash
for d in ./Input_audio_wav_16k/*; do
  echo Changing to $d;
  cd $d;
  echo Converting ~44k to 16k...;
  for i in *.wav; do
    sox -S "$i" -r 16000 -b 16 "${i/.wav/}_converted".wav;
    rm $i;
    done
  echo Changing back to original directory;
  cd ../..
  done
