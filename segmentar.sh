#!/bin/bash
#yourfilenames=`ls $1/*.{mp4,avi}`
if [ -z "$2" ]
  then
    salida=videossegmentados
  else
    salida=$2
fi
if [ -z "$3" ]
  then
    frames=10
  else
    frames=$3
fi
mkdir $salida
for entry in $1/*.avi;
do
  echo $entry
  mkdir $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')
  Yolo_mark/yolo_mark $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.') cap_video $entry $frames
  ls $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg>>train.txt
done
for entry in $1/*.mp4;
do
  echo $entry
  mkdir $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')
  Yolo_mark/yolo_mark $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.') cap_video $entry $frames
  ls $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg>>train.txt
done
if [ ! -z "$4" ]
   then
     if [ ! -z "$5" ]   
       then
         threshi=$5
     else
         threshi=0.25   
     fi
     neurona=$4
     ./darknet detector test $(ls $neurona/*.data) $(ls $neurona/*.cfg) $(ls $neurona/*.weights) -thresh $threshi -dont_show -save_labels < train.txt
fi
