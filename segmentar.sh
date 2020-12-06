#!/bin/bash
if [ -z "$2" ]
  then
    salida=videossegmentados
  else
    salida=$2
    if [ -z "$3" ]
      then
       frames=10
    else
       frames=$3
    fi
fi
if [ -d $salida ]
then
    echo "error $salida ya existe"
else
	mkdir $salida
	for entry in $1/*.avi $1/*.mp4;
	do
	  echo $entry
	  mkdir $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')
	  Yolo_mark/yolo_mark $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.') cap_video $entry $frames
	  ls $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg>>$salida/train.txt
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
	     ./darknet detector test $(ls $neurona/*.data) $(ls $neurona/*.cfg) $(ls $neurona/*.weights) -thresh $threshi -dont_show -save_labels < $salida/train.txt
	     if [ ! -z "$6" ]
               then
                  	mkdir $salida/basura
			mkdir $salida/negativos
			while read archivo; do
			  echo "$archivo"
			  archivo2=$(echo $archivo | sed -e "s/.jpg/.txt/g")
			  s=$(wc -l < $archivo2)
			  echo $s
			  if  [ $s -eq 0 ];
			  then
			      echo "es  0"
			      mv $archivo $salida/negativos     
      			      mv $archivo2 $salida/negativos  

			  else
			   if  [ $s -lt $6 ];
			      then 
			   echo "es  menor que $6"
 			      mv $archivo $salida/basura     
      			      mv $archivo2 $salida/basura  


			  fi
			  fi

			done <$salida/train.txt

             fi

	fi

fi



