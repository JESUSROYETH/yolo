#!/bin/bash
#sh segmentar dondevideos dondeguardar cuantosframes neuronaparapretiquetar treshdeneurona cuantasespeciesenimagenes guardarpreetiquetado guardaramazon 
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
          echo "$(basename $entry)" | cut -f 1 -d '.'
          if [ -e "$entry" ]
          then
          mkdir $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')
	  Yolo_mark/yolo_mark $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.') cap_video $entry $frames
	  ls $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg>>$salida/train.txt
          fi
	  
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
	     if [ ! -z "$7" ]
               then
               preetiquetar=$7
	     else
		preetiquetar=0
             fi 
          
        mkdir $salida/negativos2
          #cuantos=$(ls salida/*/*.jpg --ignore="salida/negativo*/*.jpg")
	  #shuf -zn8 -e salida/negativos/*.jpg | xargs -0 mv -vt salida/negativos2/
 	cuantos=$(ls $salida/*/*.jpg | wc -l)
	cuantos2=$(ls $salida/negativos*/*.jpg | wc -l)
	cuantos3=$(ls $salida/basura*/*.jpg | wc -l)
	m=$(($cuantos-$cuantos2-$cuantos3))
	echo $m
	#m=10
	echo $cuantos2
	shuf -zn$m -e $salida/negativos/*.jpg | xargs -0 mv -vt $salida/negativos2/
	for f in $salida/negativos2/*.jpg; do
	    mv $salida/negativos/$(basename -- "$f" | cut -f 1 -d '.').txt $salida/negativos2/
	done

          for entry in $1/*.avi $1/*.mp4;
		do
		  echo $entry
                  if [ -e "$entry" ]
                  then
	          echo $entry
          	  echo "$(basename $entry)" | cut -f 1 -d '.'
		  archivos=$salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg
                  cantidad=$(ls $archivos | wc -l)
                  echo $cantidad
                  if  [ ! $cantidad -eq 0 ];
		    then
                  contar=0
                  for textos in $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.txt ;
                  do
                  contar=$(($contar + $(wc -l < $textos)))   
                  done  
                   if  [ $preetiquetar -eq 1 ]; 
		   then      
		       zip $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')__"$contar"__"$cantidad".zip $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.txt           
		   else
                      zip $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')__"$contar"__"$cantidad".zip $salida/$(echo "$(basename $entry)" | cut -f 1 -d '.')/*.jpg
	 
		   fi
                   fi      
                     
                  fi
                  
		  
	     done
	     if  [ $preetiquetar -eq 1 ]; 
		   then      
		       zip $salida/negativos$m.zip $salida/negativos2/*.jpg $salida/negativos2/*.txt           
		   else
                      zip $salida/negativos$m.zip $salida/negativos2/*.jpg
	 
		   fi
               
             if [ ! -z "$8" ]
               then
             m=$(date +'%F-%H%M%S')
               aws s3api put-object --bucket ftp-lythium --key "$salida"__$m/
               aws s3 cp "$salida"/ s3://ftp-lythium/"$salida"__$m/ --recursive --exclude "*" --include "*.zip" --acl public-read
             fi             
            
             

	fi

fi



