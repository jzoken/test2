echo $1
i=1
for f in `ls -t $1/*.jpg`
do
  num="$(printf "%04d" $i)"
  echo  $num
  #mv "$f" "$1/IMG_${i}.jpg"
  mv "$f" "$1/IMG_${num}.jpg"
  i=$((i+1))
done
