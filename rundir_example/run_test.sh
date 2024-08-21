source apcemm.env

i_sim=0
out_file=out_0001
while [[ -f $out_file ]]; do
    i_sim=$(( $i_sim + 1 ))
    printf -v out_file "out_%04d" $i_sim
    #if [[ -f $out_file ]]; then
    #    echo "WARNING\nWARNING\nWARNING\nFile $out_file already exists!"
    #    exit 101
    #fi
done
echo "Output will be stored in $out_file"
while [[ ! -f APCEMM ]]; do
    sleep 5
done
echo "Beginning simulation and outputting to $out_file"
rm APCEMM_out/*
sleep 5
time ./APCEMM input.yaml 2>&1 | tee out_test && cp out_test $out_file
