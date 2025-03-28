awk '$3==2' Local_den_Temp0.0050MicroState0.txt > check.txt ; awk ' {print;} NR % 5 == 0 { print ""; }' check.txt > check2.txt
