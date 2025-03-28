
#vertical
#Lm1=7;arrow_offset=$(echo "1" | bc -l);for i in {0..7};do arrow=$(echo "${arrow_offset}+${i}" | bc -l);x1=$(echo "0.0*${i}" | bc -l);y1=$(echo "${i}" | bc -l);x2=$(echo "${x1}+${Lm1}+1" | bc -l);y2=${y1};echo "set arrow ${arrow} from ${y1},${x1} to ${y2},${x2} lw 1.5 dt 3 lc "black" nohead front" ;done
set arrow 1 from 0,0 to 0,8 lw 1.5 dt 3 lc black nohead front
set arrow 2 from 1,0 to 1,8 lw 1.5 dt 3 lc black nohead front
set arrow 3 from 2,0 to 2,8 lw 1.5 dt 3 lc black nohead front
set arrow 4 from 3,0 to 3,8 lw 1.5 dt 3 lc black nohead front
set arrow 5 from 4,0 to 4,8 lw 1.5 dt 3 lc black nohead front
set arrow 6 from 5,0 to 5,8 lw 1.5 dt 3 lc black nohead front
set arrow 7 from 6,0 to 6,8 lw 1.5 dt 3 lc black nohead front
set arrow 8 from 7,0 to 7,8 lw 1.5 dt 3 lc black nohead front

# horizontal
#Lm1=7;arrow_offset=$(echo "${Lm1}+2" | bc -l);for i in {0..7};do arrow=$(echo "${arrow_offset}+${i}" | bc -l);x1=$(echo "0.0*${i}" | bc -l);y1=$(echo "${i}" | bc -l);x2=$(echo "${x1}+${Lm1}+1" | bc -l);y2=${y1};echo "set arrow ${arrow} from ${x1},${y1} to ${x2},${y2} lw 1.5 dt 3 lc "black" nohead front" ;done
set arrow 9 from 0,0 to 8,0 lw 1.5 dt 3 lc black nohead front
set arrow 10 from 0,1 to 8,1 lw 1.5 dt 3 lc black nohead front
set arrow 11 from 0,2 to 8,2 lw 1.5 dt 3 lc black nohead front 
set arrow 12 from 0,3 to 8,3 lw 1.5 dt 3 lc black nohead front
set arrow 13 from 0,4 to 8,4 lw 1.5 dt 3 lc black nohead front
set arrow 14 from 0,5 to 8,5 lw 1.5 dt 3 lc black nohead front
set arrow 15 from 0,6 to 8,6 lw 1.5 dt 3 lc black nohead front
set arrow 16 from 0,7 to 8,7 lw 1.5 dt 3 lc black nohead front




set xr [-0.:8.]
set yr [-0.:8.]
#set zr [-2:2]

set ticslevel 0

set palette define (0 "white", 1 "blue")
#unset cbr
#rgb(r,g,b) = r
#int(r)*65536 + int(g)*256 + int (b)
#set cbr [-1:-1]
#set palette rgbformulae 33,13,10


set pm3d map

set cbr [0.6:0.8]
set pm3d corners2color c1


sp "ThetaPhi_Temp0.0001000000MicroState0.txt" u ($1+0.5):($2+0.5):(0):($5*cos($3)*0.50):($5*sin($3)*cos($4)*0.50):(0) w vec head size 0.2,30,60 filled lw 2 lc "black" ti ""
#sp "/home/nitin/Desktop/Telperion/data/home/n01/Spin_Fermion_Cuprates_June17_2020//PBC_Lx8_Ly8_CoolingED/Jhund2.0_fill0.312500_seed321/Local_den_Temp0.0001000000MicroState10.txt" u 1:2:($3+$4)  w pm3d ti "", "/home/nitin/Desktop/Telperion/data/home/n01/Spin_Fermion_Cuprates_June17_2020//PBC_Lx8_Ly8_CoolingED/Jhund2.0_fill0.312500_seed321/ThetaPhi_Temp0.0001000000MicroState10.txt" u ($1+0.5):($2+0.5):(0):($5*cos($3)*0.50):($5*sin($3)*sin($4)*0.50):(0) w vec head size 0.2,30,60 filled lw 2 lc "black" ti ""
#sp "/home/nitin/Documents/Codes/Monte_Carlo_TCA_1orb/Local_den_Temp0.0100000000MicroState1.txt" u 1:2:($3+$4)  w pm3d ti "", "/home/nitin/Documents/Codes/Monte_Carlo_TCA_1orb/ThetaPhi_Temp0.0100000000MicroState1.txt" u ($1+0.5):($2+0.5):(0):($5*cos($3)*0.312500 ):($5*sin($3)*sin($4)*0.50):(0) w vec head size 0.2,30,60 filled lw 2 lc "black" ti ""

