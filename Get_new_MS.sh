echo "#x              y         n_spin     Theta(x,y)       Phi(x,y)Moment_Size(x,y)" > Theta_Phi_FM_L64.txt
for line in {2..65}
do 

x=$(awk -v row=${line} 'NR==row {print $1}' ThetaPhi_Temp0.0001000000MicroState0.txt)
y=$(awk -v row=${line} 'NR==row {print $2}' ThetaPhi_Temp0.0001000000MicroState0.txt)
n_spin=0
theta=0
#$(awk -v row=${line} 'NR==row {print $3}' ThetaPhi_Temp0.0001000000MicroState0.txt)
phi=0
#$(awk -v row=${line} 'NR==row {print $4}' ThetaPhi_Temp0.0001000000MicroState0.txt)
moment=$(awk -v row=${line} 'NR==row {print $5}' ThetaPhi_Temp0.0001000000MicroState0.txt)


echo "${x}     ${y}      ${n_spin}        ${theta}           ${phi}             ${moment}"   >> Theta_Phi_FM_L64.txt

done
