# epslatex
# NOTE: thes ize is given in inches. It can also be given in cm.
set terminal cairolatex pdf size 3.5,2 dashed
set style fill transparent solid 0.05 noborder
set output "~/Dropbox/Research/Publications/journals/jmst/full-paper/figures/potential.tex"


# unset key 
# # set view map scale 1
# set pm3d map
# set xlabel "R/L"
# set ylabel "theta" 
# set cblabel "Error" 
# # set palette rgbformulae 8,8,8
# set palette defined ( 0 '#000090',\
#                       1 '#000fff',\
#                       2 '#0090ff',\
#                       3 '#0fffee',\
#                       4 '#90ff70',\
#                       5 '#ffee00',\
#                       6 '#ff7000',\
#                       7 '#ee0000',\
#                       8 '#7f0000')
# set palette maxcolors 12
# set autoscale fix
# splot 'potential.dat' matrix with image

reset
f(x,y)=sin(1.3*x)*cos(.9*y)+cos(.8*x)*sin(1.9*y)+cos(y*.2*x)
set xrange [-5:5]
set yrange [-5:5]
set isosample 250, 250
set table 'test.dat'
splot f(x,y)
unset table

set contour base
set cntrparam level incremental -3, 0.5, 3
unset surface
set table 'cont.dat'
splot f(x,y)
unset table

reset
set xrange [-5:5]
set yrange [-5:5]
unset key
set palette rgbformulae 33,13,10
p 'test.dat' with image, 'cont.dat' w l lt -1 lw 1.5
