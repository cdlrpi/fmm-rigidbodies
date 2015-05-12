# epslatex
# NOTE: thes ize is given in inches. It can also be given in cm.
set terminal cairolatex pdf size 3.5,2 dashed
set style fill transparent solid 0.05 noborder
set output "~/Dropbox/Research/Publications/journals/jmst/full-paper/figures/errorMultipole.tex"
# set terminal png transparent nocrop enhanced size 450,320 font "arial,8" 
# set output 'contours.19.png'
set key at screen 1, 0.9, 0 right top vertical Right noreverse enhanced autotitle nobox
unset key
set style textbox opaque margins  0.5,  0.5 noborder
set view map scale 1
set samples 100, 100
set isosamples 100, 100
unset surface 
set contour base
set cntrlabel start 5 interval 200
set cntrparam order 8
set cntrparam bspline
set cntrparam levels discrete 7,8,9,14
set style data lines
# set title "2D contour projection of previous plot" 
set xlabel "$\\left(r/a\\right)$" 
set xrange [0.6/0.5:3/0.5] noreverse nowriteback
set ylabel "Rotation Angle $\\left(\\theta\\right)$" 
set yrange [0:2*pi] noreverse nowriteback
set ytics ('$0$' 0,\
     '$\frac{\pi}{2}$' pi/2,\
     '$\pi$' pi,\
     '$\frac{3\pi}{2}$' 3*pi/2,\
     '$2\pi$' 2*pi)

# set zlabel "Z " 
# set zlabel offset character 0, -20, 0 font "" textcolor lt -1 norotate
# set zrange [6:16] noreverse nowriteback
# u = 0.0
# x = 0.0
## Last datafile plotted: "glass.dat"
# splot "glass.dat" using 3 with lines,\
#       "glass.dat" using 3 with labels boxed
splot "data.dat" using 1:2:4 with lines,\
      "data.dat" using 1:2:4 with labels boxed
