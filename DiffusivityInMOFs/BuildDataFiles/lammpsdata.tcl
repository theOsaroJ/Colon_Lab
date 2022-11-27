package require topotools
topo retypebonds
topo guessangles
topo retypedihedrals
molinfo top set a 25.832
molinfo top set b 25.832
molinfo top set c 25.832
molinfo top set alpha 90
molinfo top set beta 90
molinfo top set gamma 90
topo writelammpsdata molecules.data full
