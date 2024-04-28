mol load xyz molecules.xyz
package require topotools
topo retypebonds
topo guessangles
topo retypedihedrals
molinfo top set a 26.343
molinfo top set b 26.343
molinfo top set c 26.343
molinfo top set alpha 90
molinfo top set beta 90
molinfo top set gamma 90
topo writelammpsdata molecules.data full
