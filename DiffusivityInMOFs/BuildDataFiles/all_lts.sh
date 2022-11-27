ltemplify.py -name mof data.IRMOF1 > IRMOF.lt
ltemplify.py -name mol molecules.data > molecules.lt
moltemplate.sh -overlay-angles combine.lt
