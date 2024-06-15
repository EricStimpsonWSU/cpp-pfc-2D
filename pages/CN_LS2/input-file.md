The input file for CN_LS2.c is CN_LS2.in.  The format of the file is described here.

Sample file
1. `ttt`
2. `1.0 1.0 0.1`
3. `1.0 0.0`
4. `-0.2680999637 -0.4433000386 0.3 0.3 1.0 1.0`
5. `0.5 0.5 1.0 0.5 0.3 0.3`
6. `100000 50000 50000 1000`
7. `34 400 400`
8. `9`
9. `2.220076e-01 -4.233408e-02 1.725721e-01 1.459397e-01 2.220076e-01 -4.233408e-02`
10. `-1.810067e-01 -1.353370e-01 -8.670347e-03 -2.258414e-01 -1.810067e-01 -1.353370e-01`

Variable [meaning]
1. `run [title]`
2. `dx [delata x] dy [delta y] dt [delta t]`
3. `alpha V0`
4. `ns nl epsA epsB vA vB`
5. `gA gB betaB alphaAB omega mu`
6. `nend [number of time steps] nout [output every nout time steps] neng2 neng`
7. `ntype Lx [pixels wide] Ly [pixels tall]`
8. `itheta`
9. `aa aai bb bbi cc cci` - note -> `A1o = aa + aai*I, A2o = bb + bbi*I, A3o = cc + cci*I`
10. `aa aai bb bbi cc cci` - note -> `B1o = aa + aai*I, B2o = bb + bbi*I, B3o = cc + cci*I`
