rem parameters:
rem -d: the input images directory
rem -dbp: output directory for back projection images
rem -rh,-rx,-rk,-rr: reading the files for H,X,K,R as input
rem -oh,-ox,-ok,-or: output files for H,X,K,R
rem -OPTLM: generate MATLAB code for performing LM in MATLAB
rem Mode 1: only generate the H,X,K,R, does not do optimization
rem ./hw6.exe -dcalimages -ohH.txt -oxX.txt -okK.txt -orR.txt
rem Mode 2: read in H,X,K,R, generate the MATLAB input for performing LM in MATLAB
rem ./hw6.exe -dcalimages -rhH.txt -rxX.txt -okK.txt -orR.txt -OPTLM
rem Mode 2: perform LM in MATLAB.
rem matlab -automation -r "cd F:\Users\Bourne\Documents\MyFiles\DevSpace\VisualStudio\ee661\Hw6\bin; NONLINEAROPTby('LM');exit;"
rem sleep 1200
rem Mode 3: read in the refined parameters from MATLAB and then output the back projection image
rem ./hw6.exe -dcalimages -dbpcalimagesbp -rkK.txt -rrR.txt -BPMH_LM.txt
rem Mode 4: using levmar to do camera calibration, output all parameters and Refined H.
./hw6.exe -dcalimages -dbpcalimagesbp -ohH.txt -oxX.txt -okK.txt -orR.txt -BPLH_LM.txt
mkdir levmar40_1
mv *.txt levmar40_1
./hw6.exe -dbbcal -dbpbbcalbp -ohH.txt -oxX.txt -okK.txt -orR.txt -BPLH_LM.txt
mkdir bbcal20
mv *.txt bbcal20
rem Mode 5: run levmar, but using existing files of H,X,K,R.
rem ./hw6.exe -dbbcal -dbpbbcalbp -rhH.txt -rxX.txt -rkK.txt -rrR.txt -BPLH_LM.txt
