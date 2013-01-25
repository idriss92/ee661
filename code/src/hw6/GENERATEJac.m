f=sum((xp-transpose(Fp)).^2);
r=(xp-transpose(Fp));
jacr=jacobian(r);

fid = fopen('NUMJACR.m','w');
fprintf(fid,'function [returnValue] = NUMJACR( H )\n');
fprintf(fid,'%s\n',hdeftext);
textff=char(jacr);
fprintf(fid,'returnValue=%s;\n',textff(8:size(textff,2)-1));
fprintf(fid,'%s;\n',paramtext);
fprintf(fid,'returnValue=reshape(returnValue,size(returnValue,2)/paramSize,paramSize);\n');
fclose(fid);

fid1 = fopen('NUMR.m','w');
fprintf(fid1,'function [returnValue] = NUMR( H )\n');
fprintf(fid1,'%s\n',hdeftext);
textff=char(r);
fprintf(fid1,'returnValue=%s;\n',textff(8:size(textff,2)-1));
fprintf(fid1,'returnValue=reshape(returnValue,size(returnValue,2),1);\n');
fclose(fid1);

fid2 = fopen('NUMF.m','w');
fprintf(fid2,'function [returnValue] = NUMF( H )\n');
fprintf(fid2,'%s\n',hdeftext);
fprintf(fid2,'returnValue=%s;\n',char(f));
fclose(fid2);