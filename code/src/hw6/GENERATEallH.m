function [] = GENERATEallH( H , paramSize, whichmethod )
numObs=(paramSize-5)/6;
K=[H(3) H(5) H(1);0 H(4) H(2);0 0 1;];
fn=sprintf('H_%s.txt',whichmethod);
fileH = fopen(fn,'w');
fileK = fopen('K_MATLAB.txt','w');
fileR = fopen('R_MATLAB.txt','w');
for i=1:3,
	for j=1:3,
    fprintf(fileK, '%.6f ', K(i,j));
  end
end
fclose(fileK);
for i=1:numObs,
    wx=H(i*6+0);
    wy=H(i*6+1);
    wz=H(i*6+2);
    tx=H(i*6+3);
    ty=H(i*6+4);
    tz=H(i*6+5);
    theta=sqrt(wx^2+wy^2+wz^2);
    omega=[0 -wz wy; wz 0 -wx; -wy wx 0;];
    r = eye(3) + (sin(theta)/theta)*omega + ((1-cos(theta))/theta^2)*(omega*omega);
    t = [tx;ty;tz];
    tempH = K *[r(:,1) r(:,2) t];
    tempR = [r(:,1) r(:,2) r(:,3) t];
    tempH=tempH./tempH(3,3);
    for j1=1:3,
        for j2=1:3,
            fprintf(fileH, '%.6f ', tempH(j1,j2));
        end
    end
    for j1=1:3,
        for j2=1:4,
            fprintf(fileR, '%.6f ', tempR(j1,j2));
        end
    end
    fprintf(fileH, '\n');
    fprintf(fileR, '\n');
end
fclose(fileH);
fclose(fileR);

end