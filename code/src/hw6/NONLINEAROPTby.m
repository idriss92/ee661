function [] = NONLINEAROPTby( whichmethod )
%read from file for H
%read from file for x'
%read from file for x

%x = [1;2;3;4;5;6;7;8];
%xp = [1.1;2.1;3.1;4.1;5.1;6.1;7.1;8.1];
%H = [1;0;0.1;0;1;0.099;0;0;1];

%x = [552;111];
%xp = [508;71];
%H = [0.60181; 0.03433;-39.03425;-0.0866;0.59230;-18.77599;0.00005;0.00006;0.54907];

%H = [0.855892; 0.050585; 43.695240; -0.098573; 0.937354; 31.923266; -0.000230; 0.000033; 1.000000];
%x = [18; 404; 31; 237; 42; 364; 43; 283; 44; 187; 44; 212; 44; 261; 50; 169; 66; 122; 77; 168; 89; 126; 102; 226; 108; 176; 120; 270; 120; 310; 124; 109; 124; 225; 126; 92; 128; 204; 142; 397; 162; 264; 163; 306; 169; 156; 172; 244; 176; 315; 181; 164; 202; 315; 205; 152; 206; 241; 224; 122; 262; 316; 264; 394; 271; 263; 277; 166; 286; 259; 299; 209; 301; 193; 301; 228; 301; 345; 302; 175; 303; 162; 313; 284; 317; 308; 333; 251; 337; 305; 358; 361; 359; 377; 360; 288; 360; 326; 369; 246; 373; 289; 374; 263; 376; 394; 377; 365; 390; 358; 400; 288; 401; 370; 416; 363; 417; 382; 437; 358; 454; 278; 455; 317; 455; 357; 457; 295; 477; 365; 501; 312; 502; 354; 515; 354;];
%xp = [81; 404; 82; 251; 97; 368; 95; 293; 92; 204; 92; 226; 95; 272; 96; 187; 109; 142; 119; 184; 128; 144; 144; 237; 147; 190; 163; 278; 164; 316; 158; 125; 165; 236; 159; 109; 167; 215; 187; 399; 201; 271; 204; 311; 203; 168; 210; 252; 217; 319; 215; 173; 241; 319; 237; 161; 242; 247; 253; 130; 299; 318; 304; 395; 305; 265; 311; 170; 321; 262; 329; 213; 332; 198; 334; 230; 338; 346; 333; 177; 332; 164; 348; 284; 353; 309; 367; 251; 373; 305; 396; 361; 397; 378; 396; 287; 398; 326; 404; 244; 409; 287; 407; 262; 417; 391; 416; 363; 430; 357; 437; 286; 442; 370; 456; 361; 457; 381; 478; 356; 493; 273; 495; 313; 497; 355; 495; 291; 522; 363; 545; 306; 547; 351; 563; 351; ];

if (ischar(whichmethod))
   switch(whichmethod)
       case 'GD'
           GD;
       case 'GN'
           GN;
       case 'LM'
           LM;
       case 'DL'
           DL;
   end
end

%H = [0.845336; 0.046660; 44.722578; -0.107282; 0.931531; 33.188067; -0.000258; 0.000028; 1.000000];

%syms h11 h12 h13 h21 h22 h23 h31 h32 h33;
%h=[h11;h12;h13;h21;h22;h23;h31;h32;h33];
%syms Fp;
%ophx=@(x) [[h11,h12,h13]*[x;1]/([h31,h32,h33]*[x;1]),[h21,h22,h23]*[x;1]/([h31,h32,h33]*[x;1])];
%for i=1:size(x,1)/2,
%    Fp(2*i-1:2*i)=ophx(x(2*i-1:2*i)); 
%end
%f=sum((xp-transpose(Fp)).^2);
%jacf=jacobian(f);
%initialf = subs(f,{h},{H});
%disp(initialf);

%r=(xp-transpose(Fp));
%jacr=jacobian(r);

GENERATEJac;
initialf = NUMF(H);
disp(initialf);

deltaHThres = 0.000000001;

if (ischar(whichmethod))
    switch(whichmethod)
        case 'GD'
            deltaH = 10000;
            iter = 0;
            %alpha = 0.00001;
            while ((iter < 1000) && (deltaH > deltaHThres )),
                %numericJacf = transpose(subs(jacf,{h},{H}));
                %newH = H - alpha * numericJacf;
                numericJacr = subs(jacr,{h},{H});
                numericR = subs(r,{h},{H});
                newH = H-(sum((transpose(numericJacr)*numericR).^2,1)/sum((numericJacr*transpose(numericJacr)*numericR).^2,1))*transpose(numericJacr)*numericR;
                deltaH = sqrt(sum((newH - H).^2,1));
                disp(deltaH);
                disp(newH);
                iter = iter + 1;
                H = newH;
            end
            disp(H);
        case 'GN'
            deltaH = 10000;
            iter = 0;
            while ((iter < 1000) && (deltaH > deltaHThres)),
                numericJacr = subs(jacr,{h},{H});
                numericR = subs(r,{h},{H});
                numericJacrTJacr = transpose(numericJacr)*numericJacr;
                %if (det(numericJacrTJacr) < 0.0001)
                %    disp('Non-Invertable!!!!!!!!!!!\n');
                %    return
                %end
                newH = H - inv(numericJacrTJacr+0.001*eye(9,9))*transpose(numericJacr)*numericR;
                deltaH = sqrt(sum((newH - H).^2,1));
                disp(deltaH);
                disp(newH);
                iter = iter + 1;
                H = newH;
            end
            disp(H/H(9));
        case 'LM'
            deltaH = 10000;
            distance = inf;
            distThres = 20;
            iter = 0;
            tau = 0.001;
            numericJacr = NUMJACR(H);
            %numericJacr = NUMJACR(H(1), H(2), H(3), H(4), H(5), H(6), H(7), H(8), H(9), H(10), H(11), H(12), H(13), H(14), H(15), H(16), H(17), H(18), H(19), H(20), H(21), H(22), H(23));%subs(jacr,{h},{H});
            mu = tau * max(diag(transpose(numericJacr)*numericJacr));
            while ((iter < 1000) && (deltaH > deltaHThres)),
            %while ((iter < 1000) && distance > distThres),
                %numericJacr = NUMJACR(H(1), H(2), H(3), H(4), H(5), H(6), H(7), H(8), H(9), H(10), H(11), H(12), H(13), H(14), H(15), H(16), H(17), H(18), H(19), H(20), H(21), H(22), H(23));%subs(jacr,{h},{H});
                %numericR = NUMR(H(1), H(2), H(3), H(4), H(5), H(6), H(7), H(8), H(9), H(10), H(11), H(12), H(13), H(14), H(15), H(16), H(17), H(18), H(19), H(20), H(21), H(22), H(23));%subs(r,{h},{H});
                %numericJacrTJacr = transpose(numericJacr)*numericJacr;
                %numericfk = NUMF(H(1), H(2), H(3), H(4), H(5), H(6), H(7), H(8), H(9), H(10), H(11), H(12), H(13), H(14), H(15), H(16), H(17), H(18), H(19), H(20), H(21), H(22), H(23));%subs(f,{h},{H});
                numericJacr = NUMJACR(H);%subs(jacr,{h},{H});
                numericR = NUMR(H);%subs(r,{h},{H});
                numericJacrTJacr = transpose(numericJacr)*numericJacr;
                numericfk = NUMF(H);%subs(f,{h},{H});
                %if (det(numericJacrTJacr) < 0.0001)
                %    disp('Non-Invertable!!!!!!!!!!!\n');
                %    return
                %end
                newH = H - inv(numericJacrTJacr+mu*eye(paramSize,paramSize))*transpose(numericJacr)*numericR;
                %numericfkp1 = NUMF(newH(1), newH(2), newH(3), newH(4), newH(5), newH(6), newH(7), newH(8), newH(9), newH(10), newH(11), newH(12), newH(13), newH(14), newH(15), newH(16), newH(17), newH(18), newH(19), newH(20), newH(21), newH(22), newH(23));%subs(f,{h},{newH});
                numericfkp1 = NUMF(newH);%subs(f,{h},{newH});
                rho = (numericfk-numericfkp1)/(transpose(newH-H)*(mu*(newH-H)-transpose(numericJacr)*numericR));
                while (rho <= 0)
                    mu = 2 * mu;
                    newH = H - inv(numericJacrTJacr+mu*eye(paramSize,paramSize))*transpose(numericJacr)*numericR;
                    %numericfkp1 = NUMF(newH(1), newH(2), newH(3), newH(4), newH(5), newH(6), newH(7), newH(8), newH(9), newH(10), newH(11), newH(12), newH(13), newH(14), newH(15), newH(16), newH(17), newH(18), newH(19), newH(20), newH(21), newH(22), newH(23));%subs(f,{h},{newH});
                    numericfkp1 = NUMF(newH);%subs(f,{h},{newH});
                    rho = (numericfk-numericfkp1)/(transpose(newH-H)*(mu*(newH-H)-transpose(numericJacr)*numericR));
                end
                mu = mu * max(1.0/3,1-(2*rho-1)^3);
                deltaH = sqrt(sum((newH - H).^2,1));
                disp(deltaH);
                disp(newH);
                iter = iter + 1;
                H = newH;
                distance = sqrt(NUMF(newH));
            end
            disp(H);
            %levmar(x,xp,H);
        case 'DL'
            deltaH = 10000;
            iter = 0;
            deltaK = 1;
            while ((iter < 1000) && (deltaH > deltaHThres)),
                numericJacr = subs(jacr,{h},{H});
                numericR = subs(r,{h},{H});
                numericJacrTJacr = transpose(numericJacr)*numericJacr;
                %if (det(numericJacrTJacr) < 0.0001)
                %    disp('Non-Invertable!!!!!!!!!!!\n');
                %    return
                %end
                newHGD=H-(sum((transpose(numericJacr)*numericR).^2,1)/sum((numericJacr*transpose(numericJacr)*numericR).^2,1))*transpose(numericJacr)*numericR;
                newHGN=H-inv(numericJacrTJacr+0.01*eye(9,9))*transpose(numericJacr)*numericR;
                deltaHGD=newHGD-H;
                deltaHGN=newHGN-H;
                if (sqrt(sum(deltaHGN.^2))<= deltaK),
                    newH = H + deltaHGN;
                elseif (sqrt(sum(deltaHGD.^2))<= deltaK && sqrt(sum(deltaHGN.^2)) > deltaK),
                    a = sum((deltaHGN-deltaHGD).^2);
                    b = 2*transpose(deltaHGD)*(deltaHGN-deltaHGD);
                    c = sum((deltaHGD).^2)-deltaK^2;
                    beta = (-b+sqrt(b^2-4*a*c))/(2*a);
                    newH = H + beta * (deltaHGN-deltaHGD);
                else
                    newH = H + deltaK * deltaHGD / (sqrt(sum(deltaHGD.^2)));
                end
                
                numericfk = subs(f,{h},{H});
                numericfkp1 = subs(f,{h},{newH});
                rho = (numericfk-numericfkp1)/(-transpose(newH-H)*transpose(numericJacr)*numericJacr*(newH-H)-2*transpose(numericR)*numericJacr*(newH-H));
                if (rho < 1.0/4),
                    deltaK = deltaK / 4.0;
                elseif (rho <= 0.75 && rho >= 0.25),
                    pass;%deltaK = deltaK;
                else
                    deltaK = 2*deltaK;
                end
                
                if (rho <= 0 )
                    continue;
                end
                deltaH = sqrt(sum((newH - H).^2,1));
                disp(deltaH);
                disp(newH);
                iter = iter + 1;
                H = newH;
            end
            disp(H);
            %dogleg(x,xp,H);
    end
    
    %finalf = subs(f,{h},{H});
    finalf=NUMF(H);
    disp(finalf);
    GENERATEallH(H,paramSize,whichmethod);
end

end
