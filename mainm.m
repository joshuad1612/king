
clc
clear all
close all

%Image Acquisition
[aa bb]=uigetfile('.jpg');
I=imread([bb aa]);
figure,imshow(I);
title('Input Image');

%Pre processing
I1=imresize(I,[256,256]);
figure,imshow(I1);
title('resized Image');

%ADDING NOISE
In=imnoise(I1,'salt & pepper');
figure,imshow(In),title('salt&peper')

%NOISE REMOVAL
J=In;
winsz=3;
[s1, s2]=size(In);
sz=min([s1 s2]);
p=(winsz-1)/2;
if size(J,3)>1
for mnpk=1:3
    ipnoise=J(:,:,mnpk);
medop=padarray(ipnoise,[p p]);
for i=1+p:sz+p
    for j=1+p:sz+p
        submed=medop(i-p:i+p,j-p:j+p);
        medop(i,j)=median(submed(1:9));
    end
    end

medop=medop(1+p:sz+p,1+p:sz+p);
 op1(:,:,mnpk)=medop;
end
else
   ipnoise=J;
medop=padarray(ipnoise,[p p]);
for i=1+p:sz+p
    for j=1+p:sz+p
        submed=medop(i-p:i+p,j-p:j+p);
        medop(i,j)=median(submed(1:9));
    end
    end

medop=medop(1+p:sz+p,1+p:sz+p);
 op1=medop;
end
figure,imshow(op1);
title('Noise Removal');

%Scale Invarient Feature Transformation
%Ig=rgb2gray(I1);
Iad=adapthisteq(I1(:,:,2));
figure,imshow(Iad);
title('Contrast Enhanced Image');
[feature, M]=sift(Iad);

%Blood Vessel Extraction
dim = ndims(Iad);
if(dim == 3)
    %Input is a color image
    Iad = rgb2gray(Iad);
end

%Mask-based Blood Vessels Extraction with global image threshold
B = imresize(I, [256 256]);
% Read image
im = im2double(B);
% Convert RGB to Gray via PCA
cform = makecform('srgb2lab');
        lab = applycform(im, cform);
f = 0;
wlab = reshape(bsxfun(@times,cat(3,1-f,f/2,f/2),lab),[],3);
[C,S] = pca(wlab);
S = reshape(S,size(lab));
S = S(:,:,1);
gray = (S-min(S(:)))./(max(S(:))-min(S(:)));

% Contrast Enhancment of gray image using CLAHE
J = adapthisteq(gray,'numTiles',[8 8],'nBins',128);

% Background Exclusion
% Apply Average Filter
h = fspecial('average', [9 9]);
JF = imfilter(J, h);
figure, imshow(JF)
title('Filtered Image');
% Take the difference between the gray image and Average Filter
Z = imsubtract(JF, J);
figure, imshow(Z);
title('Estimate Diffenece');
% Threshold using the IsoData Method
level=isodata(Z) % this is our threshold level
%level = graythresh(Z)
% Convert to Binary
BW = im2bw(Z, level-.008);
% Remove small pixels
BW2 = bwareaopen(BW, 100);
% Overlay
BW2 = imcomplement(BW2);
out = imoverlay(B, BW2, [0 0 0]);
figure, imshow(out);
title('Vessel Extracted Image');

%Segmentation using Expectation Maximisation
K    = 5;   
I1=rgb2gray(I1);
[mask,mu,v,p]=Seg(I1,K);
figure,imshow(mask,[]);
impixelinfo

[s1,s2]=size(mask);
for i=1:s1
    for j=1:s2
        if mask(i,j)==4
            A(i,j)=1;
        else
            A(i,j)=0;
        end
    end
end

figure,imshow(A,[]);
title('Extracted Part');

%    feature extraction
% Statiscal Features
 IB=A;
 m=mean(mean(IB));
 s=std(std(IB));
 v=var(var(IB));

Stat_fea=mean([m s v])

% Texture Based feature
new = graycomatrix(IB,'Offset',[2 0]);
[out] = glcmfeature(new,0);
F1=out.contr;
F2=out.energ;
F3=out.homop;
F4=out.entro;
im=[F1 F2 F3 F4];
tex_fea=mean(im)

%Shape Based Feature
%Label the image
[Label1,Total]=bwlabel(IB,8);
%Object Number
num=1;
[row, col] = find(Label1==num);

%Find Area
Obj_area1=numel(row);

%Find Centroid
X1=mean(col);
Y1=mean(row);
Centroid1=[X1 Y1];

%Find Perimeter
BW=bwboundaries(Label1==num);
c=cell2mat(BW(1));
Perimeter1=0;
for i=1:size(c,1)-1
Perimeter1=Perimeter1+sqrt((c(i,1)-c(i+1,1)).^2+(c(i,2)-c(i+1,2)).^2);
end

%Find Equivdiameter
pi=3.14;
EquivD1=sqrt(4*(Obj_area1)/pi);

%Find Roundness
Roundness1=(4*Obj_area1*pi)/Perimeter1.^2;

%Calculation with 'regionprops'
s  = regionprops(c, 'FilledArea');
 
Sdata=regionprops(Label1,'all');
Ecc1=Sdata(num).Eccentricity;

Shape_Fea=[Obj_area1 Centroid1 Perimeter1 EquivD1 Ecc1];
Sha_fea=mean(mean(Shape_Fea))
FEAT=[Stat_fea tex_fea Sha_fea];
Fea= (mean(FEAT))/1000

%   save f12 Fea12

load netan

%  %Classification using ANN

 out = round(abs((sim(netan,Fea))))
 if out==1 
     msgbox('Normal')
 elseif out==2
     msgbox('Retinal Tear')
 elseif out==3
     msgbox('Retinal Detachment')
 else 
     msgbox('Microaneurysm')
 end
%  
















