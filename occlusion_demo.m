close all;clear;clc
% spare representation¡ªbased classification
tic;
m_person =20;
m_image =30;
load('face.mat');  % load face allface dataset
person_label = 4;   %1-20£¬
test_path =['D:\faceSRC\resource\occlusion_img\yaleB04_1.pgm'];
testImg = imread(test_path);
figure,imshow(testImg),title('Test');
testImg = imresize(testImg,[12,10],'lanczos3');
testImg = double(testImg(:));

%L1 regression by Lasso;
x_spare = myLASSO(dataset,testImg,m_person*m_image);

% ÖÃÐÅ¶È
binTrue = zeros(1,m_person);
n =1;
m =m_image;
for i =1:m_person
    binTrue(:,i) = sum(x_spare(n:m,:));
    n = n+m_image;
    m = m+m_image;
end
x=1:1:20;
figure,
bar(binTrue,0.5);
title("spare vector")

[value,whoInd] = max(binTrue)
if(whoInd == person_label)
   disp('is true')
else
    disp('is fault')
end
whoImg_path = ['D:\faceSRC\resource\yaleB',num2str(whoInd),'\1.pgm'];
whoImage = imread(whoImg_path);
figure,imshow(whoImage),title('Reconstruct'); %show the original person
toc;
