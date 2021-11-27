close all;clear;clc
% spare representation¡ªbased classification
tic;
load('nums_person.mat');
load('nums_image');
load('face.mat');  % load face allface dataset

% select test person 
person_label = 10;   %1-20£¬
right = 0;

for j = 1:nums_image
    close all;
    j
    test_path =['D:\faceSRC\resource\yaleB',num2str(person_label),'\',num2str(2*j),'.pgm'];
    testImg = imread(test_path);
    figure,imshow(testImg),title('Test');
    testImg = imresize(testImg,[12,10],'lanczos3');
    testImg = double(testImg(:));

    %L1 regression by Lasso;
    x_spare = myLASSO(dataset,testImg,nums_person*nums_image);

    % spare vector
    binTrue = zeros(1,nums_person);
    n =1;
    m =nums_image;
    for j =1:nums_person
        binTrue(:,j) = sum(x_spare(n:m,:));
        n = n+nums_image;
        m = m+nums_image;
    end

    [value,whoInd] = max(binTrue);
    if(whoInd == person_label)
        right=right+1;
        disp('is true')
    else
        disp('is false')
    end
end
right_rate = right/30
toc;
