close all;clear;clc
% spare representation¡ªbased classification
tic;
load('nums_person.mat');
load('nums_image');
load('face.mat');  % load face allface dataset

%initial
right=0;


for k = 1:nums_person
    for j = 1:nums_image
        close all;
        str=['person ',num2str(k),'--->','photo ',num2str(j)];
        disp(str)
        test_path =['D:\faceSRC\resource\yaleB',num2str(k),'\',num2str(2*j),'.pgm'];
        testImg = imread(test_path);
        figure,imshow(testImg);
        testImg = imresize(testImg,[12,10],'lanczos3');
        testImg = double(testImg(:));

        %L1 regression by Lasso;
        x_spare = myLASSO(dataset,testImg,nums_person*nums_image);

        % spare vector
        binTrue = zeros(1,nums_person);
        n =1;
        m =nums_image;
        for i =1:nums_person
            binTrue(:,i) = sum(x_spare(n:m,:));
            n = n+nums_image;
            m = m+nums_image;
        end

        [value,whoInd] = max(binTrue);
        if(whoInd == k)
            right=right+1;
            disp('is true')
        else
            disp('is false')
        end
    end
end

%Calculation accuracy
right_rate = right/(nums_image*nums_person)
toc;
