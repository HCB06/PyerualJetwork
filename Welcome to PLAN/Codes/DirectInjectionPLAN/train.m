function train()

class = 1;


disp('training');

do

photo = sprintf('train/%d.png',class);

img = imread(photo);

img = rgb2gray(img);

imshow(img);
title("Training Photo");

inputLayer = img(:);
inputLayer = single(inputLayer);

weights1 = ones(9,784); % Matrix formed by ones


%% FEATURE EXTRACTION LAYER %%
  
  weights1(class,:) = inputLayer;


if class ~= 1
	
		newWeights1 = weights1;
		
		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    
		
		
		weights1 += newWeights1;
		
end


		fileName1 = sprintf('weights/weights1.mat');
	save(fileName1, 'weights1');


		
pause(1); % Wait 1 sec.

class++;

until(class > 9)

disp('train finished');

disp('validation starting in 3..');

pause(3)

validate() % with train inputs
