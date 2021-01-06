clc;
clear all;
close all;
img=imread('1.jpg');
img=rgb2gray(img);
[m,n]=size(img);
te=zeros(256);

for i=1:m
    for j=1:n
      k=img(i,j);
    if k~=0
    sa=1+te(k);
    te(k)=sa;
        end
    %img(i,j)
    end
end     
num=1;
for i=1:256
   if te(i)~=0
    b(num)=i;
    a(num)=te(i);
    num=num+1;
   end
end
b = [1 :256];

% Empty Array of Object Huffman
thearray = Huffman.empty(256,0);

% Assign Initial Values
for i=1:length(a)
    thearray(i).probability = a(i);
    thearray(i).character = b(i);
end

temparray = thearray;

% Create the Binary Tree
for k = 1:size(temparray,2)-1

    % First Sort the temp array

    for i=1:size(temparray,2)
        for j = 1:size(temparray,2)-1
            if (temparray(j).probability > temparray(j+1).probability)
                tempnode = temparray(j);
                temparray(j) = temparray(j+1);
                temparray(j+1) = tempnode;
            end
        end
    end

    % Create a new node 

    newnode = Huffman;

    % Add the probailities
    newnode.probability = temparray(1).probability + temparray(2).probability;

    % Add Codes
     temparray(1).code = '0';
     temparray(2).code = '1';

    % Attach Chlldren Nodes
    newnode.leftNode = temparray(1);
    newnode.rightNode = temparray(2);

    % Delete the first two nodes

     temparray = temparray(3:size(temparray,2));

    % Prepend the new node

     temparray = [newnode temparray];

end

rootNode = temparray(1);
codec = '';
astr=string(a);
% Looping though the tree
% See recursive function loop.m
global di;
di=dic.empty(256,0);
p1=0;
loop(rootNode,codec);
aft='';
for i=1:m
    for j=1:n
        a=img(i,j);
        st=di(4).code;
        aft=[aft st];
    end
end
disp(aft)
global ans;
ans='';
yima(rootNode,rootNode,0);
imshow(img);