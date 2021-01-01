classdef app1_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure     matlab.ui.Figure
        File         matlab.ui.container.Menu
        open         matlab.ui.container.Menu
        save         matlab.ui.container.Menu
        transform    matlab.ui.container.Menu
        rotate       matlab.ui.container.Menu
        Move         matlab.ui.container.Menu
        Menu_17      matlab.ui.container.Menu
        Menu_18      matlab.ui.container.Menu
        Menu_19      matlab.ui.container.Menu
        Menu_32      matlab.ui.container.Menu
        RGB2gray     matlab.ui.container.Menu
        Menu_21      matlab.ui.container.Menu
        Menu_31      matlab.ui.container.Menu
        noise        matlab.ui.container.Menu
        Menu_15      matlab.ui.container.Menu
        avgFilter    matlab.ui.container.Menu
        saltPepper   matlab.ui.container.Menu
        gaussian     matlab.ui.container.Menu
        Menu_20      matlab.ui.container.Menu
        SobelMenu    matlab.ui.container.Menu
        robertsMenu  matlab.ui.container.Menu
        prewittMenu  matlab.ui.container.Menu
        Menu_23      matlab.ui.container.Menu
        Menu_24      matlab.ui.container.Menu
        OtsuMenu     matlab.ui.container.Menu
        Menu_27      matlab.ui.container.Menu
        Menu_34      matlab.ui.container.Menu
        PriwittMenu  matlab.ui.container.Menu
        RobertMenu   matlab.ui.container.Menu
        SobelMenu_2  matlab.ui.container.Menu
        Menu_28      matlab.ui.container.Menu
        Menu_30      matlab.ui.container.Menu
        Menu_29      matlab.ui.container.Menu
        Menu_33      matlab.ui.container.Menu
        UITable      matlab.ui.control.Table
        Menu_25      matlab.ui.container.Menu
        Laplace      matlab.ui.container.Menu
        Menu_26      matlab.ui.container.Menu
        ImageAxes    matlab.ui.control.UIAxes
        GreenAxes    matlab.ui.control.UIAxes
        BlueAxes     matlab.ui.control.UIAxes
        RedAxes      matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        Image % the Processing Image
    end
    
    methods (Access = private)
        
        function updateImage(app,imagefile)
                try
                    app.Image = imread(imagefile);
                catch ME
                    % If problem reading image, display error message
                    uialert(app.UIFigure, ME.message, 'Image Error');
                    return;
                end
        end
        
        function showHist(app,~)
            h = ntrop(app, app.Image);
            app.UITable.Data = (h);
            % Create histograms based on number of color channels
            switch size(app.Image,3)
                case 1
                    % Display the grayscale image
                    %imagesc(app.ImageAxes,im);
                    imshow(app.Image,'Parent',app.ImageAxes);
                    
                    % Plot all histograms with the same data for grayscale
                    histr = histogram(app.RedAxes, app.Image, 'FaceColor',[1 0 0],'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, app.Image, 'FaceColor',[0 1 0],'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, app.Image, 'FaceColor',[0 0 1],'EdgeColor', 'none');
                    
                case 3
                    % Display the truecolor image
                    %imagesc(app.ImageAxes,im);
                    imshow(app.Image,'Parent',app.ImageAxes);
                    
                    % Plot the histograms
                    histr = histogram(app.RedAxes, app.Image(:,:,1), 'FaceColor', [1 0 0], 'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, app.Image(:,:,2), 'FaceColor', [0 1 0], 'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, app.Image(:,:,3), 'FaceColor', [0 0 1], 'EdgeColor', 'none');
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    %uialert(app.UIFigure, 'Image must be grayscale or truecolor.', 'Image Error');
                    warndlg('图片格式错误。','ERROR');
                    return;
            end
            % Get largest bin count
            maxr = max(histr.BinCounts);
            maxg = max(histg.BinCounts);
            maxb = max(histb.BinCounts);
            maxcount = max([maxr maxg maxb]);
            
            % Set y axes limits based on largest bin count
            app.RedAxes.YLim = [0 maxcount];
            app.RedAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
            app.GreenAxes.YLim = [0 maxcount];
            app.GreenAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
            app.BlueAxes.YLim = [0 maxcount];
            app.BlueAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
        end
        
        function d = midfilt(~, x, n)
            [M, N] = size(x);
            x1 = x;
            x2 = x1;
            for i = 1:M-n+1
                for j = 1:N-n+1
                    c = x1(i:i+n-1,j:j+n-1);
                    e = c(1, :);
                    for k = 2:n
                        e = [e, c(k, :)];
                    end
                    x2 (i+(n-1)/2, j+(n-1)/2) = median(e);
                end
            end
            d = x2;
        end
        
        function d = avg_filter(~, x, n)
            a(1:n,1:n) = 1;
            [height, width] = size(x);
            x1 = double(x);
            x2 = x1;
            for i = 1:height-n+1
                for j=1:width-n+1
                    c=x1(i:i+(n-1),j:j+(n-1)).*a;
                    s=sum(sum(c));
                    x2(i+(n-1)/2,j+(n-1)/2)=s/(n*n);
                end
            end
            d = uint8(x2);
        end
        
        
        function d = saltPepperNoise(~,I)
            switch size(I,3)
                case 1
                    % Display the grayscale image
                    [width,height]=size(I);
                    result2 = I;
                    k1=0.1;
                    k2=0.1;
                    a1=rand(width,height)<k1;
                    a2=rand(width,height)<k2;
                    t1=result2(:,:,1);
                    t1(a1&a2)=0;
                    t1(a1& ~a2)=255;
                    result2(:,:,1)=t1;
                    d = result2;
                case 3
                    % the truecolor image
                    [width, height,~] = size(I);
                    result2 = I;
                    k1 = 0.1;
                    k2 = 0.1;
                    a1 = rand(width,height) < k1;
                    a2 = rand(width,height) < k2;
                    t1 = result2(:,:,1);
                    t2 = result2(:,:,2);
                    t3 = result2(:,:,3);
                    t1(a1&a2) = 0;
                    t2(a1&a2) = 0;
                    t3(a1&a2) = 0;
                    t1(a1& ~a2)=255;
                    t2(a1& ~a2)=255;
                    t3(a1& ~a2)=255;
                    result2(:,:,1)=t1;
                    result2(:,:,2)=t2;
                    result2(:,:,3)=t3;
                    d = result2;
                otherwise
                    % Error when image is not grayscale or truecolor
                    %uialert(app.UIFigure, '图片格式错误。', 'ERROR');
                    warndlg('图片格式错误。','ERROR');
                    d = f;
                    return;
            end
        end
        
        function d = gaussianNoise(~,I)
            [m,n,~]=size(I);
            y=0+0.1*randn(m,n);%二维高斯分布矩阵 0是均值 0.1是标准差
            %先将其double化，再除以255 便于后面计算
            K=double(I)/255;
            %加上噪声
            K=K+y;
            %将像素范围扩大至0--255
            K=K*255;
            %转换为uint8类型
            d=uint8(K);
        end
        
        function g = CustomRotate(~, f, angle)
            [h,w,~]=size(f);
            radian=angle/180*pi;
            cosa=cos(radian);
            sina=sin(radian);
            w2=round(abs(cosa)*w+h*abs(sina));
            h2=round(abs(cosa)*h+w*abs(sina));
            g=uint8(zeros(h2,w2,3));
            for x=1:w2
                for y=1:h2
                    x0=uint32(x*cosa+y*sina-0.5*w2*cosa-0.5*h2*sina+0.5*w);
                    y0=uint32(y*cosa-x*sina+0.5*w2*sina-0.5*h2*cosa+0.5*h);
        
                    x0=round(x0);
                    y0=round(y0);
                    if x0>0 && y0>0 && w>=x0 && h>=y0
                        g(y,x,:)=f(y0,x0,:);
                    end
                end
            end
        end
        
        function g = CustomRGB2gray(~,f)
            n = double(f);
            
            switch size(f,3)
                case 1
                    % Display the grayscale image
                    warndlg('请使用RGB图像进行灰度处理。','WARNING');
                    g = f;
                case 3
                    % the truecolor image
                    [x, y, ~] = size(f);
                    R = n(:,:,1);
                    G = n(:,:,2);
                    B = n(:,:,3);
                    g = zeros(x,y);
                    for i = 1 : x
                        for j = 1 : y
                            g(i,j)=R(i,j)*0.2125+G(i,j)*0.7154+B(i,j)*0.0721;
                        end
                    end
                    g = uint8(g);
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    %uialert(app.UIFigure, '图片格式错误。', 'ERROR');
                    warndlg('图片格式错误。','ERROR');
                    g = f;
                    return;
            end
        end
        
        function g = CustomReverse(~,f)
%             [M, N] = size(f);
%             I = f;
%             for i = 1 : M
%                 for j = 1 : N
%                     I(i, j) = f(i, N - j + 1);
%                 end
%             end
%             g = I;

            img = f;
            [H,W,Z] = size(img); % 获取图像大小
            I=im2double(img);%将图像类型转换成双精度
            res = ones(H,W,Z); % 构造结果矩阵。每个像素点默认初始化为1（白色）
            tras = [1 0 0; 0 -1 W; 0 0 1]; % 水平镜像的变换矩阵
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%将每一点的位置进行缓存
                    temp = tras * temp; % 根据算法进行，矩阵乘法：转换矩阵乘以原像素位置
                    x1 = temp(1, 1);%新的像素x1位置
                    y1 = temp(2, 1);%新的像素y1位置
                    % 变换后的位置判断是否越界
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%新的行位置要小于新的列位置
                        res(x1,y1,:)= I(x0,y0,:);%进行图像颜色赋值
                    end
                end
            end
            g = res;
        end
        
        function g = CustomReverseY(~, f)
            %             [M, N] = size(f);
            %             I = f;
            %             for i = 1 : M
            %                 for j = 1 : N
            %                     I(i, j) = f(M - i + 1, j);
            %                 end
            %             end
            %             g = I;
            
            
            img = f;
            [H,W,Z] = size(img); % 获取图像大小
            I=im2double(img);%将图像类型转换成双精度
            res = ones(H,W,Z); % 构造结果矩阵。每个像素点默认初始化为1（白色）
            tras = [-1 0 H; 0 1 0; 0 0 1]; % 垂直镜像的变换矩阵
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%将每一点的位置进行缓存
                    temp = tras * temp; % 根据算法进行，矩阵乘法：转换矩阵乘以原像素位置
                    x1 = temp(1, 1);%新的像素x1位置
                    y1 = temp(2, 1);%新的像素y1位置
                    % 变换后的位置判断是否越界
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%新的行位置要小于新的列位置
                        res(x1,y1,:)= I(x0,y0,:);%进行颜色赋值
                    end
                end
            end
            g = res;
        end
        
        
        
        function g = CustomMove(~,f,delX,delY)
            [R, C] = size(f);
            res = zeros(R, C);
            trans = [1 0 delX; 0 1 delY; 0 0 1];
            for i = 1 : R
                for j = 1 : C
                    temp = [i; j; 1];
                    temp = trans * temp;
                    x = temp(1, 1);
                    y = temp(2, 1);
                    if (x <= R) && (y <= C) && (x >= 1) && (y >= 1)
                        res(x, y) = f(i, j);
                    end
                end
            end
            g = uint8(res);
            %g = res;
        end
        
        function g = RGBMove(~,image,delX,delY)
            %             [W, H, G] = size(image); % 获取图像大小
            %             image_r=image(:,:,1);
            %             image_g=image(:,:,2);
            %             image_b=image(:,:,3);%获取图像的RGB值
            %             res = zeros(W, H, 3); % 构造结果矩阵。每个像素点默认初始化为0（黑色）
            %
            %             tras = [1 0 X; 0 1 Y; 0 0 1]; % 平移的变换矩阵
            %             for i = 1 : W
            %                 for j = 1 : H
            %                     temp = [i; j; 1];
            %                     temp = tras * temp; % 矩阵乘法
            %                     x = temp(1, 1);
            %                     y = temp(2, 1);%x、y分别为通过矩阵乘法得到后的平移位置的横纵坐标值
            %
            %                     % 变换后的位置判断是否越界
            %                     if (x <= W) && (y <= H)&&(x >= 1) && (y >= 1)
            %                         res(x,y,1) = image_r(i, j);
            %                         res(x,y,2) = image_g(i, j);
            %                         res(x,y,3) = image_b(i, j);%将新的RGB值赋予在背景上
            %                     end
            %                 end
            %             end
            %             g = uint8(res);
            im = image;
            [H,W,Z] = size(im); % 获取图像大小
            I=im2double(im);%将图像类型转换成双精度
            res = ones(H,W,Z); % 构造结果矩阵。每个像素点默认初始化为1（白色）
            tras = [1 0 delX; 0 1 delY; 0 0 1]; % 平移的变换矩阵
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%将每一点的位置进行缓存
                    temp = tras * temp; % 根据算法进行，矩阵乘法：转换矩阵乘以原像素位置
                    x1 = temp(1, 1);%新的像素x1位置，也就是新的行位置
                    y1 = temp(2, 1);%新的像素y1位置,也就是新的列位置
                    % 变换后的位置判断是否越界
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%新的行位置要小于新的列位置
                        res(x1,y1,:)= I(x0,y0,:);%进行图像平移，颜色赋值
                    end
                end
            end
            g = res;
        end
        
        function output = my_edge(~,input_img,method)
            if size(input_img,3)==3
                input_img=rgb2gray(input_img);
            end

            input_img=im2double(input_img);
            sobel_x=[-1,-2,-1;0,0,0;1,2,1];
            sobel_y=[-1,0,1;-2,0,2;-1,0,1];
            prewitt_x=[-1,-1,-1;0,0,0;1,1,1];
            prewitt_y=[-1,0,1;-1,0,1;-1,0,1];

            psf=fspecial('gaussian',[5,5],1);
            input_img=imfilter(input_img,psf);%高斯低通滤波，平滑图像,但可能会使图像丢失细节
            input_img=medfilt2(input_img); %中值滤波消除孤立点
            [m,n]=size(input_img);
            output=zeros(m,n);
            if nargin==3
                if strcmp(method,'sobel')
                    for i=2:m-1
                        for j=2:n-1
                            local_img=input_img(i-1:i+1, j-1:j+1);
            %近似边缘检测，加快速度    %output(i,j)=abs(sum(sum(sobel_x.*local_img)))+abs(sum(sum(sobel_x.*local_img)));
                            output(i,j)=sqrt(sum(sum(sobel_x.*local_img))^2+sum(sum(sobel_y.*local_img))^2);
                        end
                    end
                elseif strcmp(method,'prewitt')
                    for i=2:m-1
                        for j=2:n-1
                            local_img=input_img(i-1:i+1, j-1:j+1);
                            output(i,j)=sqrt(sum(sum(prewitt_x.*local_img))^2+sum(sum(prewitt_y.*local_img))^2);
                        end
                    end

                else%如果不输入算子的名称，默认使用roberts算子进行边缘检测
                    for i=1:m-1
                        for j=1:n-1
                            output(i,j)=abs(input_img(i,j)-input_img(i+1,j+1))+ ...
                            abs(input_img(i+1,j)-input_img(i,j+1));
                        end
                    end    
                end

                output=imadjust(output);%使边缘图像更明显
                thresh=graythresh(output);%确定二值化阈值
                output=bwmorph(imbinarize(output,thresh),'thin',inf);%强化细节
            end
        end
        
        function g = grow(~,I,x,y)
            I=double(I);              %转换为灰度值是0-1的双精度
            [M,N]=size(I);            %得到原图像的行列数
            x1=round(x);              %横坐标取整
            y1=round(y);              %纵坐标取整
            seed=I(x1,y1);            %将生长起始点灰度值存入seed中
            Y=zeros(M,N);             %作一个全零与原图像等大的图像矩阵Y，作为输出图像矩阵
            Y(x1,y1)=1;               %将Y中与所取点相对应位置的点设置为白点
            sum=seed;                 %储存符合区域生长条件的点的灰度值的总和
            suit=1;                   %储存符合区域生长条件的点的总个数
            count=1;                  %每次判断一点周围八点符合条件的新点的数目
            threshold=10;             %域值，即某一点与周围八点的绝对差值要小于阈值
            while count>0             %判断是否有新的符合生长条件的点，若没有，则结束
                s=0;                  %判断一点周围八点时，符合条件的新点的灰度值之和
                count=0;
                for i=1:M
                    for j=1:N
                        if Y(i,j)==1
                            if (i-1)>0 && (i+1)<(M+1) && (j-1)>0 && (j+1)<(N+1) %判断此点是否为图像边界上的点
                                for u= -1:1                                        %判断点周围八点是否符合域值条件
                                    for v= -1:1                                       %u,v为偏移量
                                        if  Y(i+u,j+v)==0 && abs(I(i+u,j+v)-seed)<=threshold%判断是否未存在于输出矩阵Y，并且为符合域值条件的点
                                            Y(i+u,j+v)=1;                                %符合以上两条件即将其在Y中与之位置对应的点设置为白点
                                            count=count+1;                               %新的、符合生长条件的点的总个数
                                            s=s+I(i+u,j+v);                              %新的、符合生长条件的点的总灰度数
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                suit=suit+count;                                   %目前区域所有符合生长条件的点的总个数
                sum=sum+s;                                         %目前区域所有符合生长条件的点的总灰度值
                seed=sum/suit;                                     %计算新的灰度平均值
            end
            g = Y;
        end
        
        function g = grow2(~,I,x,y)
            I = im2double(I);
            x = round(x(1));
            y = round(y(1));
            J = zeros(size(I)); % 主函数的返回值，记录区域生长所得到的区域
            Isizes = size(I);
            reg_mean = I(x,y);%表示分割好的区域内的平均值，初始化为种子点的灰度值
            reg_size = 1;%分割的到的区域，初始化只有种子点一个
            neg_free = 10000; %动态分配内存的时候每次申请的连续空间大小
            neg_list = zeros(neg_free,3);
            %定义邻域列表，并且预先分配用于储存待分析的像素点的坐标值和灰度值的空间，加速
            %如果图像比较大，需要结合neg_free来实现matlab内存的动态分配
            neg_pos = 0;%用于记录neg_list中的待分析的像素点的个数
            pixdist = 0;
            %记录最新像素点增加到分割区域后的距离测度
            %下一次待分析的四个邻域像素点和当前种子点的距离
            %如果当前坐标为（x,y）那么通过neigb我们可以得到其四个邻域像素的位置
            neigb = [ -1 0;
                1  0;
                0 -1;
                0  1];
            %开始进行区域生长，当所有待分析的邻域像素点和已经分割好的区域像素点的灰度值距离
            %大于reg_maxdis,区域生长结束
            while (pixdist < 0.06 && reg_size < numel(I))
                %增加新的邻域像素到neg_list中
                for j=1:4
                    xn = x + neigb(j,1);
                    yn = y + neigb(j,2);
                    %检查邻域像素是否超过了图像的边界
                    ins = (xn>=1)&&(yn>=1)&&(xn<=Isizes(1))&&(yn<=Isizes(1));
                    %如果邻域像素在图像内部，并且尚未分割好；那么将它添加到邻域列表中
                    if( ins && J(xn,yn)==0)
                        neg_pos = neg_pos+1;
                        neg_list(neg_pos,:) =[ xn, yn, I(xn,yn)];%存储对应点的灰度值
                        J(xn,yn) = 1;%标注该邻域像素点已经被访问过 并不意味着，他在分割区域内
                    end
                end
                %如果分配的内存空问不够，申请新的内存空间
                if (neg_pos+10>neg_free)
                    neg_free = neg_free + 100000;
                    neg_list((neg_pos +1):neg_free,:) = 0;
                end
                %从所有待分析的像素点中选择一个像素点，该点的灰度值和已经分割好区域灰度均值的
                %差的绝对值时所待分析像素中最小的
                dist = abs(neg_list(1:neg_pos,3)-reg_mean);
                [pixdist,index] = min(dist);
                %计算区域的新的均值
                reg_mean = (reg_mean * reg_size +neg_list(index,3))/(reg_size + 1);
                reg_size = reg_size + 1;
                %将旧的种子点标记为已经分割好的区域像素点
                J(x,y)=2;%标志该像素点已经是分割好的像素点
                x = neg_list(index,1);
                y = neg_list(index,2);
                neg_list(index,:) = neg_list(neg_pos,:);
                neg_pos = neg_pos -1;
            end
            g = J;
        end
        
        function T=Otsu(I)
            %             [m,n]=size(I);
            %             I=double(I);
            %             count=zeros(256,1);
            %             pcount=zeros(256,1);
            %             for i=1:m
            %                 for j=1:n
            %                     pixel=I(i,j);
            %                     count(pixel+1)=count(pixel+1)+1;
            %                 end
            %             end
            %             dw=0;
            %             for i=0:255
            %                 pcount(i+1)=count(i+1)/(m*n);
            %                 dw=dw+i*pcount(i+1);
            %             end
            %             Th=0;
            %             Thbest=0;
            %             dfc=0;
            %             dfcmax=0;
            %             while (Th>=0 & Th<=255)
            %                 dp1=0;
            %                 dw1=0;
            %                 for i=0:Th
            %                     dp1=dp1+pcount(i+1);
            %                     dw1=dw1+i*pcount(i+1);
            %                 end
            %                 if dp1>0
            %                     dw1=dw1/dp1;
            %                 end
            %                 dp2=0;
            %                 dw2=0;
            %                 for i=Th+1:255
            %                     dp2=dp2+pcount(i+1);
            %                     dw2=dw2+i*pcount(i+1);
            %                 end
            %                 if dp2>0
            %                     dw2=dw2/dp2;
            %                 end
            %                 dfc=dp1*(dw1-dw)^2+dp2*(dw2-dw)^2;
            %                 if dfc>=dfcmax
            %                     dfcmax=dfc;
            %                     Thbest=Th;
            %                 end
            %                 Th=Th+1;
            %             end
            %             T=Thbest;
            
            [m,n]=size(I);%得到图像行列像素
            I=im2double(I);%变为双精度,即0-1
            Th=0;
            Thbest=0;
            fc=0;
            fcmax=0;
            count=zeros(256,1);
            pcount=zeros(256,1);
            for i=1:m
                for j=1:n
                    pixel=I(i,j);
                    count(pixel+1)=count(pixel+1)+1;
                end
            end
            dw=0;
            for i=0:255
                pcount(i+1)=count(i+1)/(m*n);
                dw=dw+i*pcount(i+1);
            end
            while (Th>=0 & Th<=255)
                p1=0;%第一类像素的概率
                ave1=0;%第一类像素的均值
                for i=0:Th
                    p1=p1+pcount(i+1);
                    ave1=ave1+i*pcount(i+1);
                end
                if p1>0
                    ave1=ave1/p1;
                end
                p2=0;%第二类像素的概率
                ave2=0;%第二类像素的均值
                for i=Th+1:255
                    p2=p2+pcount(i+1);
                    ave2=ave2+i*pcount(i+1);
                end
                if p2>0
                    ave2=ave2/p2;
                end
                fc=p1*(ave1-dw)^2+p2*(ave2-dw)^2;%类间方差
                if fc>=fcmax
                    fcmax=fc;
                    Thbest=Th;
                end
                Th=Th+1;
            end
            T=Thbest;
        end
        
        function g = OtsuProcess(~,I)
            %             I = rgb2gray(I);
            %             I = double(I);
            %             [m,n] = size(I);
            %             Th = Otsu(I);
            %             for i=1:m
            %                 for j=1:n
            %                     if I(i,j)>=Th
            %                         I(i,j)=255;
            %                     else
            %                         I(i,j)=0;
            %                     end
            %                 end
            %             end
            %             g = I;
            
            %             if( ~( size(I,3)-3))%判断是否为彩色图
            %                 I=rgb2gray(I);
            %             end
            %             I=double(I);
            %             [m,n]=size(I);
            %             J=Otsu(I);
            %             for i=1:m
            %                 for j=1:n
            %                     if I(i,j)>=J
            %                         I(i,j)=255;
            %                     else
            %                         I(i,j)=0;
            %                     end
            %                 end
            %             end
            
            if( ~( size(I,3)-3))%判断是否为彩色图
                I=rgb2gray(I);
            end
            I=double(I);
            [m,n]=size(I);
            J=Otsu(I);
            for i=1:m
                for j=1:n
                    if I(i,j)>=J
                        I(i,j)=255;
                    else
                        I(i,j)=0;
                    end
                end
            end
            
            g = I;
        end
        
        function g = LaplaceSharpRGB(~,I1)
            I=im2double(I1);
            [m,n,c]=size(I);
            A=zeros(m,n,c);
            %分别处理R、G、B
            %先对R进行处理
            for i=2:m-1
                for j=2:n-1
                    A(i,j,1)=I(i+1,j,1)+I(i-1,j,1)+I(i,j+1,1)+I(i,j-1,1)-4*I(i,j,1);
                end
            end
            if c > 1
                %再对G进行处理
                for i=2:m-1
                    for j=2:n-1
                        A(i,j,2)=I(i+1,j,2)+I(i-1,j,2)+I(i,j+1,2)+I(i,j-1,2)-4*I(i,j,2);
                    end
                end
            end
            
            if c > 2
                %最后对B进行处理
                for i=2:m-1
                    for j=2:n-1
                        A(i,j,3)=I(i+1,j,3)+I(i-1,j,3)+I(i,j+1,3)+I(i,j-1,3)-4*I(i,j,3);
                    end
                end
            end
            
            B=I-A;
            g = B;
        end
        
        function [image_out] = GaussianLowPass(~,image_in,D0)
            %GLPF为高斯低通滤波器，D0为截止频率
            %输入为需要进行高斯低通滤波的灰度图像，输出为经过滤波之后的灰度图像
            f=image_in;
            f=im2double(f);
            % 1、给定一副大小为M×N的输入图像f(x,y)，得到填充参数P=2M，Q=2N
            M=size(f,1);   N=size(f,2);
            P=2*M;          Q=2*N;
            % 2、对输入图像f(x,y)添加必要数量的0,形成大小为P×Q的填充图像fp(x,y)
            fp=zeros(P,Q);
            fp(1:M,1:N)=f(1:M,1:N);
            % 3、用（-1）^(x+y)乘以fp(x,y)移到其变换的中心
            for i=1:P
                for j=1:Q
                    fp(i,j)=(-1)^(i+j)*double(fp(i,j));
                end
            end
            % 4、计算来自步骤3的图像的DFT，得到F（u,v）
            F=fft2(fp,P,Q);
            % 5、生成一个实的、对称的滤波函数H(u,v)，其大小为P×Q，中心在（P/2，Q/2）处。用阵列相乘形成乘积G（u,v）=H(u,v)F(u,v)
            H=zeros(P,Q);
            a=2*(D0^2);
            for u=1:P
                for v=1:Q
                    D=(u-P/2)^2+(v-Q/2)^2;
                    H(u,v)=exp(-D./a);
                end
            end
            G=F.*H; %频率域滤波
            % 6、得到处理后的图像
            gp=ifft2(G); %频域转换到时域图像
            gp=real(gp);
            for i=1:P
                for j=1:Q
                    gp(i,j)=(-1)^(i+j)*double(gp(i,j));
                end
            end
            % 7、通过从gp(x,y)的左上象限提取M×N区域，得到最终处理结果g(x,y)
            image_out=gp(1:M,1:N);
        end
        
        function g = divide2(~,I)
            if( ~( size(I,3)-3))%判断是否为彩色图
                I=rgb2gray(I);
            end
            I=double(I);
            [m,n]=size(I);
            Smax=0;
            for T=0:255
                sum1=0; num1=0;
                sum2=0; num2=0;
                for i=1:m
                    for j=1:n
                        if I(i,j)>=T
                            sum2=sum2+I(i,j);
                            num2=num2+1;
                        else
                            sum1=sum1+I(i,j);
                            num1=num1+1;
                        end
                    end
                end
                ave1=sum1/num1;
                ave2=sum2/num2;
                S=((ave2-T)*(T-ave1))/(ave2-ave1)^2;
                if(S>Smax)
                    Smax=S;
                    Th=T;
                end
            end
            for i=1:m
                for j=1:n
                    if I(i,j)>=Th
                        I(i,j)=255;
                    else
                        I(i,j)=0;
                    end
                end
            end
            g = I;
        end
        
        function y = mat2lpc(~, x, f)
            %MAT2LPC Compresses a matrix using 1-D lossless predictive coding.
            % Y = MAT2LPC(X, F) encodes matrix X using 1-D lossless predictive coding.
            % A linear prediction of X is made based on the coefficients in F. If F is
            % omitted, F = 1 (for previous pixel coding) is assumed. The prediction
            % error is then computed and output as encoded matrix Y.
            %
            % See also LPC2MAT.
            narginchk(1, 2);   % Check input arguments
            if nargin < 3                   % Set default filter if omitted
                f = 1;
            end
            
            x = double(x);                  % Ensure double for computations
            [m, n] = size(x);               % Get dimensions of input matrix
            p = zeros(m, n);                % Init linear prediction to 0
            xs = x; zc = zeros(m, 1);       % Prepare for input shift and pad
            
            for j = 1 : length(f)           % For each filter coefficient
                xs =[zc xs(:, 1:end - 1)];  % Shift and zero pad x
                p = p + f(j) * xs;          % Form partial prediction sums
            end
            
            y = x - round(p);               % Compute prediction error
        end
        
        function y = mat2huff(~,x)
            if ~ismatrix(x) || ~isreal(x) || (~isnumeric(x) && ~islogical(x))
                error('X must be a 2-D real numeric or logical matrix.');
            end
            
            % Store the size of input x.
            y.size = uint32(size(x));
            % Find the range of x values and store its minimum value biased by +32768
            % as a UINT16
            x = round(double(x));
            xmin = min(x(:));
            xmax = max(x(:));
            pmin = double(int16(xmin));
            pmin = uint16(pmin + 32768);
            y.min = pmin;
            
            % Compute the input histogram between xmin and xmax with unit width bins,
            % scale to UINT16, and store.
            x = x(:)';
            h = histc(x, xmin:xmax);
            if max(h) > 65535
                h = 65535 * h / max(h);
            end
            h = uint16(h);
            y.hist = h;
            % Code the input matrix and store the result.
            map = huffman(double(h));  % Make Huffman code map
            hx = map(x(:) - xmin + 1); % Map image
            hx = char(hx)';            % Convert to char array
            hx = hx(:)';
            hx(hx == ' ') = [];        % Remove blanks
            ysize = ceil(length(hx)/16); % Compute encoded size
            hx16 = repmat('0', 1, ysize * 16); % Pre-allocate modulo - 16 vector
            hx16 (1:length(hx)) = hx;     % Make hx modulo -16 in length
            hx16 = reshape(hx16, 16, ysize); % Reshape to 16-character words
            hx16 = hx16' - '0';       % Convert binary string to decimal
            twos = pow2(15: -1: 0);
            y.code = uint16(sum(hx16 .* twos(ones(ysize, 1), :), 2))';
        end
        
        function cr = imratio(f1, f2)
            %IMRATIO Computes the ratio of the bytes in two images/variables
            % CR = IMRATIO(F1, F2) returns the ratio of the number of bytes in
            % variables/files F1 and F2. If F1 and F2 are an original and compressed
            % image, respectively, CR is the compression ratio.
            error(nargchk(2, 2, nargin));  %check input argument
            cr = bytes(f1) / bytes(f2);    %Compute the ratio
            
            %.......................................................................
            function b = bytes(f)
                %Return the number of bytes in input f. If f is a string. assume that it is
                %an image filename; if not, it is an image vraiable.
                if ischar(f)
                    info = dir(f);
                    b = info.bytes;
                elseif isstruct(f)
                    %MATLAB is whos function reports an extra 124 bytes of memory per
                    %structure field because of the way MATLAB stores structures in memory.
                    %Don't count this extra memory; instead, add up the memory associated
                    %with each field.
                    b = 0;
                    fields = fieldnames(f);  %fildnames函数获得结构体f中各字段的名称
                    k1 = length(fields);
                    for k = 1:k1
                        elements = f.(fields{k});
                        k2 = length(elements);
                        for m = 1:k2
                            ele = elements(m);
                            b = b + bytes(ele);
                        end
                    end
                else
%                     info = whos('f'); %whos函数返回携带matlab工作空间中变量'f'信息的结构体
                    b = info.bytes;
                end
            end
        end
        
        function g = customHisteq(~,I)
            [m, n, d] = size(I);
            if d == 1
                f1 = I;
            elseif d == 3
                I = rgb2gray(I);
                f1 = I;
            end
            [count, ~] = imhist(I);
            PDF = count/(m*n);
            CDF = cumsum(PDF);
            
            for i = 1:256
                num = find(I==i);
                len = length(num);
                for j = 1:len
                    f1(num(j)) = round(CDF(i)*256-1);
                end
            end
            g = f1;
        end
        
        function g = scale(~, B, S)
            [r,c] = size(B);
            nr= round(r*S);             %根据放大倍数乘原行数的结果，取其四舍五入的值作为新的行
            nc= round(c*S);             %根据放大倍数乘原列数的结果，取其四舍五入的值作为新的列
            A = zeros(nr,nc);           %用新的行列生成目标图像矩阵
            SB = zeros(r+1,c+1);        %新建一个矩阵SB，大小在B的基础上行列都加1
            %%%%%处理SB边界%%%%%
            SB(2:r+1,2:c+1)=B;
            SB(2:r+1,1)=B(:,1);
            SB(1,2:c+1)=B(1,:);
            SB(1,1)=B(1,1);
            %%%%%处理SB边界%%%%%
            for Ai=1:nr
                for Aj=1:nc
                    Bi=(Ai-1)/S;       %求出Ai对应的Bi坐标，Ai是由Bi先缩放S倍，再在竖直方向正向平移1得到
                    Bj=(Aj-1)/S;       %求出Aj对应的Bj坐标，Aj是由Bj先缩放S倍，再在水平方向正向平移1得到
                    i=fix(Bi);         %向零方向取整，求出坐标Bi的整数部分
                    j=fix(Bj);         %向零方向取整，求出坐标Bj的整数部分
                    u=Bi-i;            %求出坐标Bi的小数部分
                    v=Bj-j;            %求出坐标Bj的小数部分
                    i=i+1;             %这是在矩阵SB上计算的，不是在矩阵B上计算的，竖直方向上有平移量，加1对应B上的i值
                    j=j+1;             %这是在矩阵SB上计算的，不是在矩阵B上计算的，水平方向上有平移量，加1对应B上的j值
                    A(Ai,Aj)=(1-u)*(1-v)*SB(i,j)+u*v*SB(i+1,j+1)+u*(1-v)*SB(i+1,j)+(1-u)*v*SB(i,j+1);%双线性插值法计算A(Ai,Aj)
                end
            end
            g = uint8(A);
        end
        
        function y = quantize(x, b, type)
            %QUANTIZE Quantizes the elements of a UINT8 matrix.
            % Y = QUANTIZE(X, B, TYPE) quantizes X to B bits. Truncation is used unless
            % TYPE is 'igs' for Improved Gray Scale quantization
            
            narginchk(2, 3);
            if ~ismatrix(x) || ~isreal(x) || ~isnumeric(x) || ~isa(x, 'uint8')
                error('The input must be a UINT8 numeric matrix.');
            end
            
            % Create bit masks for the quantization
            lo = uint8(2 ^ (8-b) - 1); %lo有8-b位1在低位，其余位为0
            hi = uint8(2 ^ 8 - double(lo) - 1); %hi有b位1在高位，其余位为0
            
            % Perform standard quantization unless IGS is specified
            if nargin < 3 || ~strcmpi(type, 'igs') %进行一般灰度量化
                y = bitand(x, hi); %计算x和hi的按位与运算，即保留像素灰度的高位值，灰度差异的大小体现在高位
                
                % Else IGS quantization. Process column-wise. If the MSB's of the pixel are
                % all 1's, the sum is set to the pixel value. Else, add the pixel value to
                % the LSB's of the previous sum. Then take the MSB's of the sum as the
                % quantized value.
                % MSB：最高有效位；LSB：最低有效位
            else %进行IGS量化
                %IGS扰动方法：对于任何高位不是hi的像素，将其加上前一列相邻像素的对应lo低位的值
                [m,n] = size(x);
                s = zeros(m, 1);
                %获得对应于x中高位不是hi的那些元素的集合
                hitest = double(bitand(x, hi) ~=  hi);
                %figure;imshow(hitest);
                x  = double(x);
                for j = 1:n
                    tt = double(bitand(uint8(s), lo));
                    %tt为经IGS扰动后的X中前一列像素的对应于lo低位的值
                    s = x(:,j)+hitest(:,j) .* tt; %经扰动后的当前列的像素值
                    y(:, j) = bitand(uint8(s), hi);%保留当前列像素灰度的高位值
                end
            end
        end
        
        
        
        function g = grayLevel(~,I)
            f=I;
            q=quantize(f,4,'igs');%用函数quantize进行igs量化到4bit
            qs=double(q)/16;
            e=mat2lpc(qs);    %使用预测编码后
            g = e;
        end
        
        function h = ntrop(~, x, n)
            %NTROP Computes a first-order estimate of the entropy of a matrix.
            % H = NTROP(X, N) returns the entropy of matrix X with N symbols. N = 256
            % if omitted but it must be larger than the number of unique values in X
            % for accurate results. The estimate assumes a statistically independent
            % source characterized by the relative frequency of occurrence of the
            % elements in X. The estimate is a lower bound on the average number of
            % bits per unique value (or symbol) when coding without cooding redundancy.
            error(nargchk(1, 2, nargin)); % Check input arguments
            if nargin < 3
                n = 256;  % Default for n
            end
            
            x = double(x); % Make input double
            xh = hist(x(:), n); % Compute N-bin histogram
            xh = xh / sum(xh(:)); % Compute probabilities
            
            % Make mask to eliminate 0's since log2(0) = -inf.
            i = find(xh); %等价于find(xh ~= 0)
            h = -sum(xh(i) .* log2(xh(i)));  % Compute entropy
            
        end

        function g = prewittSharp(~,I)
            if( ~( size(I,3)-3))%判断是否为彩色图
                I1 = rgb2gray(I);
            else
                I1=I;
            end
            model=[-1,0,1;
                -1,0,1;
                -1,0,1];
            [m,n]=size(I1);
            I2=I1;
            for i=2:m-1
                for j=2:n-1
                    tem=I1(i-1:i+1,j-1:j+1);
                    tem=double(tem).*double(model);
                    I2(i,j)=sum(sum(tem));
                end
            end
            I2 = double(I2)+ double(I);
            g = uint8(I2);
        end
        
        function g = robertSharp(~, I)
            if( ~( size(I,3)-3))%判断是否为彩色图
                I1 = rgb2gray(I);
            else
                I1=I;
            end
            model=[0,-1;1,0];
            [m,n]=size(I1);
            I2=double(I1);
            for i=2:m-1
                for j=2:n-1
                    I2(i,j)=I1(i+1,j)-I1(i,j+1);
                end
            end
            I2 = I2 + double(I);
            g = uint8(I2);
        end
        
        function g = sobelSharp(~,I)
            if( ~( size(I,3)-3))%判断是否为彩色图
                I1 = rgb2gray(I);
            else
                I1=I;
            end
            model=[-1,0,1;
                -2,0,2;
                -1,0,1];
            [m,n]=size(I1);
            I2=double(I1);
            for i=2:m-1
                for j=2:n-1
                    I2(i,j)=I1(i+1,j+1)+2*I1(i+1,j)+I1(i+1,j-1)-I1(i-1,j+1)-2*I1(i-1,j)-I1(i-1,j-1);
                end
            end
            I2 = I2 + double(I);
            g = uint8(I2);
        end
        
        
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Menu selected function: open
        function openFile(app, event)
            % Display uigetfile dialog
            filterspec = {'*.jpg;*.tif;*.png;*.gif','All Image Files'};
            [f, p] = uigetfile(filterspec);
            
            % Make sure user didn't cancel uigetfile dialog
            if (ischar(p))
               fname = [p f];
               updateImage(app, fname);
               showHist(app);
            end
        end

        % Menu selected function: save
        function saveFile(app, event)
            %unsavedImage = app.Image;
            %imwrite(unsavedImage,"app1.jpg","jpg");
            
            [file,path] = uiputfile({'*.jpg'; ...
                '*.tif'; ...
                '*.png'; ...
                '*.gif'}, ...
                'Save as');
            
            fullFilePath = fullfile(path,file);
            imwrite(app.Image,fullFilePath);
        end

        % Menu selected function: Menu_15
        function midfiltMenu(app, event)
            pImage = app.Image;
            
            an = inputdlg('滤波器矩阵大小：');
            an = str2double(an);
            pImage = midfilt(app,pImage,an);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: saltPepper
        function saltPepperMenu(app, event)
            pImage = app.Image;
            pImage = saltPepperNoise(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: gaussian
        function gaussianMenu(app, event)
            pImage = app.Image;
            pImage = gaussianNoise(app,pImage);
            %todo: unfinished parameter input
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: avgFilter
        function avgFilterMenuSelected(app, event)
            pImage = app.Image;
            an = inputdlg('滤波器矩阵大小：');
            an = str2double(an);
            pImage = avg_filter(app,pImage,an);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: rotate
        function rotateMenuSelected(app, event)
            an = inputdlg('Input the rotation angle of the picture');
            an = str2double(an);
            pImage = app.Image;
            pImage = CustomRotate(app,pImage,an);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: RGB2gray
        function RGB2grayMenuSelected(app, event)
            pImage = app.Image;
            pImage = CustomRGB2gray(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Move
        function MoveMenuSelected(app, event)
            prompt = {'X轴平移距离：','Y轴平移距离'};
            dlg_title = 'Input';
            num_lines = 1;
            def = {'20','20'};
            an = inputdlg(prompt,dlg_title,num_lines,def);
            
            disX = str2double(an(1,1));
            disY = str2double(an(2,1));
            
            pImage = app.Image;
            
            switch size(pImage,3)
                case 1
                    % Display the grayscale image
                    pImage = CustomMove(app,pImage,disX,disY);
                case 3
                    % the truecolor image
                    pImage = RGBMove(app,pImage,disX,disY);
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    %uialert(app.UIFigure, '图片格式错误。', 'ERROR');
                    warndlg('图片格式错误。','ERROR');
                    return;
            end
            
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_19
        function Menu_19Selected(app, event)
            pImage = app.Image;
            pImage = CustomReverse(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_18
        function Menu_18Selected(app, event)
            pImage = app.Image;
            pImage = CustomReverseY(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: SobelMenu
        function SobelMenuSelected(app, event)
            pImage = app.Image;
            pImage = my_edge(app,pImage,'sobel');
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: robertsMenu
        function robertsMenuSelected(app, event)
            pImage = app.Image;
            pImage = my_edge(app,pImage,'');
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: prewittMenu
        function prewittMenuSelected(app, event)
            pImage = app.Image;
            pImage = my_edge(app,pImage,'prewitt');
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_24
        function Menu_24Selected(app, event)
            pImage = app.Image;
            if( ~( size(pImage,3)-3 ))
                pImage = rgb2gray(pImage);%转化为单通道灰度图
            end
%             app.ImageAxes_2.HandleVisibility = 'on';
%             set(0, 'CurrentFigure', app.ImageAxes_2)
%             set(app.ImageAxes_2, 'CurrentFigure', app.ImageAxes)
            [y, x, ~] = impixel(pImage);
%             app.ImageAxes_2.HandleVisibility = 'off';
%             [x,y] = getpts;
%             imshow(pImage);title("双击选择生长点");
            pImage = grow2(app,pImage,x,y);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: OtsuMenu
        function OtsuMenuSelected(app, event)
            pImage = app.Image;
            pImage = OtsuProcess(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Laplace
        function LaplaceMenuSelected(app, event)
            pImage = app.Image;
            
            switch size(pImage,3)
                case 1
                    % Display the grayscale image
                    %todo: grayscale picture LaplaceSharp
                    pImage = LaplaceSharpRGB(app,pImage);
                case 3
                    % the truecolor image
                    pImage = LaplaceSharpRGB(app,pImage);
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    %uialert(app.UIFigure, '图片格式错误。', 'ERROR');
                    warndlg('图片格式错误。','ERROR');
                    return;
            end
            
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_26
        function Menu_26Selected(app, event)
            an = inputdlg('截止频率：');
            an = str2double(an);
            pImage = app.Image;
            pImage = GaussianLowPass(app,pImage,an);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_27
        function Menu_27Selected(app, event)
            pImage = app.Image;
            pImage = divide2(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_29
        function Menu_29Selected(app, event)
            pImage = app.Image;
            pImage = mat2lpc(app,pImage);
            pImage = mat2gray(pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_30
        function Menu_30Selected(app, event)
%             pImage = app.Image;
%             pImage = mat2huff(app,pImage);
%             [file,path] = uiputfile({
%                 '*.mat'}, ...
%                 'Save as');
%             
%             fullFilePath = fullfile(path,file);
%             app.save squeeze c;
            
            
%             app.Image = pImage;
%             imshow(app.Image,'Parent',app.ImageAxes);
            %todo: huffman编码解码是结构体，保存到文件
        end

        % Menu selected function: Menu_31
        function Menu_31Selected(app, event)
            pImage = app.Image;
            pImage = customHisteq(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_32
        function Menu_32Selected(app, event)
            an = inputdlg('缩放倍数：');
            an = str2double(an);
            pImage = app.Image;
            pImage = scale(app,pImage,an);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_33
        function Menu_33Selected(app, event)
            pImage = app.Image;
            pImage = grayLevel(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: PriwittMenu
        function PriwittMenuSelected(app, event)
            pImage = app.Image;
            pImage = prewittSharp(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: RobertMenu
        function RobertMenuSelected(app, event)
            pImage = app.Image;
            pImage = robertSharp(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: SobelMenu_2
        function SobelMenu_2Selected(app, event)
            pImage = app.Image;
            pImage = sobelSharp(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.9412 0.9412 0.9412];
            app.UIFigure.Position = [100 100 1020 720];
            app.UIFigure.Name = 'MATLAB App';

            % Create File
            app.File = uimenu(app.UIFigure);
            app.File.Text = '文件';

            % Create open
            app.open = uimenu(app.File);
            app.open.MenuSelectedFcn = createCallbackFcn(app, @openFile, true);
            app.open.Text = '打开图片';

            % Create save
            app.save = uimenu(app.File);
            app.save.MenuSelectedFcn = createCallbackFcn(app, @saveFile, true);
            app.save.Text = '保存图片';

            % Create transform
            app.transform = uimenu(app.UIFigure);
            app.transform.Text = '变换';

            % Create rotate
            app.rotate = uimenu(app.transform);
            app.rotate.MenuSelectedFcn = createCallbackFcn(app, @rotateMenuSelected, true);
            app.rotate.Text = '图像旋转';

            % Create Move
            app.Move = uimenu(app.transform);
            app.Move.MenuSelectedFcn = createCallbackFcn(app, @MoveMenuSelected, true);
            app.Move.Text = '图像平移';

            % Create Menu_17
            app.Menu_17 = uimenu(app.transform);
            app.Menu_17.Text = '图像翻转';

            % Create Menu_18
            app.Menu_18 = uimenu(app.Menu_17);
            app.Menu_18.MenuSelectedFcn = createCallbackFcn(app, @Menu_18Selected, true);
            app.Menu_18.Text = '垂直翻转';

            % Create Menu_19
            app.Menu_19 = uimenu(app.Menu_17);
            app.Menu_19.MenuSelectedFcn = createCallbackFcn(app, @Menu_19Selected, true);
            app.Menu_19.Text = '水平翻转';

            % Create Menu_32
            app.Menu_32 = uimenu(app.transform);
            app.Menu_32.MenuSelectedFcn = createCallbackFcn(app, @Menu_32Selected, true);
            app.Menu_32.Text = '图像缩放';

            % Create RGB2gray
            app.RGB2gray = uimenu(app.transform);
            app.RGB2gray.MenuSelectedFcn = createCallbackFcn(app, @RGB2grayMenuSelected, true);
            app.RGB2gray.Text = '灰度变换';

            % Create Menu_21
            app.Menu_21 = uimenu(app.UIFigure);
            app.Menu_21.Text = '直方图';

            % Create Menu_31
            app.Menu_31 = uimenu(app.Menu_21);
            app.Menu_31.MenuSelectedFcn = createCallbackFcn(app, @Menu_31Selected, true);
            app.Menu_31.Text = '直方图均衡';

            % Create noise
            app.noise = uimenu(app.UIFigure);
            app.noise.Text = '噪声';

            % Create Menu_15
            app.Menu_15 = uimenu(app.noise);
            app.Menu_15.MenuSelectedFcn = createCallbackFcn(app, @midfiltMenu, true);
            app.Menu_15.Text = '中值滤波';

            % Create avgFilter
            app.avgFilter = uimenu(app.noise);
            app.avgFilter.MenuSelectedFcn = createCallbackFcn(app, @avgFilterMenuSelected, true);
            app.avgFilter.Text = '均值滤波';

            % Create saltPepper
            app.saltPepper = uimenu(app.noise);
            app.saltPepper.MenuSelectedFcn = createCallbackFcn(app, @saltPepperMenu, true);
            app.saltPepper.Text = '椒盐噪声';

            % Create gaussian
            app.gaussian = uimenu(app.noise);
            app.gaussian.MenuSelectedFcn = createCallbackFcn(app, @gaussianMenu, true);
            app.gaussian.Text = '高斯噪声';

            % Create Menu_20
            app.Menu_20 = uimenu(app.UIFigure);
            app.Menu_20.Text = '边缘';

            % Create SobelMenu
            app.SobelMenu = uimenu(app.Menu_20);
            app.SobelMenu.MenuSelectedFcn = createCallbackFcn(app, @SobelMenuSelected, true);
            app.SobelMenu.Text = 'Sobel算子';

            % Create robertsMenu
            app.robertsMenu = uimenu(app.Menu_20);
            app.robertsMenu.MenuSelectedFcn = createCallbackFcn(app, @robertsMenuSelected, true);
            app.robertsMenu.Text = 'roberts算子';

            % Create prewittMenu
            app.prewittMenu = uimenu(app.Menu_20);
            app.prewittMenu.MenuSelectedFcn = createCallbackFcn(app, @prewittMenuSelected, true);
            app.prewittMenu.Text = 'prewitt算子';

            % Create Menu_23
            app.Menu_23 = uimenu(app.UIFigure);
            app.Menu_23.Text = '分割';

            % Create Menu_24
            app.Menu_24 = uimenu(app.Menu_23);
            app.Menu_24.MenuSelectedFcn = createCallbackFcn(app, @Menu_24Selected, true);
            app.Menu_24.Text = '区域生长';

            % Create OtsuMenu
            app.OtsuMenu = uimenu(app.Menu_23);
            app.OtsuMenu.MenuSelectedFcn = createCallbackFcn(app, @OtsuMenuSelected, true);
            app.OtsuMenu.Text = 'Otsu方法';

            % Create Menu_27
            app.Menu_27 = uimenu(app.Menu_23);
            app.Menu_27.MenuSelectedFcn = createCallbackFcn(app, @Menu_27Selected, true);
            app.Menu_27.Text = '类间最大距离';

            % Create Menu_34
            app.Menu_34 = uimenu(app.UIFigure);
            app.Menu_34.Text = '锐化';

            % Create PriwittMenu
            app.PriwittMenu = uimenu(app.Menu_34);
            app.PriwittMenu.MenuSelectedFcn = createCallbackFcn(app, @PriwittMenuSelected, true);
            app.PriwittMenu.Text = 'Priwitt锐化';

            % Create RobertMenu
            app.RobertMenu = uimenu(app.Menu_34);
            app.RobertMenu.MenuSelectedFcn = createCallbackFcn(app, @RobertMenuSelected, true);
            app.RobertMenu.Text = 'Robert锐化';

            % Create SobelMenu_2
            app.SobelMenu_2 = uimenu(app.Menu_34);
            app.SobelMenu_2.MenuSelectedFcn = createCallbackFcn(app, @SobelMenu_2Selected, true);
            app.SobelMenu_2.Text = 'Sobel锐化';

            % Create Menu_28
            app.Menu_28 = uimenu(app.UIFigure);
            app.Menu_28.Text = '压缩';

            % Create Menu_30
            app.Menu_30 = uimenu(app.Menu_28);
            app.Menu_30.MenuSelectedFcn = createCallbackFcn(app, @Menu_30Selected, true);
            app.Menu_30.Text = '霍夫曼压缩';

            % Create Menu_29
            app.Menu_29 = uimenu(app.Menu_28);
            app.Menu_29.MenuSelectedFcn = createCallbackFcn(app, @Menu_29Selected, true);
            app.Menu_29.Text = '无损预测';

            % Create Menu_33
            app.Menu_33 = uimenu(app.Menu_28);
            app.Menu_33.MenuSelectedFcn = createCallbackFcn(app, @Menu_33Selected, true);
            app.Menu_33.Text = '灰度级量化';

            % Create UITable
            app.UITable = uitable(app.UIFigure);
            app.UITable.ColumnName = {'熵'};
            app.UITable.RowName = {};
            app.UITable.Position = [843 315 178 58];

            % Create Menu_25
            app.Menu_25 = uimenu(app.UIFigure);
            app.Menu_25.Text = '模糊';

            % Create Laplace
            app.Laplace = uimenu(app.Menu_25);
            app.Laplace.MenuSelectedFcn = createCallbackFcn(app, @LaplaceMenuSelected, true);
            app.Laplace.Text = '空域锐化';

            % Create Menu_26
            app.Menu_26 = uimenu(app.Menu_25);
            app.Menu_26.MenuSelectedFcn = createCallbackFcn(app, @Menu_26Selected, true);
            app.Menu_26.Text = '高斯低通';

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.UIFigure);
            app.ImageAxes.XColor = [1 1 1];
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickLabel = {'[ ]'};
            app.ImageAxes.YColor = [1 1 1];
            app.ImageAxes.YTick = [];
            app.ImageAxes.ZColor = [1 1 1];
            app.ImageAxes.Position = [1 1 843 720];

            % Create GreenAxes
            app.GreenAxes = uiaxes(app.UIFigure);
            title(app.GreenAxes, 'Green')
            app.GreenAxes.XLim = [0 255];
            app.GreenAxes.XTick = [0 128 255];
            app.GreenAxes.Position = [839 488 182 117];

            % Create BlueAxes
            app.BlueAxes = uiaxes(app.UIFigure);
            title(app.BlueAxes, 'Blue')
            app.BlueAxes.PlotBoxAspectRatio = [2.46808510638298 1 1];
            app.BlueAxes.XLim = [0 255];
            app.BlueAxes.XTick = [0 128 255];
            app.BlueAxes.Position = [839 372 182 117];

            % Create RedAxes
            app.RedAxes = uiaxes(app.UIFigure);
            title(app.RedAxes, 'Red')
            app.RedAxes.PlotBoxAspectRatio = [2.46808510638298 1 1];
            app.RedAxes.XLim = [0 255];
            app.RedAxes.XTick = [0 128 255];
            app.RedAxes.Position = [839 604 182 117];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = app1_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end