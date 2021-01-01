classdef app1 < matlab.apps.AppBase

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
        Menu_25      matlab.ui.container.Menu
        Laplace      matlab.ui.container.Menu
        Menu_26      matlab.ui.container.Menu
        Menu_28      matlab.ui.container.Menu
        Menu_30      matlab.ui.container.Menu
        Menu_29      matlab.ui.container.Menu
        Menu_33      matlab.ui.container.Menu
        UITable      matlab.ui.control.Table
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
                    warndlg('ͼƬ��ʽ����','ERROR');
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
                    %uialert(app.UIFigure, 'ͼƬ��ʽ����', 'ERROR');
                    warndlg('ͼƬ��ʽ����','ERROR');
                    d = f;
                    return;
            end
        end
        
        function d = gaussianNoise(~,I)
            [m,n,~]=size(I);
            y=0+0.1*randn(m,n);%��ά��˹�ֲ����� 0�Ǿ�ֵ 0.1�Ǳ�׼��
            %�Ƚ���double�����ٳ���255 ���ں������
            K=double(I)/255;
            %��������
            K=K+y;
            %�����ط�Χ������0--255
            K=K*255;
            %ת��Ϊuint8����
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
                    warndlg('��ʹ��RGBͼ����лҶȴ���','WARNING');
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
                    %uialert(app.UIFigure, 'ͼƬ��ʽ����', 'ERROR');
                    warndlg('ͼƬ��ʽ����','ERROR');
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
            [H,W,Z] = size(img); % ��ȡͼ���С
            I=im2double(img);%��ͼ������ת����˫����
            res = ones(H,W,Z); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ1����ɫ��
            tras = [1 0 0; 0 -1 W; 0 0 1]; % ˮƽ����ı任����
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%��ÿһ���λ�ý��л���
                    temp = tras * temp; % �����㷨���У�����˷���ת���������ԭ����λ��
                    x1 = temp(1, 1);%�µ�����x1λ��
                    y1 = temp(2, 1);%�µ�����y1λ��
                    % �任���λ���ж��Ƿ�Խ��
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%�µ���λ��ҪС���µ���λ��
                        res(x1,y1,:)= I(x0,y0,:);%����ͼ����ɫ��ֵ
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
            [H,W,Z] = size(img); % ��ȡͼ���С
            I=im2double(img);%��ͼ������ת����˫����
            res = ones(H,W,Z); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ1����ɫ��
            tras = [-1 0 H; 0 1 0; 0 0 1]; % ��ֱ����ı任����
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%��ÿһ���λ�ý��л���
                    temp = tras * temp; % �����㷨���У�����˷���ת���������ԭ����λ��
                    x1 = temp(1, 1);%�µ�����x1λ��
                    y1 = temp(2, 1);%�µ�����y1λ��
                    % �任���λ���ж��Ƿ�Խ��
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%�µ���λ��ҪС���µ���λ��
                        res(x1,y1,:)= I(x0,y0,:);%������ɫ��ֵ
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
            %             [W, H, G] = size(image); % ��ȡͼ���С
            %             image_r=image(:,:,1);
            %             image_g=image(:,:,2);
            %             image_b=image(:,:,3);%��ȡͼ���RGBֵ
            %             res = zeros(W, H, 3); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ0����ɫ��
            %
            %             tras = [1 0 X; 0 1 Y; 0 0 1]; % ƽ�Ƶı任����
            %             for i = 1 : W
            %                 for j = 1 : H
            %                     temp = [i; j; 1];
            %                     temp = tras * temp; % ����˷�
            %                     x = temp(1, 1);
            %                     y = temp(2, 1);%x��y�ֱ�Ϊͨ������˷��õ����ƽ��λ�õĺ�������ֵ
            %
            %                     % �任���λ���ж��Ƿ�Խ��
            %                     if (x <= W) && (y <= H)&&(x >= 1) && (y >= 1)
            %                         res(x,y,1) = image_r(i, j);
            %                         res(x,y,2) = image_g(i, j);
            %                         res(x,y,3) = image_b(i, j);%���µ�RGBֵ�����ڱ�����
            %                     end
            %                 end
            %             end
            %             g = uint8(res);
            im = image;
            [H,W,Z] = size(im); % ��ȡͼ���С
            I=im2double(im);%��ͼ������ת����˫����
            res = ones(H,W,Z); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ1����ɫ��
            tras = [1 0 delX; 0 1 delY; 0 0 1]; % ƽ�Ƶı任����
            for x0 = 1 : H
                for y0 = 1 : W
                    temp = [x0; y0; 1];%��ÿһ���λ�ý��л���
                    temp = tras * temp; % �����㷨���У�����˷���ת���������ԭ����λ��
                    x1 = temp(1, 1);%�µ�����x1λ�ã�Ҳ�����µ���λ��
                    y1 = temp(2, 1);%�µ�����y1λ��,Ҳ�����µ���λ��
                    % �任���λ���ж��Ƿ�Խ��
                    if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%�µ���λ��ҪС���µ���λ��
                        res(x1,y1,:)= I(x0,y0,:);%����ͼ��ƽ�ƣ���ɫ��ֵ
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
            input_img=imfilter(input_img,psf);%��˹��ͨ�˲���ƽ��ͼ��,�����ܻ�ʹͼ��ʧϸ��
            input_img=medfilt2(input_img); %��ֵ�˲�����������
            [m,n]=size(input_img);
            output=zeros(m,n);
            if nargin==3
                if strcmp(method,'sobel')
                    for i=2:m-1
                        for j=2:n-1
                            local_img=input_img(i-1:i+1, j-1:j+1);
            %���Ʊ�Ե��⣬�ӿ��ٶ�    %output(i,j)=abs(sum(sum(sobel_x.*local_img)))+abs(sum(sum(sobel_x.*local_img)));
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

                else%������������ӵ����ƣ�Ĭ��ʹ��roberts���ӽ��б�Ե���
                    for i=1:m-1
                        for j=1:n-1
                            output(i,j)=abs(input_img(i,j)-input_img(i+1,j+1))+ ...
                            abs(input_img(i+1,j)-input_img(i,j+1));
                        end
                    end    
                end

                output=imadjust(output);%ʹ��Եͼ�������
                thresh=graythresh(output);%ȷ����ֵ����ֵ
                output=bwmorph(imbinarize(output,thresh),'thin',inf);%ǿ��ϸ��
            end
        end
        
        function g = grow(~,I,x,y)
            I=double(I);              %ת��Ϊ�Ҷ�ֵ��0-1��˫����
            [M,N]=size(I);            %�õ�ԭͼ���������
            x1=round(x);              %������ȡ��
            y1=round(y);              %������ȡ��
            seed=I(x1,y1);            %��������ʼ��Ҷ�ֵ����seed��
            Y=zeros(M,N);             %��һ��ȫ����ԭͼ��ȴ��ͼ�����Y����Ϊ���ͼ�����
            Y(x1,y1)=1;               %��Y������ȡ�����Ӧλ�õĵ�����Ϊ�׵�
            sum=seed;                 %��������������������ĵ�ĻҶ�ֵ���ܺ�
            suit=1;                   %��������������������ĵ���ܸ���
            count=1;                  %ÿ���ж�һ����Χ�˵�����������µ����Ŀ
            threshold=10;             %��ֵ����ĳһ������Χ�˵�ľ��Բ�ֵҪС����ֵ
            while count>0             %�ж��Ƿ����µķ������������ĵ㣬��û�У������
                s=0;                  %�ж�һ����Χ�˵�ʱ�������������µ�ĻҶ�ֵ֮��
                count=0;
                for i=1:M
                    for j=1:N
                        if Y(i,j)==1
                            if (i-1)>0 && (i+1)<(M+1) && (j-1)>0 && (j+1)<(N+1) %�жϴ˵��Ƿ�Ϊͼ��߽��ϵĵ�
                                for u= -1:1                                        %�жϵ���Χ�˵��Ƿ������ֵ����
                                    for v= -1:1                                       %u,vΪƫ����
                                        if  Y(i+u,j+v)==0 && abs(I(i+u,j+v)-seed)<=threshold%�ж��Ƿ�δ�������������Y������Ϊ������ֵ�����ĵ�
                                            Y(i+u,j+v)=1;                                %����������������������Y����֮λ�ö�Ӧ�ĵ�����Ϊ�׵�
                                            count=count+1;                               %�µġ��������������ĵ���ܸ���
                                            s=s+I(i+u,j+v);                              %�µġ��������������ĵ���ܻҶ���
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                suit=suit+count;                                   %Ŀǰ�������з������������ĵ���ܸ���
                sum=sum+s;                                         %Ŀǰ�������з������������ĵ���ܻҶ�ֵ
                seed=sum/suit;                                     %�����µĻҶ�ƽ��ֵ
            end
            g = Y;
        end
        
        function g = grow2(~,I,x,y)
            I = im2double(I);
            x = round(x(1));
            y = round(y(1));
            J = zeros(size(I)); % �������ķ���ֵ����¼�����������õ�������
            Isizes = size(I);
            reg_mean = I(x,y);%��ʾ�ָ�õ������ڵ�ƽ��ֵ����ʼ��Ϊ���ӵ�ĻҶ�ֵ
            reg_size = 1;%�ָ�ĵ������򣬳�ʼ��ֻ�����ӵ�һ��
            neg_free = 10000; %��̬�����ڴ��ʱ��ÿ������������ռ��С
            neg_list = zeros(neg_free,3);
            %���������б�����Ԥ�ȷ������ڴ�������������ص������ֵ�ͻҶ�ֵ�Ŀռ䣬����
            %���ͼ��Ƚϴ���Ҫ���neg_free��ʵ��matlab�ڴ�Ķ�̬����
            neg_pos = 0;%���ڼ�¼neg_list�еĴ����������ص�ĸ���
            pixdist = 0;
            %��¼�������ص����ӵ��ָ������ľ�����
            %��һ�δ��������ĸ��������ص�͵�ǰ���ӵ�ľ���
            %�����ǰ����Ϊ��x,y����ôͨ��neigb���ǿ��Եõ����ĸ��������ص�λ��
            neigb = [ -1 0;
                1  0;
                0 -1;
                0  1];
            %��ʼ�������������������д��������������ص���Ѿ��ָ�õ��������ص�ĻҶ�ֵ����
            %����reg_maxdis,������������
            while (pixdist < 0.06 && reg_size < numel(I))
                %�����µ��������ص�neg_list��
                for j=1:4
                    xn = x + neigb(j,1);
                    yn = y + neigb(j,2);
                    %������������Ƿ񳬹���ͼ��ı߽�
                    ins = (xn>=1)&&(yn>=1)&&(xn<=Isizes(1))&&(yn<=Isizes(1));
                    %�������������ͼ���ڲ���������δ�ָ�ã���ô������ӵ������б���
                    if( ins && J(xn,yn)==0)
                        neg_pos = neg_pos+1;
                        neg_list(neg_pos,:) =[ xn, yn, I(xn,yn)];%�洢��Ӧ��ĻҶ�ֵ
                        J(xn,yn) = 1;%��ע���������ص��Ѿ������ʹ� ������ζ�ţ����ڷָ�������
                    end
                end
                %���������ڴ���ʲ����������µ��ڴ�ռ�
                if (neg_pos+10>neg_free)
                    neg_free = neg_free + 100000;
                    neg_list((neg_pos +1):neg_free,:) = 0;
                end
                %�����д����������ص���ѡ��һ�����ص㣬�õ�ĻҶ�ֵ���Ѿ��ָ������ҶȾ�ֵ��
                %��ľ���ֵʱ����������������С��
                dist = abs(neg_list(1:neg_pos,3)-reg_mean);
                [pixdist,index] = min(dist);
                %����������µľ�ֵ
                reg_mean = (reg_mean * reg_size +neg_list(index,3))/(reg_size + 1);
                reg_size = reg_size + 1;
                %���ɵ����ӵ���Ϊ�Ѿ��ָ�õ��������ص�
                J(x,y)=2;%��־�����ص��Ѿ��Ƿָ�õ����ص�
                x = neg_list(index,1);
                y = neg_list(index,2);
                neg_list(index,:) = neg_list(neg_pos,:);
                neg_pos = neg_pos -1;
            end
            g = J;
        end
        
        function T=Otsu(~,I)
            [m,n]=size(I);
            I=double(I);
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
            Th=0;
            Thbest=0;
            dfc=0;
            dfcmax=0;
            while (Th>=0 & Th<=255)
                dp1=0;
                dw1=0;
                for i=0:Th
                    dp1=dp1+pcount(i+1);
                    dw1=dw1+i*pcount(i+1);
                end
                if dp1>0
                    dw1=dw1/dp1;
                end
                dp2=0;
                dw2=0;
                for i=Th+1:255
                    dp2=dp2+pcount(i+1);
                    dw2=dw2+i*pcount(i+1);
                end
                if dp2>0
                    dw2=dw2/dp2;
                end
                dfc=dp1*(dw1-dw)^2+dp2*(dw2-dw)^2;
                if dfc>=dfcmax
                    dfcmax=dfc;
                    Thbest=Th;
                end
                Th=Th+1;
            end
            T=Thbest;
        end
        
        function g = OtsuProcess(~,I)
            I = rgb2gray(I);
            I = double(I);
            [m,n] = size(I);
            Th = Otsu(I);
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
        
        function g = LaplaceSharpRGB(~,I1)
            I=im2double(I1);
            [m,n,c]=size(I);
            A=zeros(m,n,c);
            %�ֱ���R��G��B
            %�ȶ�R���д���
            for i=2:m-1
                for j=2:n-1
                    A(i,j,1)=I(i+1,j,1)+I(i-1,j,1)+I(i,j+1,1)+I(i,j-1,1)-4*I(i,j,1);
                end
            end
            if c > 1
                %�ٶ�G���д���
                for i=2:m-1
                    for j=2:n-1
                        A(i,j,2)=I(i+1,j,2)+I(i-1,j,2)+I(i,j+1,2)+I(i,j-1,2)-4*I(i,j,2);
                    end
                end
            end
            
            if c > 2
                %����B���д���
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
            %GLPFΪ��˹��ͨ�˲�����D0Ϊ��ֹƵ��
            %����Ϊ��Ҫ���и�˹��ͨ�˲��ĻҶ�ͼ�����Ϊ�����˲�֮��ĻҶ�ͼ��
            f=image_in;
            f=im2double(f);
            % 1������һ����СΪM��N������ͼ��f(x,y)���õ�������P=2M��Q=2N
            M=size(f,1);   N=size(f,2);
            P=2*M;          Q=2*N;
            % 2��������ͼ��f(x,y)��ӱ�Ҫ������0,�γɴ�СΪP��Q�����ͼ��fp(x,y)
            fp=zeros(P,Q);
            fp(1:M,1:N)=f(1:M,1:N);
            % 3���ã�-1��^(x+y)����fp(x,y)�Ƶ���任������
            for i=1:P
                for j=1:Q
                    fp(i,j)=(-1)^(i+j)*double(fp(i,j));
                end
            end
            % 4���������Բ���3��ͼ���DFT���õ�F��u,v��
            F=fft2(fp,P,Q);
            % 5������һ��ʵ�ġ��ԳƵ��˲�����H(u,v)�����СΪP��Q�������ڣ�P/2��Q/2����������������γɳ˻�G��u,v��=H(u,v)F(u,v)
            H=zeros(P,Q);
            a=2*(D0^2);
            for u=1:P
                for v=1:Q
                    D=(u-P/2)^2+(v-Q/2)^2;
                    H(u,v)=exp(-D./a);
                end
            end
            G=F.*H; %Ƶ�����˲�
            % 6���õ�������ͼ��
            gp=ifft2(G); %Ƶ��ת����ʱ��ͼ��
            gp=real(gp);
            for i=1:P
                for j=1:Q
                    gp(i,j)=(-1)^(i+j)*double(gp(i,j));
                end
            end
            % 7��ͨ����gp(x,y)������������ȡM��N���򣬵õ����մ�����g(x,y)
            image_out=gp(1:M,1:N);
        end
        
        function g = divide2(~,I)
            I=rgb2gray(I);
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
            nr= round(r*S);             %���ݷŴ�����ԭ�����Ľ����ȡ�����������ֵ��Ϊ�µ���
            nc= round(c*S);             %���ݷŴ�����ԭ�����Ľ����ȡ�����������ֵ��Ϊ�µ���
            A = zeros(nr,nc);           %���µ���������Ŀ��ͼ�����
            SB = zeros(r+1,c+1);        %�½�һ������SB����С��B�Ļ��������ж���1
            %%%%%����SB�߽�%%%%%
            SB(2:r+1,2:c+1)=B;
            SB(2:r+1,1)=B(:,1);
            SB(1,2:c+1)=B(1,:);
            SB(1,1)=B(1,1);
            %%%%%����SB�߽�%%%%%
            for Ai=1:nr
                for Aj=1:nc
                    Bi=(Ai-1)/S;       %���Ai��Ӧ��Bi���꣬Ai����Bi������S����������ֱ��������ƽ��1�õ�
                    Bj=(Aj-1)/S;       %���Aj��Ӧ��Bj���꣬Aj����Bj������S��������ˮƽ��������ƽ��1�õ�
                    i=fix(Bi);         %���㷽��ȡ�����������Bi����������
                    j=fix(Bj);         %���㷽��ȡ�����������Bj����������
                    u=Bi-i;            %�������Bi��С������
                    v=Bj-j;            %�������Bj��С������
                    i=i+1;             %�����ھ���SB�ϼ���ģ������ھ���B�ϼ���ģ���ֱ��������ƽ��������1��ӦB�ϵ�iֵ
                    j=j+1;             %�����ھ���SB�ϼ���ģ������ھ���B�ϼ���ģ�ˮƽ��������ƽ��������1��ӦB�ϵ�jֵ
                    A(Ai,Aj)=(1-u)*(1-v)*SB(i,j)+u*v*SB(i+1,j+1)+u*(1-v)*SB(i+1,j)+(1-u)*v*SB(i,j+1);%˫���Բ�ֵ������A(Ai,Aj)
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
            lo = uint8(2 ^ (8-b) - 1); %lo��8-bλ1�ڵ�λ������λΪ0
            hi = uint8(2 ^ 8 - double(lo) - 1); %hi��bλ1�ڸ�λ������λΪ0
            
            % Perform standard quantization unless IGS is specified
            if nargin < 3 || ~strcmpi(type, 'igs') %����һ��Ҷ�����
                y = bitand(x, hi); %����x��hi�İ�λ�����㣬���������ػҶȵĸ�λֵ���ҶȲ���Ĵ�С�����ڸ�λ
                
                % Else IGS quantization. Process column-wise. If the MSB's of the pixel are
                % all 1's, the sum is set to the pixel value. Else, add the pixel value to
                % the LSB's of the previous sum. Then take the MSB's of the sum as the
                % quantized value.
                % MSB�������Чλ��LSB�������Чλ
            else %����IGS����
                %IGS�Ŷ������������κθ�λ����hi�����أ��������ǰһ���������صĶ�Ӧlo��λ��ֵ
                [m,n] = size(x);
                s = zeros(m, 1);
                %��ö�Ӧ��x�и�λ����hi����ЩԪ�صļ���
                hitest = double(bitand(x, hi) ~=  hi);
                %figure;imshow(hitest);
                x  = double(x);
                for j = 1:n
                    tt = double(bitand(uint8(s), lo));
                    %ttΪ��IGS�Ŷ����X��ǰһ�����صĶ�Ӧ��lo��λ��ֵ
                    s = x(:,j)+hitest(:,j) .* tt; %���Ŷ���ĵ�ǰ�е�����ֵ
                    y(:, j) = bitand(uint8(s), hi);%������ǰ�����ػҶȵĸ�λֵ
                end
            end
        end
        
        
        
        function g = grayLevel(~,I)
            f=I;
            q=quantize(f,4,'igs');%�ú���quantize����igs������4bit
            qs=double(q)/16;
            e=mat2lpc(qs);    %ʹ��Ԥ������
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
            i = find(xh); %�ȼ���find(xh ~= 0)
            h = -sum(xh(i) .* log2(xh(i)));  % Compute entropy
            
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
            
            an = inputdlg('�˲��������С��');
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
            an = inputdlg('�˲��������С��');
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
            prompt = {'X��ƽ�ƾ��룺','Y��ƽ�ƾ���'};
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
                    %uialert(app.UIFigure, 'ͼƬ��ʽ����', 'ERROR');
                    warndlg('ͼƬ��ʽ����','ERROR');
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
                pImage = rgb2gray(pImage);%ת��Ϊ��ͨ���Ҷ�ͼ
            end
%             app.ImageAxes_2.HandleVisibility = 'on';
%             set(0, 'CurrentFigure', app.ImageAxes_2)
%             set(app.ImageAxes_2, 'CurrentFigure', app.ImageAxes)
            [y, x, ~] = impixel(pImage);
%             app.ImageAxes_2.HandleVisibility = 'off';
%             [x,y] = getpts;
%             imshow(pImage);title("˫��ѡ��������");
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
                    %uialert(app.UIFigure, 'ͼƬ��ʽ����', 'ERROR');
                    warndlg('ͼƬ��ʽ����','ERROR');
                    return;
            end
            
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            showHist(app);
        end

        % Menu selected function: Menu_26
        function Menu_26Selected(app, event)
            an = inputdlg('��ֹƵ�ʣ�');
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
            pImage = app.Image;
            pImage = mat2huff(app,pImage);
            app.Image = pImage;
            imshow(app.Image,'Parent',app.ImageAxes);
            %todo: huffman��������ǽṹ�壬���浽�ļ�
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
            an = inputdlg('���ű�����');
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
            app.File.Text = '�ļ�';

            % Create open
            app.open = uimenu(app.File);
            app.open.MenuSelectedFcn = createCallbackFcn(app, @openFile, true);
            app.open.Text = '��ͼƬ';

            % Create save
            app.save = uimenu(app.File);
            app.save.MenuSelectedFcn = createCallbackFcn(app, @saveFile, true);
            app.save.Text = '����ͼƬ';

            % Create transform
            app.transform = uimenu(app.UIFigure);
            app.transform.Text = '�任';

            % Create rotate
            app.rotate = uimenu(app.transform);
            app.rotate.MenuSelectedFcn = createCallbackFcn(app, @rotateMenuSelected, true);
            app.rotate.Text = 'ͼ����ת';

            % Create Move
            app.Move = uimenu(app.transform);
            app.Move.MenuSelectedFcn = createCallbackFcn(app, @MoveMenuSelected, true);
            app.Move.Text = 'ͼ��ƽ��';

            % Create Menu_17
            app.Menu_17 = uimenu(app.transform);
            app.Menu_17.Text = 'ͼ��ת';

            % Create Menu_18
            app.Menu_18 = uimenu(app.Menu_17);
            app.Menu_18.MenuSelectedFcn = createCallbackFcn(app, @Menu_18Selected, true);
            app.Menu_18.Text = '��ֱ��ת';

            % Create Menu_19
            app.Menu_19 = uimenu(app.Menu_17);
            app.Menu_19.MenuSelectedFcn = createCallbackFcn(app, @Menu_19Selected, true);
            app.Menu_19.Text = 'ˮƽ��ת';

            % Create Menu_32
            app.Menu_32 = uimenu(app.transform);
            app.Menu_32.MenuSelectedFcn = createCallbackFcn(app, @Menu_32Selected, true);
            app.Menu_32.Text = 'ͼ������';

            % Create RGB2gray
            app.RGB2gray = uimenu(app.transform);
            app.RGB2gray.MenuSelectedFcn = createCallbackFcn(app, @RGB2grayMenuSelected, true);
            app.RGB2gray.Text = '�Ҷȱ任';

            % Create Menu_21
            app.Menu_21 = uimenu(app.UIFigure);
            app.Menu_21.Text = 'ֱ��ͼ';

            % Create Menu_31
            app.Menu_31 = uimenu(app.Menu_21);
            app.Menu_31.MenuSelectedFcn = createCallbackFcn(app, @Menu_31Selected, true);
            app.Menu_31.Text = 'ֱ��ͼ����';

            % Create noise
            app.noise = uimenu(app.UIFigure);
            app.noise.Text = '����';

            % Create Menu_15
            app.Menu_15 = uimenu(app.noise);
            app.Menu_15.MenuSelectedFcn = createCallbackFcn(app, @midfiltMenu, true);
            app.Menu_15.Text = '��ֵ�˲�';

            % Create avgFilter
            app.avgFilter = uimenu(app.noise);
            app.avgFilter.MenuSelectedFcn = createCallbackFcn(app, @avgFilterMenuSelected, true);
            app.avgFilter.Text = '��ֵ�˲�';

            % Create saltPepper
            app.saltPepper = uimenu(app.noise);
            app.saltPepper.MenuSelectedFcn = createCallbackFcn(app, @saltPepperMenu, true);
            app.saltPepper.Text = '��������';

            % Create gaussian
            app.gaussian = uimenu(app.noise);
            app.gaussian.MenuSelectedFcn = createCallbackFcn(app, @gaussianMenu, true);
            app.gaussian.Text = '��˹����';

            % Create Menu_20
            app.Menu_20 = uimenu(app.UIFigure);
            app.Menu_20.Text = '��Ե';

            % Create SobelMenu
            app.SobelMenu = uimenu(app.Menu_20);
            app.SobelMenu.MenuSelectedFcn = createCallbackFcn(app, @SobelMenuSelected, true);
            app.SobelMenu.Text = 'Sobel����';

            % Create robertsMenu
            app.robertsMenu = uimenu(app.Menu_20);
            app.robertsMenu.MenuSelectedFcn = createCallbackFcn(app, @robertsMenuSelected, true);
            app.robertsMenu.Text = 'roberts����';

            % Create prewittMenu
            app.prewittMenu = uimenu(app.Menu_20);
            app.prewittMenu.MenuSelectedFcn = createCallbackFcn(app, @prewittMenuSelected, true);
            app.prewittMenu.Text = 'prewitt����';

            % Create Menu_23
            app.Menu_23 = uimenu(app.UIFigure);
            app.Menu_23.Text = '�ָ�';

            % Create Menu_24
            app.Menu_24 = uimenu(app.Menu_23);
            app.Menu_24.MenuSelectedFcn = createCallbackFcn(app, @Menu_24Selected, true);
            app.Menu_24.Text = '��������';

            % Create OtsuMenu
            app.OtsuMenu = uimenu(app.Menu_23);
            app.OtsuMenu.MenuSelectedFcn = createCallbackFcn(app, @OtsuMenuSelected, true);
            app.OtsuMenu.Text = 'Otsu����';

            % Create Menu_27
            app.Menu_27 = uimenu(app.Menu_23);
            app.Menu_27.MenuSelectedFcn = createCallbackFcn(app, @Menu_27Selected, true);
            app.Menu_27.Text = '���������';

            % Create Menu_25
            app.Menu_25 = uimenu(app.UIFigure);
            app.Menu_25.Text = '�˲�';

            % Create Laplace
            app.Laplace = uimenu(app.Menu_25);
            app.Laplace.MenuSelectedFcn = createCallbackFcn(app, @LaplaceMenuSelected, true);
            app.Laplace.Text = '������';

            % Create Menu_26
            app.Menu_26 = uimenu(app.Menu_25);
            app.Menu_26.MenuSelectedFcn = createCallbackFcn(app, @Menu_26Selected, true);
            app.Menu_26.Text = '��˹��ͨ';

            % Create Menu_28
            app.Menu_28 = uimenu(app.UIFigure);
            app.Menu_28.Text = 'ѹ��';

            % Create Menu_30
            app.Menu_30 = uimenu(app.Menu_28);
            app.Menu_30.MenuSelectedFcn = createCallbackFcn(app, @Menu_30Selected, true);
            app.Menu_30.Text = '������ѹ��';

            % Create Menu_29
            app.Menu_29 = uimenu(app.Menu_28);
            app.Menu_29.MenuSelectedFcn = createCallbackFcn(app, @Menu_29Selected, true);
            app.Menu_29.Text = '����Ԥ��';

            % Create Menu_33
            app.Menu_33 = uimenu(app.Menu_28);
            app.Menu_33.MenuSelectedFcn = createCallbackFcn(app, @Menu_33Selected, true);
            app.Menu_33.Text = '�Ҷȼ�����';

            % Create UITable
            app.UITable = uitable(app.UIFigure);
            app.UITable.ColumnName = {'��'};
            app.UITable.RowName = {};
            app.UITable.Position = [843 315 178 58];

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
        function app = app1

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
