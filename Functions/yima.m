function f = yima(root,tempNode,codec,pos)
global ans;
len=length(codec);%�ַ�������
pos=1; %�±��1��ʼ
if pos<len
if ~isempty(tempNode.character)
     if codec(pos)=='1'
    yima(root,tempNode.leftNode,codec,pos+1,ans);
     else
         yima(root,tempNode.rightNode,codec,pos+1,ans);
     end
else
    ans=[ans tempNode.character];%�������ַ�����������
    yima(root,root,codec,pos+1,ans);
end
end
        
        
end