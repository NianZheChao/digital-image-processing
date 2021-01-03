function f = yima(root,tempNode,codec,pos)
global ans;
len=length(codec);%字符串长度
pos=1; %下标从1开始
if pos<len
if ~isempty(tempNode.character)
     if codec(pos)=='1'
    yima(root,tempNode.leftNode,codec,pos+1,ans);
     else
         yima(root,tempNode.rightNode,codec,pos+1,ans);
     end
else
    ans=[ans tempNode.character];%把两个字符串连接起来
    yima(root,root,codec,pos+1,ans);
end
end
        
        
end