function f = loop(tempNode,codec)
global di;
ab=0;
if ~isempty(tempNode)
codec = [codec tempNode.code];

if ~isempty(tempNode.character)
disp(tempNode.character); %�ڽ�������ʾ
disp(codec);  %Ҫ�ڽ�������ʾ
ab=tempNode.character;

di(ab).code=codec;
%disp(di(ab).code)
end
if ~isempty(tempNode.code)

end
loop(tempNode.leftNode,codec);
loop(tempNode.rightNode,codec);
end
f=di;
end