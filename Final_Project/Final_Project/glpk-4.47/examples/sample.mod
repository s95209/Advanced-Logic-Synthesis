var A, >= 0, <= 1;
var B, >= 0, <= 1;
var C, >= 0, <= 1;

minimize

value: A+B+C;
subject to

final: A+B+C=1;

aa:B=A;
bb:0.25*A +C = B;
cc:0.75*A=C;

end;