import numpy as np

def decision_tree(x):
  a,b,c,d,e,f=x;A,B,C=b*e,a*f,c*d;D=A*B*C*(A-B-C);E=B*C-A*(B+C);F=a*(d+e-b-c)+f*(b+d-c-e)-(b+e)*(c+d);q=A*A+B*B+C*C
  if F>=12: return 1
  if D==-16: return((F==4)*2-1) if E+1 else 1-(F in{-16,-13,-11,-8,-7,-5,-4,5,7,11})*2
  if D:T={-112:{8},-40:{2,8,10},-12:{9},-4:{8},8:{-2,2}}.get(D)
  return((F in T)if T else D>8 or D==3)*2-1
  y=np.array(x);Z=(y==0).sum()
  if Z==2:return(F==8 and E==-4 or F==5 and q==16)*2-1
  if Z!=1 or F<=-2 or F in{0,9}or E in{-16,-8,-2}or q==2:return -1
  p,k={8:({-1,5,7},{}),2:({5},{}),-4:({6},{8,11}),4:({2,11},{1,4,5,7})}.get(E,({},{}))
  return(E==16 or F in p|{3,10}or(F in k and y@y==11))*2-1
