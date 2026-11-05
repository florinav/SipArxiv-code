(* Numerical instance of t:KBoverlapCusp and c:Nin: reproduces tab:KBoverlapNum and tab:KBnineNum of SipArxiv.tex in 60-digit arithmetic *)
wp = 60;
phi[q_, a1_, a2_] := a1 q + a2 q^2 + 128/343 q^3 + 1024/343 q^4;
F[q_, a1_, a2_] := phi[q, a1, a2]/(1 + phi[q, a1, a2]);
{al, be, ga} = {2/10^7, 1/10^6, 1/(2 10^7)};
f[{x1_, x2_, x3_}, a1_, a2_] := {-x1 + F[x1 + al x2, a1, a2], -x2 + F[x2 + be x1 x3, a1, a2], -x3 + F[x3 + ga x2, a1, a2]};
X = {x1, x2, x3}; J = D[f[X, a1, a2], {X}];
(* cusp: f=0, J v=0 (v1=1), adj(J) row 1 . D2f(v,v) = 0 *)
v = {1, v2, v3};
D2 = Table[D[f[X, a1, a2][[i]], {X, 2}] . v . v, {i, 3}];
adjrow = Det /@ Table[Drop[Transpose[Drop[J, {j}]], {1}], {j, 3}] (Table[(-1)^(1 + j), {j, 3}]);
eqs = Join[f[X, a1, a2], J . v, {adjrow . D2}];
sol = FindRoot[eqs, {{x1, 1/8}, {x2, 1/2}, {x3, 1/2}, {a1, 342/343}, {a2, 368/343}, {v2, 0}, {v3, 0}}, WorkingPrecision -> wp, MaxIterations -> 200];
Print["cusp=", N[{x1, x2, x3} /. sol, 12], " a=", N[{a1, a2} /. sol, 12], " res=", N[Norm[eqs /. sol], 3]];
ev = Eigenvalues[J /. sol]; Print["eig=", N[ev, 10]];
(* Lyapunov-Schmidt g(x1;a1,a2): solve eqs 2,3 for (x2,x3) *)
c = {x1, x2, x3, a1, a2} /. sol;
Y[t_?NumericQ, b1_?NumericQ, b2_?NumericQ] := Module[{y2, y3},
   {y2, y3} /. FindRoot[Rest[f[{t, y2, y3}, b1, b2]], {{y2, c[[2]]}, {y3, c[[3]]}}, WorkingPrecision -> wp, AccuracyGoal -> 50, PrecisionGoal -> 50]];
g[t_?NumericQ, b1_?NumericQ, b2_?NumericQ] := First[f[{t, Sequence @@ Y[t, b1, b2]}, b1, b2]];
h = 10^-6;
dg[k_, t_, b1_, b2_] := Which[k == 0, g[t, b1, b2], k == 1, (g[t + h, b1, b2] - g[t - h, b1, b2])/(2 h),
   k == 2, (g[t + h, b1, b2] - 2 g[t, b1, b2] + g[t - h, b1, b2])/h^2,
   k == 3, (g[t + 2 h, b1, b2] - 2 g[t + h, b1, b2] + 2 g[t - h, b1, b2] - g[t - 2 h, b1, b2])/(2 h^3)];
{t0, A1, A2} = c[[{1, 4, 5}]];
Print["g,gx,gxx,gxxx=", N[Table[dg[k, t0, A1, A2], {k, 0, 3}], 8]];
da = 10^-6;
G[b1_, b2_] := {dg[0, t0, b1, b2], dg[1, t0, b1, b2]};
Ja = Transpose[{(G[A1 + da, A2] - G[A1 - da, A2])/(2 da), (G[A1, A2 + da] - G[A1, A2 - da])/(2 da)}];
Print["unfoldDet=", N[Det[Ja], 8], " baseValue=", N[(1/8)^2 (7/8)^4, 8]];
(* implicit differentiation of the resident equations: g_xxx and the unfolding determinant *)
Yv = {y2[tt, bb1, bb2], y3[tt, bb1, bb2]};
Hy = Rest[f[{tt, Sequence @@ Yv}, bb1, bb2]]; G1 = First[f[{tt, Sequence @@ Yv}, bb1, bb2]];
pt = {tt -> t0, bb1 -> A1, bb2 -> A2}; rul = {y2[t0, A1, A2] -> c[[2]], y3[t0, A1, A2] -> c[[3]]};
Scan[Function[ds, Module[{ud, ee}, ud = D[Yv, Sequence @@ ds] /. pt; ee = D[Hy, Sequence @@ ds] /. pt /. rul;
    rul = Join[rul, Thread[ud -> LinearSolve[D[ee, {ud}], -(ee /. Thread[ud -> 0])]]]]],
  {{tt}, {bb1}, {bb2}, {{tt, 2}}, {tt, bb1}, {tt, bb2}, {{tt, 3}}}];
gd[ds___] := D[G1, ds] /. pt /. rul;
Print["gxxxImp=", N[gd[{tt, 3}], 8], " unfoldImp=", N[gd[bb1] gd[tt, bb2] - gd[bb2] gd[tt, bb1], 8], " gxxImp=", N[gd[{tt, 2}], 8]];
(* eight stable equilibria near z^S *)
sa = y /. FindRoot[-y + F[y, A1, A2], {y, 1/2}, WorkingPrecision -> wp];
starts = Tuples[{0, sa}, 3];
eq = Table[X /. FindRoot[f[X, A1, A2], Transpose[{X, st}], WorkingPrecision -> wp], {st, starts}];
Do[Print["S=", starts[[k]] /. {sa -> "s"}, " e=", N[eq[[k]], 10], " maxRe=", N[Max[Re[Eigenvalues[J /. Thread[X -> eq[[k]]] /. {a1 -> A1, a2 -> A2}]]], 6]], {k, 8}];
R12 = A1 (1 + Sqrt[al be sa]); R23 = A1 (1 + Sqrt[ga be sa]);
Print["sa=", N[sa, 12], " R12=", N[R12, 12], " R23=", N[R23, 12], " a1margin=", N[1 - A1, 8]];
(* nine stable equilibria (Corollary c:Nin): shift a so that g=0, g_x=-eps at x1=t0 *)
eps = 10^-6;
dA = LinearSolve[Ja, {0, -eps}]; {B1, B2} = {A1, A2} + dA;
c3 = dg[3, t0, A1, A2]; off = Sqrt[6 eps/c3];
st3 = Quiet[Table[{t0 + k off, Sequence @@ Y[t0 + k off, B1, B2]}, {k, {-1, 0, 1}}]];
sb = y /. FindRoot[-y + F[y, B1, B2], {y, 1/2}, WorkingPrecision -> wp];
starts9 = Join[Tuples[{0, sb}, 3], st3];
eq9 = Table[X /. FindRoot[f[X, B1, B2], Transpose[{X, st}], WorkingPrecision -> wp, MaxIterations -> 200], {st, starts9}];
res9 = Table[Norm[f[e, B1, B2]], {e, eq9}];
sab9 = Table[Max[Re[Eigenvalues[J /. Thread[X -> e] /. {a1 -> B1, a2 -> B2}]]], {e, eq9}];
Print["b=", N[{B1, B2}], " sb=", N[sb], " R12,R23=", N[B1 {1 + Sqrt[al be sb], 1 + Sqrt[ga be sb]}]];
Print["eq9=", N[eq9]];
Print["resBelow10^-", Floor[Min[Accuracy /@ res9]], " sab9=", N[sab9], " minDist=", N[Min[Norm[#[[1]] - #[[2]]] & /@ Subsets[eq9, {2}]]]];
