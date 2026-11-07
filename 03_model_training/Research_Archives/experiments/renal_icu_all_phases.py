#!/usr/bin/env python3
"""Single-file, reproducible renal ICU-transfer research pipeline for MIMIC-IV v3.1.

Contains the original Phase 1-6 implementation in compressed embedded source
modules, extracted in memory at runtime (no other local .py files needed).
The two model training tracks are exploratory Phase 3 and the cleaner Phase 4B.
This is retrospective research, not clinical decision support.

Install: python3 -m pip install numpy pandas pyarrow scikit-learn lightgbm catboost shap matplotlib joblib
Run: python3 renal_icu_all_phases.py --base /path/to/MIMIC_IV --output /path/to/results --threads 4
Use --resume to skip completed stages. Use --from-phase phase4b to start there.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import traceback
import zlib

# Compressed copies of the individually tested phase scripts. The single-file
# launcher executes these scripts with their original command-line interfaces.
MODULES = {
    'phase1': (
        'c-pO5eQ(<~8vj3^g0}*bjB}K=+pt~^K49psKryTshTRSapCK?3ZF7-Gjil0eS@zlQ_mI?^EqldqL7Px~Uik2NC;8$pH?1`{cT(L5^}w2g+iUgt'
        ')hC~P^8KD$!Cu}nBb?E;5rvZv0$++(?A@DxT|2{--3r4huF9I5dsgVZHjZtzVc)&`?%kW~cR#VuvzHmTUCFv37&p6y8!Ni+pRHC`Tcc~%aJR4I'
        '9gBeP@!26YSb$rGe@)rVskUwoG^LuWtE*CMS)~i^M0y9Ym#o%hTj8&DqF&ryG5i><on<Sw4jA;C-LRzDAFR|$xJ;W}LNi-g3F}9-tYsk|{1_Wa'
        '2C8Go#xFrTr)8K_VY#q!2eR#UD<27%CRt%0vfZy>>4(W;K&77sf_hFjaVr?4v>Eo-$YwFYms{3k(&i+4dc@0bUbxmM)&xy(iCw$}>U-_p>Q<Hi'
        'G)9|ro4nylVbP#U_7s+WzImdRpGnR&WZp1P!0u0^jOS$0>z|i1+F!ygs<u{k$x1M_t8@|NqdlHCT0k*UxitAf@KUNBl+`TRwrcHu^`mJ;z$v%j'
        ')0CM3QVtPQ;Ivj?RlDk!uJnhz7PU5q)mvWKNJ>U2$R1v@L2Ko9U2FnZr*i^2S&P(9FPR`^*;P_#VG5B1R#Cy_Hfn=gD~!uqC4Xs&(yIY-@H<h('
        '{ljL&QCAySPVsb+adk)+3_^ADW!qGeG*oubLC9dmpLo@Z&U{av{qrq*0`)ViTkF`JU_l6mC}K*hYt>d&Z?(HtR-M^&>zdZZCBjEm-<Q&)h*_bW'
        '^;(Al9;J1;z8@I@(JKkVo3!_XWRJu_F5ZDLUMIG_`x)6Jmu0d{c7l8f1Fi~WkDTws@Np2_B%81TFY7bR0g;wsc!vERz%U{#M`pyhl(yJ+V}VQ)'
        '$33Uq9h&KccT^0L4XyFV%%arg;~Ak#$#+UyVRKoOm*fUGFLWv5@Ie^s#R&Xm(H<>;nI!zNjvp?ngb?B)DsBK)58@Nq>1;H`#J4-!V$tzBBcsnz'
        'K?u@DC(Fc&bNJW~Ezn6k+F%!dzAUWR({oTfe7itubE)Dr^ipoeq2rO)1feje98(E}C|wL$HKZQI6k$<FMU30!b23dro+PZ582%xj_#-~X{)^uk'
        'Txm^zdnd|}b1DAWrykxr)M!4P0N6j+gu!By6}mxo7>n}2D_Q2`Z_@E9$8Lnd+(f7s>@AlSD=K(m$+QBk+9KmPWLp-k$ttPtZFaPYTA1C~_FAo`'
        '=HKfND|nEi;AUPHq_I^}Ba7uos;Ib0uNVXr!hRE3519bPZmOu#Wt0=x^%#+6&Z&4OGKl%|b*EdJh+3V~8AkzfGX{*YydI*Fge@Pl?mOd_Qbr07'
        'HB_k<83tL~G{U4|44<ob0gQ~i7ypgeiN{!tJO+_AQJxs)eGNjpBaAH4xMT#MsYU!=f-p@ZRwjN)|3(TJ%NrKt#@P?j?bGDT*Iz&;#iKwA_S?9J'
        '7_)l$*&p$G|MIH|uc+et@=r+h<^NHt*E3Rue%9O4I}KF#8;N>0ZEhxh<H^^_*WbPi?t3pi8$X>%l(xrB?;41}*U5zJ&9~opGOYLUWYigYyq(c)'
        'b9YG32@2mubs=VP%j+QMGOWAFYW|omX5-o~;Dh=+^D_aCz@EnUB(x@niQF%9K3}erds!-R$o)>erc{$zQGq=5@r~_x<1ap?FKjD(c{T$ueD+XZ'
        '+W<s%Aw?|xSgYy~G!uqELJ8rJd6l&4Ug-}iiFx1!9e=20<WxDt&PaE-MN?E%Z7JBW8%BP|zJC87R`Uw2W-ZE_aegH(27Rh2myxeQ_khOQ?YRPL'
        'Kmc~tfjRWHo&@09A9~Qn5&5uK$j>3?ho*o^W)|%=sw6Z!nDht`D*n-Atp4L_G@3>&8Yp8>_5025#esT4E1ahxC;xpAGVni=anu$}nw*ZlC!Yl3'
        '&sr)`Mzr9><U#9kyEAs=`3`Qg36*T<OD0|(PF{3|Npjz3CFH!x;&Oo_ODc<&8I=+O21!RYAXb?Y=_!DM0qH0LqH!wMVGi~G1ZmeJ^_v?Zbrzjt'
        'HDf^3G1^;)OGX&)X>2gmstU8|Vy@}=od%t?rIjk3t}|bfx9D5)5M2m?d?10^a^IVb)5g?SejoeSAFep1Js}-QKc(ABV_l@QXe{#wI+WRT?0kJM'
        '-L<frMOGT!<oyxPrak&%Jwu9wP5Bb%(0Hm{J$0bl#EalH5~rT$6T8Qo_3=t<-Q`5?Gjey?lLwi)maTIi<Ww{)KVnph%JFosTeK&yz8G;p%O-!t'
        'i3$h5kEQ?WRoAY5LkBEIn9yV0FllkD1&dPB;S9x*+>-iSe_16exs!LYLV{++UKIC#XFeP_RMDkeQFDo0jm%0dKel#Y`Yy6S19nH(Bbdz2beKc4'
        'j*#<!v4V!yy*n7jqdsDBIz3B08xUPbFMov-13kwOwUoR3)ffBACyrw;nF@0Pot0@$BxG*Tc+#?^{x=W2MgoOf0&S%?%xZqY={e{GT!|2ag9CML'
        '@lS40CVSiZD-`c3cLLEa*jKOip&Bdg%PzSMgs&Un%c^)l8oQ0WO3mmF%t{>isQCIj6($|kUCy>`RdpVMdZWb7yIOD$56nAY-T|o?|9SO~w|ja-'
        '+@0#=<9+Uwb`t5>caAV~tv&9&IzukcZQ1u&9Tsv}H$UmC3ZalLML9Xbulwn|H&E^k6r`Y2dGN|XJnHL4+)zU6t^&n7>1iBHXi<4bv`6l}_Y#p&'
        'y6bK@bq;q5eY=OBiqda<G1tY%pxH`lem$)yr%iC+JJm|ZbBx*PBg!!&=2#USr_L60oG!3ps}=ifF~=}m(d7Mo&yiT;ZZqEI95HB&K;03l2o+^p'
        'X?}52wcrXo=mwQ;Wum%}?yX?&QLwr`{k?{Nx)uqbuNanc1NtBMmO$YaRcYx0IHZ43qBYm%ddmynZS%V-Ms^R&WAUYjgmd?5(;B%<!ky-D)A=#C'
        'j7O__QF@PCpNKPYTklhLz`k&|IOEq7R!3KTQ+cKrct^n5Tf89IlrGFo(0u~#9N<EgIEPV*v=7ZmJq>s0<CQwZ@!i@iZN{rAJuBp#mm(8<Hj;A4'
        '*;Vh{lhk`EoUcPTI;~IpyYrXW<Nj(n&EjV)zR3LLpruo}R>xM(qKhr&T$1jl8g5`)mxdREo1~$a!<TulzMjNp5fx@1-%E^~_98!zaDl}O_Jg2H'
        'w8Eh>JB{>W1vdpszc%-_;f1(m9H$6JU83`CAF`^u?jJCRERl`!{{@)>LhhMflI)l+B)*JZKT#t-80pY!i$@&CFq4%nBuE2JV5Z#a6;7GRkeUub'
        'tE6??>#m-}N3#E_e;e&Ye7HeJt8bJ3O=pC7X$kac$f<cw$BaCujGyE=HOP6M#78@Ny){%%<4bP#wb`{b$iMfHbTL9hUysTb;c0SxO>Y$b>hqUY'
        '!Ykr?5~aZvJ^11%BC2MU^smSM33RI~N}LxNhixW9b3yFVPTf%@bWrctih*W!lqe%t|A+Y6xQ*~SMiW_hUh%_Dr~d=_3Ul5'
    ),
    'phase2': (
        'c-o~}YjfK+7X7YYfhMyh)zGvg$B!!2&ZJ4Ro0(0TNxSU_kA?%0ki?n-S%S2qdi~$`+zWt|Ea$cBX>Ag~ec<BU7nsN29Ji)EUgY{%>1$DKn{}yA'
        '29F*+dbgHFiOE!K)TXSrM~kx6naUzjD0wAU>PX(mT8Tz&s<M`aDC9ycm2BEt8L=#D@#giL*DsD<|0Yi2aXc8jS4PP?U5io|+gSYDU>VV*Ygso<'
        'zER@&+n2&LWv%$_hnbi>U1N#PHnP5o#qVpS2mO3*gj^V<n@B8WUKo)UrBRt!Y(=J)vMrieyeuI^myM{E%tYBXX$eY5RfUOZlY^YJZ-lH@m8^~O'
        '?|+z550-Vg5tVG#MZOSj;vHW7BHeDPt&osV`JoEhVgUbD*<kRm=RYQIp1%__aXS#FqwmL)X^=vdCfB)EL3BDEjn1aQqSbgFeSdK>4NRHk?S@~^'
        '$J3xH8)=Lm84>-uDC;}}HL<+7z|bO3WxXi1Y>4OR;(PE(Ws!Qav(Yrz$dy(dqjQW@>(W$freWnyAZ}H(DU8Tm#^6%OwDDZeE~bIhd8w0?tVs8h'
        '5h!jJDLqdn&|?jttcns?VwM;j^Pp_Xw9Xq|cXkTN1$Y*!F+6g9J`L*ZhGkrw(r${pZowpJu=&F6^9=lDv1m8Bw%Uy8z2WC`(#+WBvlAM>k_8N$'
        '_$^AKoJJ=Qmh0LIehO8rR_CNuV>wLNHr51ST&3m2j`ML$g2CP3vG^S)AdFn9W-Hh+OeADg6*&M%_!e^6s9NOkj^kG64N$JqD-(;irTDXzy2%R#'
        'z~)tz>s35>|NQMge)@%Q(rt1Y#;4JEG>jhJ&NLij7C*e8;rA2!KG->?X*i0`c#W}Zo;sRc%uK@*UNktcJ(OTEM9^w41aBT*({K_^n6$&?G&~(e'
        '<5OPMW1Tsj(4fZ{n2kVdHQqr9=%e#fEZl(;i=i<(JGKn7^6+HlPnl7Va|}+b0r?1t?Sys5;ZDOd@VMX??*+Wx?xhDP?bzzbXV1F1H}9BO;!mow'
        '+VkcI3@E=zOcKUvmM!Dld>UTBRu?ovj>P*o#p=7k05MDyWeP>Y1rB^T9SHnTT&+t4v|I}wkLcGiwy`xhbJfIUy$X(lCGchNcav}OG{43Zf^gs#'
        '?+-f8p1Wz6%lrfL58~8Z$E&}Bh@W;>(ho((87XrdkP(C9FrdhKjgV}DCzj7}k0M_(QLH}XrZM4ginxq?p?Q>7)-p%Zc?Ip?mdz`2te@(-ti#|%'
        '*%ldx@@1|wK?09EX{6mH6rBr7QMG{drnY7qT1(cgw&oHQrL}QH$)IYHJJNNluM(60<xOlS;>}f-*CCPu37Gv_w<=<_lk$q+?9QrMWwR=aDZeCG'
        '{Aa70(2Af#p}&dGU<obGQzS4GyQP8UQma4PoH*SErd|A@(k96>SbZ(CjeY8PQ0fcr2F{u`wOQggHy}17pWf24XgAuBjoGHZ$)Z)xm~XHYaYxt`'
        'KNfE7)EUlmJUU=D9D2z-?DT_VEyH0f^)?&|U}$eLYY`R+$C+R9kV<b$eS|;>qX@t2-olsn_CF_+H)WE6D-BpB_tr#;_?lV3q8zWsRCNfch5A4`'
        'AuRt<0_dd{KmPpdKLw%}0`oFgHKKW~=p4vOX{nc3&CiI#h~mtQA7`xI<(D1Xd=5iaFaXn=vaM4!+x2m`;aoRi@E(XFiO`QBYNkPCNup)Zn)MC{'
        '4PsRa`_3-s{GP&ZkbMl`RF^j<A#euM5u%P;m{49ypj!e>2@9p>d3lXON-heOEVdnNdFC1!C`(ZAs5(y*%4kepOT9wm99clpFofPYB?>e9s9{I}'
        'e(0({Ja@HSzye#Psa&y-ZMg~NT?ou0wh0+Wu~KA0?}i8lp|ezL6zf61ED1UKtoQk-C~p#5;Lcvj0%s77cfy>v<mHYb=TByZ(joKc!yCU^#xaic'
        '(mvU3!as5yx?*tnt)1$92XhuyZfLj%6X$Q12(@;zof0B}##TKj9oASW%P}hT$Btkgt;?Gk@~Kv+hWL14!x{3Y-Xi%Y$R)w>0XuvOLJwQ!%ap&|'
        'LBfH1Tl#K6=i6ziaLU2(KC2!`A7Rxk_&BS_;+F<kZv-tfjoehCDiH>@l(Y=G3nMnIX~aU=4hDr^*S~NR;-Wna!z|*@K&9JVjG#{8^xJ!?cH?^~'
        '^*(Xox40L24w-#y-2=>fqSe7s9Afpccn_buMi%aX!TT57P3?Ac&upc<%xjpDCS;nb3WkjDpXVChjHA3S{OF-A9O1x!#otU7jSPv>%qCBfx#Fye'
        'xvEVlD4=2whsP&pqtR#1g*ed0N1fx+K+%L?da(K4PaY1X|DT6|AFQw!SvIoj0?Ym7T?q9~;KH}FnxFoRH|-ciSBO?9zaaY_kB4E<NW^DO3LLa$'
        'o8`?j)GWfFzSE7cjZvuq7J^B<SBHk!ZA2@Ek?xXUysFE#T5LmWjNRN2Wp!C;>7sZ{!5#GmnTZO`Jy>6<?TqRiuoylG#D84b!@~9!vhJ5yKA-U;'
        'aL|N!jYh*Kl%#dP9C^fBN-p?VZm*`-m!mmv<|?|TJ=iwSD|OdXuk+RVA@cE;kw1)Fe>j9*sORz94I>1Q+)ttSlWG5>V5X{c7tnh)^$8?KyR39('
        '$v;&=Ek02taA5dIu`u7$DgWDKG@Fg4uDn_XFSr^IxB1;OLCIa1R3aH>;<nrVqb}(`nopy<;Gpnwd-YrLb?tl5L%*)F6{^TrIbs3bJegR$pt}I>'
        'c~EC^Y=jd4fp8bN3JmOOLxHR%W9N`WN`8!`C>TR8Gs;&Q_DIkasSi7<A^dfE_4F*C93|<haufcKwtvv*zRrGGTFbDbBgm)a27NqslrNWxS}L(o'
        '(zLbO_&{J|M6;%A6?;b)J6^xYYh{B6+lDUt$nqQH;JkuSHChMV7KLL|rfFN-fTOoP9|mu&=$4Mm_!6dlRYOvEc}p8*a0k;X?3a!|m|oxQx}GU1'
        '!-qBsI0qcM-R^LvC`;w1?IB$UIB||cJK#P#vWI0O6|>mnA5gyY>G%l8)*r^+w6V6`QMec8JPJc*@b|HAusY;F(EE9_S!-pgVT`~D)OD_I#Dglc'
        ';emrqsS#IE^T0GXlhvwLE4txX`PPy3xOCdgx$;5{6#B-o2O$|W;mr{MzYKF7x?aIO03`#-RG|4AXwz<xWpiajh4SO<`0RK*Iie6DfIW2n@LoZc'
        'i3C;}fY22hq|aqQ*G!x!)3ym&MWT!_j5cNzokinGG<iCY?By}a>h)-pjLyhmMrX-*LJ#9fGI@$rR3of5_K0xSY0+jXK?Ft_x#OX*-u{Unnt^qS'
        'N;LKLHSoc-ef$Ok&{()e0sKkRpPYZo4{SNh$}}(?dLy*+Hco`S2KXN{njFD3`LkO~LbtyN4^7|_%?;XPUS8$bWfNKq_bPkp%RSxg3D3SrO%^uq'
        'g~c7qWpE3ocgZcm0F3d3ioCfC?2Q#3=~yB+Oa8h`KkwD&;X^|>P<;A0<`H4m*B4}bu3Tr~ZY1QPU>Mom?9Z`Phf!bU(AtDhhIKBJ_4Ef*j$H3E'
        'r8*}V4nMxB%YZ9kVA~DER=1;W-paIAQiZGxpw_Q-didpWul2qZQ<RlT;CP9v)=@3qtP$%&$1`3vD54|JhqXF(a}gG6>B=c~%}#?)qs@Y?Lzx3n'
        '<n|oh*D4CMN0<!&2M}{&9|RZyYmaQcqpzJfJO=+sjBkoub51a%-n`!rH{*p`HoLtzdho6lca$vUD@k3@d_f-Af1yLdB4sheU(rx)l*`?u9B~5Q'
        'xSK*(0{YfB?(dlrd(o4!rBX6cCxZH#82Xx<yjh1qVq*Z`5bllZU^w5i)n(6lK6_{oC@cyz!=s_U5?bV0rZi$#p0)`}DQ!?^AQ~s<lePD&gywSX'
        '@(S`ty7teIvF>WhI{&NG2^6%Ruh2090j=rpT|0qs&ZVN_K0oxT9$daYHwfXh#3kXM{g+7lDb-)VcG)=SGCNh+p6&1(F1QIuY{X97=dW7tEAXOR'
        ')91=m)>*EBnr2J47^v!ZMN>5FZx&WAEq3|%Ga_DVhNHb5Ic6#-odHIAbx(+IsYg3^$o&Lhk6!!J!eG#&a(Q$E&ns`3HrRJRDn<PL{JrYT*AEGm'
        'f!%t4Lw{JTxgzrRyYGCbz+mEHq`wU_386b-Ri@N(*bTXp^3~jMCm+yH;9G?qMvmN{{19liN+u3A{elAk9(l6<3jdzKn%i`h@$R|d;ZK)IrhulN'
        'GiiT7YcIrtX<x6|?xp)i-<9Sx{KIVL>Zhk5Ga+iA#=<EJzOZ=)7JT>Ce>!!(cmM2M*-fA85hHJEl;$|k4^2oT@P|!hLc4~~Q8%;6aQB-13zr4d'
        'gJjbn`YPxQ2Lm8HNvOI?lG!Xsk_~K~B!Ml^tC|022l4ZI)l$@bM{l)D9aSu|ERk+1433T#{3W~DR%%B75FzS59PNCOIv}#2p}v~TZuP6lxSv1b'
        'vc!_Yev$hJ8MD(7e}1;r=d6l(^F(gkmgzg4jQLBQjNPX?8M|+FGT!-EH~24Tn11s'
    ),
    'phase3': (
        'c-o~|ZExH*68`RA!L-0hy2^^}B-h^NbU>S?MS!*!B<-Dn#X_JY?yhQyQbo#}jp6@(&+tVnt=H)V7X)639L@}f&%9AzK6yH*{%Kyxr$XMcZm+kk'
        'e0KK57hk;HawXWa1*-+W;Tv(zcf1#D)3-w>N~U{`HMhL3O0KI`KGi~LCNMCV_k_I@O7OnevR2mn1RT$*rfYl6c)#g*uSCp#SFJo-_ie*EuD5lS'
        'Ge`0kubx~EO}A%UG1>W{j>{4Q_}`Uwy}E&#y-b?66m_OVEeg`Y^ZSvse!m^o>sq|;>PjE261uMn<*EC&$oNoXs%U#?$!|pu17uw<ib{d2pQzhS'
        'R<}yUEblAP+hH?e+1J%(t2cS$`Tq3dcR#<Xxl+{{<S~QndY>uXc3mYmmfyjGmPf&L-nKBU<ME2?@96!&=j`nL>(}3>&#%tTO0i~cFX29BU9>o3'
        '_}2?P^pf2LGFVhnhqaL5K8j)%bgim%bt{ye*e7aeLOauwCPEe>Sgh-o(_|yKq*=q%{|op4dwHUn>7YYrnDL?ON0LX(>4@jNja~i1V<V63{W*9-'
        '{|*A#B}H9zVHaPD^KatIbNY#*=-wF%AP#{?<t)_>IwkCxu>^13pre;GoHRFO)rSDCkXog`_JfGUhf3+Jy)kc*VXzh)wiYVw%A^;(%&_2(LFn+n'
        'wIj#6ik=4cA=83Rme*O$^USYL+`7QB%|ZEREH6FFOK_59!}}Yuaak3*?T;BkmZ;Tdv7`k@T-I&RByU7qu3077G7P-_LF}KRI1EVrx{@M@>4`=x'
        'yCAl|(P}ZCd$nfeI@$8F$*MA`l;mNQaJdg725FP2a)2YlJw)jV77RfNLF?aeA@REJ+df<ecckF`0{WR?WtfaBnPcYm1KEKZ^4}3WXpr_a9!SyF'
        'LnGBe(ebE9lCp2R(9xw+MhEWW$>^#P|EGXCuhTn_ttKJv^t_0Nb_#wn6s#E#YKm5BI0k#L1B7~>(h(vQ^cxX+#o|_`fj4EmZFf*ktaT8({2<bx'
        'l_Jw^hMy@f`r~+66_?AY`tgBjP!&Ulq$AZDLE!3n_Vv{^SglC%BTH6RkxEL0kWXm>n^8n0U7{x`JsOk}QjW$QT2I;UUAtpVr5di0@p#cA{4nD>'
        '>fkGRRhHsWedD78ePqjq*|oLY3=g9KD^CrotBR@}RJtsTrxt{SYuYY(i*9d(q>&=2+nwm)UqmzxFK6PNHb@dywX#yh);|LU{(0N>)klPMA?=28'
        '%ElTfQD$=J71<uF@Vc&ggim^2MJs<22enUG`E9Eaek)#6NG{q&`08ZAg0GbB!^85(nt|MW1EmUHr!~TLc%9hlAS?C06Dq7^Q4giSskG)$pcoh-'
        'ALGF--a|p}JBPrB0b{;{fCal=Vsw?-pM)zC5qf@6vaaeH@hr%)pWnQH|K^t;vkar2Y?NkYpzaZ1`|$J%@6{904i-zdh*a>83GcdkA0pJl@@_B|'
        'WKYBpV`Avgi>~H{2$x8hm0ZW?_IE`IRB{I?Ue{ny*AG>~hF4>(uIun{o_-B%lU~mO>!*-lr>8V!J{h0w+%L6a_SPFtxQn+fa69_C5>(3!vO*$5'
        'HS;9ByNECEPoeGghYo;1mOy#T`~8TyU8T1yZ}F#~Dr|IdQyVL*%)!NdVIlhTK^YpN_@Q5wUdB{FXR7)r(&X|*jD3`CV|+)XRN%R&DQ>~?gC)j8'
        'D#ZsK8t#eV6kf;s*ec>Yzy!e05dAdy=7jjKE-g2W3T(&`PU*)=Hv+u$q^jFu>C{}s`=fWO*?chaoRM?Pp4`OJ%38f-2;Q>^E8QNx+r(X9)(4A`'
        'O&lD$lVIUk!~yLURL*|)qdRlLm1Q<b!I;QTO?Y(wBS$bT4W_1Bv*b&@T1D(HkIm+o^|1REFzQKDALyVqBApL!$MqI8KWLRyC?QoC&1xuTc(QLI'
        'J=mY5SD&bneBM{4W5XXYBHrNKDRHg)`1+Q|*FeVddbXpR)e3R4+q+zHWAlhTnAvf-V^}%=1;;%&?uq06nB!-$`jY)zp)ZmfDUg5>Z{ZmD(b$az'
        '7^ZAP%6{S0`QYalMI936;OXYT$+LM)aYt_(l^Di1LUbv%?(@Kso)L{q$g_>nKhxymYK16k^2`Zil}q9>Gf_$V3MW)9aw71lPe($G8E&?m-{k4M'
        'yM%kgEEVbziu0FO7Z<VVb+P}fC#Oy>p1p{X4l)d*0Zgx+#SQ<Em7>$z^!)PBh&6DhVq4W^rg+oUApQCxh865vmgDpU*nJbjCGFVkC0qC6k95MX'
        '7wFD-lb1X_5ocnV{odwC4%CF`Ep;a<M#q>_r@Dt?k>4<~sWz4)_)xQowqK{qkPHblMJqdK`{6Wr&~^s%UUOn#--V@g3NZ%ioEJAZk9or$e*MB}'
        'Cej~xO^waGgAXsFRpeml`}X-kO7xem=*|VRb%KgbmUDL;z9@JKoF*;>;P}xg_VC%z{GE9I$JC5Ch|S`dhY<>^k4V=k{rSui1ys@G9L=r5M~l}y'
        'eRwkwA~5yF5<uW{Akc#A#RDWhkLxSK;W`62(!U)JhPwD{JxFtzlLkMv8`I~{PxkG*Z-09gj8GDyv`8AcSQuqRJ4k&BSpN9jP7JU{3j<-1ff!6T'
        'w&xZZBj_g70G>f#JjY6g@Gnc1{{7;V%<eWrbqYwkejOlfb8eU|7x5)BASjBoI3}2hwWJPi{}FWe#B8aHxaJ}{I|R;`KpCUZd5#@0b8p7Rdozo)'
        'onfL<)PU&TRYwIPYNhzfIib0VDdBQ|a<K!4-2IjI_*Hm$zKiIwh@#l8+C?@J?g*YvqTNMKIhuFzE=6Dpt+4n;>{A;(m|p1Rg>$e@pSB<Zs<)v2'
        'v;{#qu^>@pFjo3N@t1avEJu_2FbM(fx{Yq?P?4A%7A)>=7Iy)fJW5P(g<wIgBocEJ2c1px?)q(7Bg_*wkse>w9mwpZ-Mvker)z_I?+wA>O2{fg'
        'NhBIu9STdq!k{~#V!gvww>XomIiyv)?t1Kq&JS$75OZ|?`ZF8DfV+OgF&Tn!JfmAYGvu#3kJl6@mW;w6zdT5@7z!4<*v*VQ{S1z4Ts*B2?*h*x'
        'Sd7=G052iHn#k~ZFj+)S)X|K?(8Tz*_&g~`i<rQ4T8^DOC5L7_9w&m3K?o_!;2j8CMSTnvJH3!dJ8xcnk6Ld_*U1gvGuHvLvf>+w$fOu}@f=mm'
        '7LkLmsf4<EzD<r2CW-l1j<WR@RBvAW#(eL^DrGX5_+zNd9kOP+MG1-Gt1!}kV-i976u)2TBZ9w;SKD@|OG=Csd}eX15HIO=N<jp%x6!NEa=8&y'
        '%t3H3P|-`VW81ba>HaoY*;=2PC3ma&EVHhf(02%Axg82V0ZXE%p^>B<nofn5Caz=&QR$WIR?tp=k3R1Q6FlywH#!U3jFNZyE@L%m2NNm}ggV^&'
        'oTT@coR0X|P-H7tn+FSEU}QRafeCMUg&43876hyVt{%)4C;40FnbUft-A8np;BSqm7ByRNF@~bucw=k~_+ggy<|cHxHCRw6ne4=V_~7PCgem(h'
        '-DF&4&`04Vvt|fSE{?e>up<O9rvz=GP`5MnA2Lrsazc6cIR>?AUFBxPJ{r)3QxAvEQ=cyA%|=eY)f2Jzv6yxrPqj6;SBaJE+Ow{wc=_k}HDtki'
        'PTp%jbSSuKK>VY#Gqf04MqOc+r74O*`cQ&O&=#6q>ObKn-}alK0pGXu*1L+cOL$pk+)ahS`MLdwfEvxZW@*2eynKZ4>VF`#DjyR*BHopIK}=O_'
        'ijAWZUGt&V>2tfpd<W_hvnN`Kj2WH%7gkd9xB'
    ),
    'phase4': (
        'c-pO4TW{Mq7JkpK;B<kNv{jU*?d%i|{4m(*4EoTWVA34~Fbsi~Xopi-R7om{rs#j)@9<*Dc9L#)w+L#A<oS4b?mU0_?yNWZY$e+>(cZFdus5oG'
        'b@J@lv&$Q91baPW;=XH?=2q!}-SS4(+)CB53d5E5<681ftBjSEVQH(F6~;1NRl=C8IQdOfsx?;kl{L)6zG54#`cBl$bd9t)gIBg0E?6sWg=TAs'
        'oqTtM(gksJBDbB=0M?t1Ya`<KA4at&Ypu4d<Msw5EG%5&bGJ#YuoA!8e%lR<8`gGls^e{q3H)_+oZE7XyIUe0hedBhn*6xgB-!!4q8n)BU>fVW'
        'G9Mekbz5vzEt=AZMpRB}f%q4a#MMo|UN_>3q;WtctR}gJ-Kkbp$@{7_mC_<-9P;8DQFdBX(f~FroUo_<t+W){?<^Z-OwJm$aii?y<m&C)pV*we'
        'etB|Ii}h5|CG5ncL(aOK`H%Vei)`l3s@I_D>f$GE`OlhfMfy1zk{KjiaKp9cLpo$ROS=8JF8h)N`b7f7Wa(i(RlQB?b<q`?w;xkT;YQW-WG%I^'
        '32;8mJp<sJuS8?!Kl8>252c0eHLP-7I4fHXsY8aaOex!1+=tBt-)z!$o`TOjG4fAAz-ncLz7=(t-U{B*G~`>$%DVY(I<Jhh@>ZA#({Gb3^OD<g'
        '*`}~o6GWtXi_4K!y4eEXU2n4sS}3+3YpGKhwrH)Hf6zUYEAFMSrTXYT{oU4r<8s1yDQUs$5;UKBVbc+~BPV;Au``I<4NEJ?e0WnrT4ls20%hW%'
        'SLB&rg*_i(&w<v~TdqHrK|D$yf_m+IGb|AuAgz^VE$k{MQ<E)Qq)EI;lE+b&L)XA)WGj-KMl|E`lHC7h%h}F5<eJRnhS%Ft)`c`JPhl4AkY)_W'
        'O_u5&u32#yK-@6fb1}^&r1v{-dhu3krPFotnYjHj1FtS@I=~$Y3&6>dT6fy1wd&GJHT@QTwt!`={y^j)nmpMF-pZ{=mM|4M)XWo!K&(YTMX2UL'
        'Brzb4=#5AtIPAQgCy|Go-Ke{H(ulRiI0ShyPgE;Pt4jRr1JX|b^demxu(tzFWYw1zE^O8aI4@t9KfJt2mdnS&6PS7gX&HEwZ7cNCOhTuPXj5V`'
        'W8cl)hqDCIz|l&uiBX$JnoNFEcWf&SqI`7|&bW?JcuGm56!z8ih*UhX=}jjqsd}=7Mdi%CB7484i;?A7`l^aX-3bjlr42L{FR<7{4~NM%oLoy&'
        '-NZ4BA&zg9mVY9`i`MhIqEuyJEXFPN+K`X|g-;iR-FTTTqsT|TXAg+*Qdz_`$s5R=e?-EXvE(^wf^;Oy{Qb#0Hed#62Lfmi0MqNjC(W|9Lnlls'
        '+p6hn0dXrQZ3??clI;eAKt6!ABB*5yADD2m8;=;(e!ajdB_BUPf}+FHh@2H`*)#|bNm;&scXjpdpTCqPX5+Tqx&Aa17y<`8q{51|${w)crFgvv'
        'WS))qQSh#7kSXf|8VHl4!#i1n5iPogS0Y^?xyp8(pZZ_;4F)Z5N4vc~Fxo=~A%NvB6gEwo1<M?w%|BqhQ4X*(zu9G86DdZwh=~wY-u$@qNm0Jg'
        'FY1ciRgop7<#}=beGa2)YqfPoDdtq8gped|0ec^|V+7n*+}qS0QaHHuIv;Y+QLr)>M!t=Mv*OJ|Ts%Mbh!*u!zoNiIzmW-dPGq8xjj9%bVJK!e'
        'obW8}&t?%G#e}~oLSMLN*t76w10YU#CPv>xzMW$4z?K>N>(Oe!UuMyfe>5nGsQ!!2jA9Vp%oJ+RhIeDweH7_|Q=+>5563)xo&WZrRo~VP6-55&'
        'fOyO6!>Z-gShc(=U43r~zdeA!$;b&j)*=Z$mn<*H`}bDHV32sdA>$ZC9_*nT5m3fX-S~PxvOywY??aLIeSk4|Db&pIxGOPY5Dw(Q)UIz`kl528'
        '#YD5jxg9E>F`kAnW+q|ucHzSp{r-<Zv|2f!$4|u$f13OK#?!FNV|HaTQA8s*H+HkyCd(oewJ2`Yil^7N*<zNTF9S`rue_DQ`}c^ZT1{x~sJMd_'
        '*)-CvP=M5BJh8kJG>U~lhwXQN19Q*Q^V7QwqsoV2io?UcdO~K{iAgmz87>~i&TLJ3Y`AC9Y>_zduq<e^OCzY!ErqUZubLVLS1|z?MGl3tWVuMj'
        'b^v{$+95kqWZzV3As2uQv40D!uMzLwQOWYK6gza+Nk0J=J#9&-9&aaB%k`R?2=s~$?S^1boMigQiy5ruUD(!>a`ztElT`?W5;&z-Q{l-hQjX`a'
        'yW9SioOYdH`c79=!E&pb`MEg#!Ih$iD1DVbjRKB<GJ<?fP?z2oNh7RQ?mQ@4tGchmt`RelUuZR4u&WQhUdAYhH#RxgQ4?S%5BengtIRFs+_MS)'
        'u&qY&m4wCzM^<1{l26~@-qiQo29qLs<_nHJTOXH;TMBgMq1VK&_<jf>KI)zz$4x4Q%6Nv020wK!?Jtwex-S`BI=*HPFW!x!-#ufG@4c3roZrv8'
        '#?qyzab!}=xr!tGd;WcJL4IG*TDo|FP?G!qlhf1io}DL`La!l^sgja?N+<hW3nO=*vC)wyYJ~cIw1)Kj1rwio-kdtWa(d22TuMauf$F{B?bsn<'
        '@BZ<x(^wy}snqfz5d0d>-row=@f5MrHh9*(eYov7v8a)kMr@#<G`k>)7M7xD+3*2MiF1_w@cWd379Bkx>Q2tj-@M2!NGaYiK^LE9TRh^1`R6IZ'
        'B<js5;)P(D<&%&$+Y5yU_Le|!z!Pl+YxDU5TdBW9dxP7D1;hw9Me!s~bw_}%yBF6FTYcVhk7#25FOtP?+KwPqHL^>)9AW28ett?n*)awJ46g8)'
        '=B7<z)iy)HCdPKh4eJdIf)c#ox*3?Us_Uq7LB-9S2eMu7^a8<buLB+G8fjsF1^Yn#XBempdA<*g>Ew&!*V}+gey__3Q!xYHX=#N|fi8j8$g3s|'
        'x_3U@+WE`inw=)q{-^trD_4YBHA`}Ek)50%nUp2<iOO<5N1Z|~Llh}Kvb!#~p2vQ7@uS}0_1D@<H$SO}u!Cx%H1yB#4EAlmICzDDZ#E{`%%c{('
        'u1g*OQ;H_f@s(?Myo4i9I*zW38Hn5ukr^rv{ta-661M'
    ),
    'phase4b': (
        'c-oa(YmeJD7X3cIg4?1?sg)JmN!qTvS|Cjy8=z@}q%E>A41tztTNznWMQUc;VE+4_%ZDUCCTWAflK42hyzfiSr~i1=srJ#js2<7chBXJhtE<OX'
        'pM3JkYpJAYv)!bw$^&a<B}(?<`ERUgWnN^usH+8Qgf3*ICwZaX)kURQE^lO6H+wWPQROTXWwCArjgGH=Q(`L@eht>_gpyCzscuD4rGBK0n**Cn'
        'Oy`WfdiLvo{ny-o)tzoSZN7-mt76~OE!JzdjcAqh-|tjiU2WQWPh{?Z&RoYUeD;ItHG2G4)$OsmP^@ZvQ-f`w0souax9x>)%39OJ;Wh4*4EVF{'
        'mWSu#uqk>oFr~Issp}F9uHt>2%Q986l!mLG${*XhYu@a-&8C!ZnzA^ZC3Rb5%8zdAEEQdrs;q8h#Ketk!R!>m$_fQk-?45B+1ih^td)vbS#QnT'
        'F??B6c&3Inzs67XwotmrUdwGO6`>x_DO9cNs>uBK7w-FXd|7OFdb{5DxIcdT@_8vRVguqL2C(vwDqS~CQEe@S@<x=M(IbnI3B9gsr9H-Tp}(c?'
        '0q5%K&G+Ab#}YQ5&cB>ZXOFM0a=C##TU`)NP*l)k2@Y(D(|B%LVI&YSC1oBit{DC)V!f6vfX*ZmfUWdY$t+@8DxIq0PnirjEtIF;V=K_>#DE%1'
        '0XYD#l6N}r__4BxP5h^9YZa8Tr!WjXjfs84MOkM{*SiWWXe)KsLVIlMQmmr>eZ8`KcTAMQA!3c&Uqgg6aaI-$EiRO3TX6`Qh~@gwU^8VcGzgf<'
        '$=CR06Z3YoB)Nm2N?`wn>I2c+)Cyd_k;*g!Sao~X(-Il7${aRz?xnEv4D9%y=%Cqd;E|4F4-LVIu4(;%2ejS|J;OG(;HIA1g@^an;fDE7+Z|fA'
        'X6qCp-^;-KMx>j{aTj{ul^QHIU2#@7hrse~Q@1RuVOy0Mzv!v6Q<mskVczHqvLT5CT*2}yxS4oUlw~CXo~AEfym|BD=O5D)t!@}b9ZAU|S&H}Y'
        'xB>j7C)0V<#JEP&5!<G4)E)1y^P&x4K=AG=`L*q^S9u4Qkk;4c(~e%lR|Bfa<5r41#rQuuse=L9`u31LV%)x_CG2D`+UwK>m%7C$JC9q?Qmoa-'
        'LFWVLJk^q1gD8Ox0Z9Qa$P2KtRDYXVnskCh@~n*bwOE&u<hl$v@q$P63tb90?Se=2>puZx8t@EPgAx=Q3lr}|zE6uBz7>LoY%Ayw!&1k+c?ahK'
        'HIlg>vpCXrL|Dk=cUZstzHRF^*zos6?EQl63;4@wOPD<hbGXO6r$?N(brV1vyM3jSC0E`09l{okjd(9bUF;=?|B<woN_erYU{?<9Q}!(>NG|vF'
        'wiV5ef*4c^R!~u}5aoi&ve*|D5dwAS5Py+PxHXya{;x<7Y#v+fgw>aV{Fs8^43%zh7{mEUqy)wt2PB5R3y$nP_xrs6@IDnH3|Q+Ha{e|KT41yR'
        '-OF|>&C(I8tE9(`*si`!cqum;?{IrXPLY7ul}vS=;=zx==rMpnCzdBH?tyucb(A?(Y6HJDf0BMR-|^Mz+{xL0AnF06m7}q!D%pORNGK)wcOo)m'
        '|47V-QEgJ5vmU3^Bgb~Cg#2}V3%PcXF0>HIQnqs|X;_{le{^x1-rm+lp34fO9%Rz7Mn)EPW#+&2>^_ook80^qoob4#s5{b&WoDf=nf6;^1{s<*'
        '4m>XFTiL=gX#h1{5X16lFHVBcPeU0;2A%EvJA9RY-_>pLCjyzQu%Ao9wNAvO7@aZ`x>k|g0FTd2(ZWB|`zl;{DGuBEkaS)l&K2a!;Qq6EFCkTN'
        'Z8CVyzF-J1!GPmz;^3INbe#6F2-~y+Z2sa|KS<4VQXC}vLqrMQ#x@~NO%|enkd<97p~bbCgRQwR7;$o0+qykqi@kXdG4?#dzFlS%TIumOW9qP!'
        'T>$lhL>7KYIWkt&+OO?YZt2F`)%`~-nx;g?mB)~pYaD#1kqanjWmAew21}$%MYV}0_PL7itc257QI@Go`fOS!pl;+$ym5LL{OYuDa~9`QSG8cK'
        'EMrffe&!PL2_@ugM&D2h)ONG!XVJL8vkgO{PT#T1Kq%obxjbdb6mI)0d_?k)iN~{P#3+3aPv?)<8^er-j6fd6|8hWI%+8?fF8&J0*B)f<+!L|J'
        'vE}0cPanl!d4R{^P*jkJ-gAS+7Z#Ear4L`2_lO%Z`NBL%*ZckR#pnZ5d&UC|T)yv30Odv@NEPj}ae>T8jZ*}u>%r9&Vii|iRUokmj3=LeJdqH{'
        'BFq^9B1P{jGOELzm3TVVpkw_ZIvg#6{TuZW9Uo9YKPq{|DrEmWZ(grXPzG9`DJ5Itil^79zE2w#3>%2p`}Mt}_F78HCEi+SBbSwTE6d|nSh&f`'
        'O!*SB|0FD)&Q~yF<HEwRQycpS(_~x(K4`fiBE4M>-*+0d@SW8Hc4y_!?jUpgYRa%ZHI|*r>($B?7-p&KfzuZ6KzIj-cW`(IgLklahf}=EK;@RI'
        '*W;=kr7(1^KoBRdHJeYTkvSBAEu?g+lf=`<Pa}A@6wOo?&L0oqY)|JuyQ0ifCH75;UKk8mZokqsD3?I^Izqs(ZATxdzT_V$!AB(&sx+}*=OTF;'
        '9o67ugi)2gtJf+q*}dLT+OHxg!<tIgLo#um0=q;o&;f~oSV#7iLi3Vk;Dlm%8Ln)XJ&;NGSrK{?w_OOKOxIT>328)xhH@>kYvk3(Cm_LBPaS<S'
        '`9YLaOB^4G1k-TUYa2v!ei#zK)~us|aGpRoxoq?f+&LA}lc=1Nh&Cyhn0z^qPE~KZ=_(T*68=-Yr51j?_gAl<{q~&qESEC39Pd#qyHk_&T@rxb'
        'px=y<Kwn>zC`@r}1cJA|tg{?E;&U&r(R?d62Vtcb9rHScTd8uDAZ}csh-&9`jmpsAjdNRDV+W)(u7-sI)NA2JE*H^kMGK$wo*W$*%IsjfV6#rk'
        '2tSl!n1!t$a8ZO#D~xk__~gTy`R(Tf^hGe6+=ld;g^@kaxDCS-zeyH<OBRppp)4xZP|`7tX3>1+638@Sw~~(Vu<~Bq1(uBHS{@Q>bc`JPf_|1$'
        'zmJUek&X5NZTDShvw)w|Fyxj*r)dbOCG$B;DYV9>5%j2Dt>|oKQs=q)h7P)9gN>43A7=;<zs*1})COGIRlN=TiC2?FGdcM(noa0IMLyPkpwqER'
        '72ADXP~vcY9vVDPX7X_auB2pe$!rQ!J99usI7Y)QdC2k9P9V*k%%T|-v}aL&u$o*rQw<I&{AV(a@Dz7&g9yQCy-C+rkvVOJALE!cvAbafM+}|9'
        'QyuL{JJ!`pL|R=v0(ThwS`Gy3`Q-DB$?SVULE(P+`KSM}&YVMx#b=5H_Uo=tC=>TcvjAEFnbi$Kjp6+_4s)ylqZHJHEt=6OnL}qX$xtn!>?Acu'
        'N{zxEi^k5~APuF`*Cu9gb^b~M=@+=g9b5Qf#4VI@ZlioL=77z<Yz-qxSs<feSYd1}*Nfp)kr2K54Wki?BXj<0H1TV%iH#<s(j%kN()WyPN;}K<'
        '7Nwm<Q3r|if>r1v#eX!m34O^X%-J_-$wk)bbJm@xmM7(sa$Tx=FED-3GacT$RNveT5J5%__~QNb;yo9+Ir2gz#AJ75eG`Tecb8eNd7NL~jFRH;'
        'p(7ScSby$qQ4T!-oqvbnz0IhioAWgI{uG$|;Kmn3BPE7mU+hx)Kn1{Q)Wftj31f)&*f#e!E^fyNQpbZ*3v6yw>=+7Fr(@13^th=BGpSWEzOYY7'
        '-Kl?iRJHilGqlktmr*Dqv*Fr;%;P2;^KdYdhxs`8g9l)>JV71w`C(K#qtu?oOa2<jwQA_7@kX+|5Zel>pe9HeiV7Yet9Q`GOJ_Lzmo-Y18;qgr'
        'RewOl@*0ka&Jg4u9Xu+m&zz333R%@fm{+a5p|coNVidQn<9Yew<%{Q&7r)!W+f`?Xn3|j_{tLP>f{5vM7l{~(C8g;0HZn`27l~TkWjjF0J2%c9'
        '5uT^EmX7CxUmq?t>*~z2BY8N$8;_m&k?WS4-3rSB<GnXyK)$j@9@x8yN8H~gTdR-h@|TD4Z43FOnl9R?CC<BjqXIi7DonUY=FW-Pt6(!E7@G{t'
        'Wo#XI!>6qW`GTzSFBqk<OJgIw$($L>iHp1xSuVkT*vRgqQwEtUT@~Biy2PN#<L@AmA51Fxetep`FfrH`z0jc5=*%uASxI%#Kl>xtv+4fE!p-GQ'
        'Bl!YG_yNGrKYxGy1-H>+UBGP~_~Pax7JDDkfttV2dFWvLggYZ&G5n#!aMxpW=5AaPY>I7!w5usKv4$@BRe&~g5gdFKed&}-+{M%vET<IG>@S~8'
        '+}#f<C)sYsNf8P0<YKA0yw9Ns$&(W~HVD2EKUn74!$mqCxF2%<e96}ZjEQFck##yiL&H}woX~XuXSu8M#GWDZFq%!>P1Sh+=Ee{HCdp$HGS`Q|'
        'K!&l{ZUe_d)PVtTBTgqJK&qX`L+hb$9q>G}2siE_BJ=5|LUnrX5qo+z)8Wva<LXNQvmStr)h9Z*F=OvL>Af%IF2~+Y7rK-Qf8o95$W^tnEF3QC'
        'hpF}06MM5kM2Ebmu9RDNfgt<_IudKL{InDYXrXITsO0dx;MV#j;)YJW^QK7V)2XvxXkUg5Qj+Sqrk29^N1dJ>#@`6yfUXxj@)y5X#fGIRu{KRv'
        'k}#g8bhw$O-2LGI{jE)#_`kY|pS9bLBF8J!?oPk#pjhO2DqMfS{a^R+s0F7Ah-Y+r{Gl21zca(`<L}IHM-e<SfA|2tL<ruZD|Pb3&NJ8MZnAmM'
        'beLdx^?w~$JSh'
    ),
    'phase5': (
        'c-oa&U2ohr7Jc`xVA=;t8(E3n1TDtUL)xU-0Gn)3w?%-#K%gYfjABVtMauSU;Qzko@<Y-%PLl?KM?;a9m-pk`b2;BXdNS(aNl|xCWOvQ_gW0R@'
        'hm-HV`|kZ-Xvv<h*dWcIbT3O&UrToR?-%cxyzLv&2~(?1v#lDK*-JKv8}^@iw>PhUe*5GXVSZLh8zws~+oF*vM4r@buLi@!u<OO3Wx_sc)tzhy'
        ')v{igeNz|APkE28VR|>V{ecP1x;_l`qN^}~|9ut4KB@u&!nanKzEOro=Ero@GU6|GJ05>NFYOQXWQ8`yX{E0kDTXd>RVABT%Z99=x}fFjL5=<8'
        'e%x*wdD%C$IXSs}_3CBz?EK`Ul3NI`yGX>V6NZ0&joypiepjQ(=$Mg6@z{?hzM!$Ry{hUVf`MdbboTo&%0%AQ+T`lWev9quV(h9$X712wqeK;X'
        'VLdtSSrI?s{wcTG>(tHVZhfg5ojD<EjxS@|X}-B&r7(+eVdPE?btxKpcav?gk%_`<xV5YDhJ3y&+Wbb+6W@3Rv3J&2=^#aw<1~Mb(j2LH7BoD|'
        '5lmGphO4|EWL1|&4LbEpyt-Q{%s5E>w-V2T#Pi%pvbJa-y2!)gdou9~X8Cp<Y4&}!5c{2IWm0Wf-LZAV>DWA>zi|@rurp7>6OB4iJaM1##>Iz?'
        'ku9sX>0VTAURP<YI}ydH=nhfLu+!pIH6q-|Jyn>1L=`nM_D|7_^3^b?A=>g!q}&};yB1AdF~9o0O(*+QgX$w64B49NvG|C9rxgjGd}QiY@=YRX'
        '3!U+L*Qr5zk>9ho!jyZ92M$1<{mcv$>*{V7He|Zo%T`>l=@2$pn!1y0Tg#>brcm-Z0q1}V$=ohgYIm8|LCMnkpd{;uV~`c?07*^#aTwHR<)e%T'
        '8*IMC2J=qte3dXFvtnC0>|_*HC90bxCwq01@kVY9PpsXsKI5vBxluVjc<fhCZg+vOzGv}C9<R$WhZQ@$l|z31EdTL*&o>*|^b1QiPKUbd<nSdW'
        'PJF;2i6c5!{3xSG8@dk!6aPi+Eq>MildRl}ZYM(^eaL7%vZ(GVc{>@8ASQ<Wdqij0>_hNJYE%PIi?A!r3md-BS`GxUe$!ul$f%)`18ZyDQhf0T'
        '<b6{{@Rw7_O{qC&vJDynz!m$jE;kwdNrg6t9#un|0Wl2UZR@5%Uh+JD`{wfU&AZomj?pm9L+0VW*oC6?7H*VXDI?k|_L~8Q;2G393A~MHXi71A'
        '4D2yTlwB-acYRuS+vL=JZW4TUq6;bbCnRhS%%c8jsallQ8c;o@A1UhcL0l`K0<5YjJwvAG8gXEngG)C5Hq(Gc>(NtsvJUqPCTIAm583^#-@__3'
        '*?z_NW7sv)2rT%`uaDQO^QW6Q9OqKBJ+xO{b0C&bQ&}X762@q8Eh{Dw;G<<D%oh+#Orj3e92LJ}Mlmx2U0flI-N9(;--7>94W0MM7kk_*2e84E'
        'f{b>rokkB?<kiDU2WK2ZH4>{Xfl}7&Hp|xbajp~~8YCu+#G7S$`{h2vX<qylmEOG6tWx$2HsinoOQ>gtv4p1(Due}IxdG9E2n0S=s!hRr#Mqb|'
        'qY;sz;{#wcE%yq~L8*iQC2NT&rP{hWS*jc-KI*eyMWa3War3ed;63Hq&)j&Vxxijq<t(*$o%};^UE)?*WNydh;61=73rb@9_a8zAW_docD<<X@'
        'NR_8#OGZ`@ulD7u&zlU(QXfGGO;w1<DT_2p&ak}C%M8oz%_$4_-g!{mWH<kUu*cEa=}nBMG9(|-eF_U{bB;JZNWcz{x`$I%OiQEy;i=2)WoMAE'
        'cmMsJiK?opb3pbI7AY-&MP9m14`@@=;LZmpGG%{gFYT1Nj(S^{R*l3$R_G^=7kVWEX2B%6yv>Mer@s$Uz9OwaFNO$jys@t{H!G!CxjSO8T*w_t'
        'D{TZpb#t9iy+Yp5%Io#1Hyyqq_JtP1AP$%{m(}YOk=JI~4`InFnGEjU)6Viy7E!$Woo(z}n0=Vg{n5d~(&nIbjTON@1z~p+5Bp`N?##vE%#TVz'
        'Q43_WMr>rO*-dTsY--x9=ojzb=nF=^MB5i$EyDr=5EKC8_ewR^IeP4=$DtkO-JNNjV=S)=F+BBpSni@34;g(!bS9WJ5%<g%N|%dA_NKVr$!TKl'
        'BwB}&u!IsU<CidvwIYiB<e%~@_J4f#RSz87%e$hy=L$LpOn@`k2ncaQ?IA$tDstvYh^WVNBT#LrG`$3f{W{=H^eMCwd_~$Nyp^KMMWJ&WzAh*E'
        'icC{zFOz7d0IZ^U%$~+~aY|)2MEb)55}`}<bX4ybaO)HzdQ$4^2tDeP&{?>}v(-~f;E63z_XQZ$XVk!;+D7M3<FqDlUQPC}(NXUbahsw0PJz|b'
        'jOz$vwm?K@Pm`bG_(I&Kg&6h`#HHRO2|Yh_oOyGs0&B@{Ezt%W!DNQX;4ellFpfnI#XQ7~9ilttc2inzYSYMw4*r-=Y-&C{kC!J&4M8~C#sdt7'
        'vP45fSMYgk!iDvEREwnQ>+I~Or||{Zp=^|vWQ1AvAhB-;hj{2l3;~?xiV<Jfu@HC3Tr&bTymKT|Ohk+oQtBHqv@BX?ZV6+62B0%5q=Q#W2)3da'
        '9Mkqp%9GZkW{jfLHDaCzQ3`wTt8OtB?^E9|#Pc#Q%1yNRt;ZJbIOG%#kwsCg-y|+y*I|C5so}99m$c$GuWAj*KV*nB2Q9Fra~}_PR6c8+czFJv'
        '&)tfp3nxKKH%@lXgAI?om=uM&&Fc;wFNnOA$CsYvqG=|-{3bimoCcp%vPUXXpTKM?SR+jsE=n_E?IAE>+hhZiYyhVdLopIgrJD&}g6A6Q!zDNm'
        '=mh@u$_TK{vBk6Kh}!N%&llGbj;S4Y+1}#h+aSEO;7|7&Xp{|iLzr(d#(j-9%R3GV2`?MWNp;X`lyma9GiH9q`YA}v$7~TWenR|h!0}`9+~I~s'
        '2ts8K^=mI>lh~m8h!DzvF`@zE&(ddK!j7<az=VT`l;f!K(exR1_|<@q*}MM~Hs+Q2C-T7E*I@$z14^C|F-E&X>NrEvU5C3<TryiXd*-0|f@*Oz'
        's(Et}rUYpDY)Uyz0cFesX}dQKuNE6peF<%t%`Dq`J<pxH9_+e4uQcD&%`vrzGk2u;8tP)4Z>yz1Lt_&q0e&KT=mHxRp`~a1A@H-AWiL*=Tz9)E'
        'sc8ln?ubw+usF3lRaK>zM0t^YVwZsN`m4b`1W<lJYZFnW9(I7L2v&$=do8GL?!QJ?<X$h5k_0>D3EsWz3H=exBx}O_B-DYW2%CntncIX|K{OKw'
        'y@tl<zQJ&xuQvB8v_-%#@DeM{k}{%C>lz@<kH|?7p5PVO$T6LOsqXSm_FQ-QqKo(fqd-##wX;v$M+d|&SsZs4_R}5DYNB;>p$zxctD9sx-*L2B'
        'PO%KxmAn7OBuA~_Hm(XShih5o@)}{6Q=JU+Y`#6vUQUbWec^LCAjFM5G+*(zpEp3E53~27-4RK#0NjqCq6Dpl^g-&bKBVh@n7yvY(Bed?6I|0z'
        'DO#1a-CZCpEJCA25KYP#YGN3t<G%qz#8B=}?WN4|4@AojFlFx)ZD$-`_O~~-K_6Oo(SXv}jHDIyf&|<tzq9)tVSsaYN<SB&alRR9BXc8fO++JU'
        'HMYHu+#DM!oypEU)9(jLr@xza;im9QdzWMsPq1@*a)LUQ=S0!+Jj>9!(~mjm+g%p*S@;nveKG7%5<Byrz731pfh$*d-R}ofZtYJK;o(_<C-=h-'
        'N-jh4R8=m#tcahU^2b-nApaU`x~H0d`1yRI3wrLF9ULX`S1{T0r=I=T-=X@{>Pt&Cg74)2mqCJ^'
    ),
    'phase6': (
        'c-oa%ZFAeW5&o`Ufy>Pe$yB0cUvk%TtDDK>WtuO!Oq2GDs^LH+Bq1U}9059JQ~&oq3xY3_CEM|MYzf#mEOwvW#g9K+)TX*vOLZaC4bzR?mFnlo'
        'vuDpf?zj=`muprDTa`wO)XE#d-oO6W%vM~Pji^}4wXG|`YL(&JuF<942`f{c(-u#qZQ{x2oiq%;AO85vI7@S>90AYsGUZm5ij@_!mEg7z%Ffn8'
        'iH)?ZC^M1AWXVJpx~weY)mHP$i0=CrQ>w|PDhsB$-Q{x4{KiLo?TA!e=!S8_l<t-^R~Z)YtFvyc;8y3Qg^<bcF|LgWgIC*a5Z$MXbwi(=nA}b#'
        'uU~z9_4@ti_isP31zXPJ`O7Jr$Cq>W*L}WnpMRdOCX-BTSh_1qBa*bt!)!B!oBynZu_BwocPf(^x57-Bt}6|)=z7BNBR2zGyK=nB6l{`;Wt$k1'
        '>q41<j(#dGajpO6c`e>nRau3b;G5TR$65~dt(W#gb>{?bN?dWwUfpn+^K~w+*|%I88xC5cJ2<IVA`lj8Fv@Mfb?HtR@=gkYh#s<YI4d0^Jv4pQ'
        '9Xb~(^!yR~VZknkL!CI#Yp%%XTJd5nw{=;YykRm!bjeM_j*lC&*MkkZ5(acQpI3MH$f;vQn=wq>>Rck;*=*L!pMfa$Uhy9BLYG|(6kMv%-!0d0'
        'O&8<wt7=<QK76EaRhS8rR#H>CERrNE(<F&JSIo04;hsDUW;0Eh`(homF(oUJEk0MZn6jP7^&)tKmDHA1gYzp31EktA&y$E7IMQWh0#{@*rH>|t'
        'O}I)MoYh&@R@e4e<yrDdnmxG(3~&RlbG!IZD)Er`v@FE;umGd7vY1su16d5XY@@|ODm(CB&gb*9{6>iEB-e{8&r@wFn8E69jF?RrbrDP7;?{=5'
        '+ra+*VBFrbnH9CMY%SRDi%ai$_yH;j-$JLLb$h=T7yC?BA?(2+jK6hpi<nKy{pd!d+Q?iSNb&&?u?rSFF&LbvYy&Ymi-F%+;&;MUeydAYk{sG6'
        '5$j-yF0xOg*p@D}BA+gZ4?;%BK7z^CuYP;VmO-shP2{>l`oe&!T=TWeQFtIesQA}JC?g8!J>J8Ty2mgU)(JQwIVW%QH8~KoA~`U4xIzl$2TnJl'
        'pac4^q(i24(iPw2>P7PF)$VwizQfOg@+4@mTnlGGC#4Ti_+Av+HdKAH&2l;TbfkZ^iltFJjAE{u?(ah}{MLyCmLh^Ii7uou1TbFc1*u^aSHcXo'
        '1UsG;iOhl){u$DM?DMN}Ruj0AGoNC@n~s`!{%b;KKpu)C0E1QKSEkXfV3TYsmA6B;fo#hJJ%gxZiW}*@0v8GrZx(^m7(|{>H(};5Vw0ELXJT_o'
        '(%7rtBMk1LVx}~1Vgs<YBGm9go`(_nHw-m!WgfBrF?{-+T}DrE(Z@EM;Sf6C05q+9HiWXIvaHa~OfOf#eS1*6nkntF%ayn3A+>+BE*SKo?c}IK'
        '^NSAO$a7814LJcQ;$5K729-uq|4E2T%|H`P8R1G3o4V*w1E!7W7Hs~IF(1mFxC>5q)&caoFz!k?qIf~)Zc#e+^&r7|3!zjoTGAicC+n~dq<cjF'
        '-<b0xdW$l@f!EiKi^8&^wmBqsVx?T8(9^EBCW>3Ua70cqX+aYj&h(T)Y_<rDQ_~q(es$ed+cg=Hw=v(ZP|vlgxI=L|>Cmwbe%gb);J4vr>s}~D'
        'D8gNeJ@0v=Xz28SU+9;><jh=N?^pfSPFga@-O{q9Ta9n4y%tv>Jt?#~ND5CBPZW!;6c6llqAzq5&~C*kw&<|?HN4QS@Zr*8*d$XCBD#$HuUPD<'
        'dC@sf#o`^$jfjr==HAlfmsNkRR^a_};VNRTHA+nQ?Lbb@4Ms`l&fvQD>~yr<!;=BQetIaxmX_y-B*fe<iyKt`8XvlvK-+y{97%EjM`4QU>BqSY'
        '<aEvlDA5k2k~AmWCY{Ni?PC7`BlrLlr?}1<*gC$0@ZQLTE1owPHpxBDd+rN$n!gZS4ZKDcb?&M&oKILAj_x-mTHNMoLA?w;zk1oTBXs|)IR=be'
        'j78DF2eQXy{r2mVWP$9QEO)?j_j2xx7yE~#oxH5GqWUX17U@r!<Z@$^8<QN|DDP6hiFJ_>9$e@6ymyDGPca3Y%5EKQyIPAo4O^^1R$E@7$v=hw'
        'kBfbJK9-TB1uIV3fuwsseH(L4-HXi2bP3~D%YJ=zs@W018iyvJjULmWyULxoma_Akc8VhE^L!N<8p3N2RgU7W8%GUFSKFq%L6v(~@dCjH)Ee(o'
        'SCX2W5C!R?9SVNwBLRI(fp%Y-QpKPNs9t2^b_|^dvOBj&r-XWNXlY#Z)|s^D<Zk2^o$@VN4;xLrrgs<<V%gz%_}A%655+{&9%XsrmRCd96+Oo+'
        '*iLn~F^LOrr#<H*_=Rq+;MHDKi=c$LuK8XB|D-<Gm;Y{TJ|X(yY#WI-Bhij&{D2mcfM`&I=_^mx!hRKKmF}xMgs{~Jde2i5Kpjx>9GACnPg{p}'
        '{r2)F=JB;{G%Ny==Yw@<slr>&^N&++hAjMvm78%VHdML@erP=NyRMmW#BI@|2TIl#KGv5sO!Cs^0#+V8XF9h29TQ#H$w$ls+mj}jMTm~LU(Cgp'
        'kfe{!*vjqBCOK~q<fCQ7Zv_4y=aU1+v_{s(8o`22N6W?4m4BL`7*BH`8U=Wlo$f7i!M+8itgBQcG;Sl&jjg)b+Yg;dJ6{ui_48dY8b#W@@x*xq'
        'K*8t{;9kG+J)K4iUJxc!D#m0N9%M&d8@5lifXDLCIFqEHdtv5BZ`D5iM$s6v_m*WPdU0yH-4io-eAKL|_fFGVbCqJO>0hcvF1NCs>|KI5motTK'
        'A89}^h-iTpH%d+EEm~4c^)p5LIL>pVf#S}t{;_*X6I|}b6^addVWkmu)+-Tc1oBy0VqAhXf>#I6=IbxrK}A-l5MoRG>e=4e(j7kz?wsQe5!Z!K'
        'Oi$>TKEGBN`w}Z|ZAdE^xC?E<mZk+@Wfxb`!Gq3!RPYy%7N@{<da{$9p6)lNPiQWukKyq239aJvDb2{r6$a!IoEMI(&(ZlmaX+o;+LG@B_jD3X'
        'CMfbrqEPjcgvQVyN$8z7NrG$t5KfQ2lm7#jt8j?'
    ),
}

STAGES = ['phase1','phase2','phase3','phase4','phase4b','phase5','phase6']
REQUIRED = {
 'phase1':['phase1_landmark_predictors.parquet','phase1_labels.parquet','phase1_cohort_audit.parquet','phase1_report.json'],
 'phase2':['phase2_temporal_lab_features.parquet','phase2_report.json'],
 'phase3':['phase3_models.joblib','phase3_report.json','phase3_test_predictions.parquet'],
 'phase4':['phase4_diagnostics.json','validation_reliability.png'],
 'phase4b':['phase4b_models.joblib','phase4b_report.json','phase4b_evaluation_predictions.parquet'],
 'phase5':['phase5_report.json'],
 'phase6':['phase6_report.json','phase6_capacity_comparison.csv'],
}

def execute(stage, argv):
    encoded=''.join(MODULES[stage])
    source=zlib.decompress(base64.b85decode(encoded)).decode('utf-8')
    old_argv=sys.argv[:]
    try:
        sys.argv=[stage+'.py']+argv
        scope={'__name__':'__main__','__file__':stage+'.py'}
        exec(compile(source,stage+'.py','exec'),scope)
    finally:
        sys.argv=old_argv

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base',required=True,help='Root of MIMIC-IV v3.1 files')
    p.add_argument('--output',required=True,help='Folder for phase1_outputs through phase6_outputs')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=500000)
    p.add_argument('--sample',type=int,default=500,help='Phase 5 SHAP evaluation sample size')
    p.add_argument('--resume',action='store_true',help='Skip a stage when its expected output files exist')
    p.add_argument('--from-phase',choices=STAGES,default='phase1')
    p.add_argument('--through-phase',choices=STAGES,default='phase6')
    args=p.parse_args()
    if STAGES.index(args.from_phase)>STAGES.index(args.through_phase): p.error('from-phase must precede through-phase')
    base=Path(args.base).expanduser().resolve()
    root=Path(args.output).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    if not base.is_dir(): p.error('MIMIC root does not exist: '+str(base))
    phase={name:root/(name+'_outputs') for name in STAGES}
    common={'phase1':str(phase['phase1']),'phase2':str(phase['phase2']),
            'phase3':str(phase['phase3']),'phase4b':str(phase['phase4b'])}
    def flags(d):
        return [v for key,value in d.items() for v in ('--'+key.replace('_','-'),str(value))]
    args_by_stage={
      'phase1':{'base':base,'output':phase['phase1']},
      'phase2':{'base':base,'phase1':common['phase1'],'output':phase['phase2'],'chunk_size':args.chunk_size},
      'phase3':{'phase1':common['phase1'],'phase2':common['phase2'],'output':phase['phase3'],'threads':args.threads},
      'phase4':{'phase1':common['phase1'],'phase2':common['phase2'],'phase3':common['phase3'],'output':phase['phase4']},
      'phase4b':{'phase1':common['phase1'],'phase2':common['phase2'],'output':phase['phase4b'],'threads':args.threads},
      'phase5':{'phase1':common['phase1'],'phase2':common['phase2'],'phase4b':common['phase4b'],'output':phase['phase5'],'sample':args.sample},
      'phase6':{'phase4b':common['phase4b'],'output':phase['phase6']},
    }
    chosen=STAGES[STAGES.index(args.from_phase):STAGES.index(args.through_phase)+1]
    for stage in chosen:
        if args.resume and all((phase[stage]/f).exists() for f in REQUIRED[stage]):
            print('SKIP',stage,'(existing outputs)',flush=True)
            continue
        print('\n'+'='*70+'\nRUNNING '+stage.upper()+'\n'+'='*70,flush=True)
        try: execute(stage,flags(args_by_stage[stage]))
        except Exception:
            print('FAILED:',stage,'; earlier phase outputs remain available.',file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        missing=[f for f in REQUIRED[stage] if not (phase[stage]/f).exists()]
        if missing: raise RuntimeError(stage+' finished without required outputs: '+repr(missing))
    print('\nCompleted stages:',', '.join(chosen),'\nResults:',root,flush=True)
    print('NOTE: The ICU allocation simulation is retrospective top-k ranking, not a validated clinical allocation policy.')

if __name__=='__main__': main()
