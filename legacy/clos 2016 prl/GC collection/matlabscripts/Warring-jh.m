(* ::Package:: *)

(* ::Section:: *)
(*prolog*)


BeginPackage["Warring`"]


(* ::Section:: *)
(*declarations*)


warring::usage="Package by U.W. June 2010-- no garantie for anything\n
List of functions implemented:
	- GaussProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]
	- LorentzProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]
	- ThermalBSB[t_,nb_,\[Delta]_,\[CapitalOmega]_,maxn_]
	- ThermalRSB[t_,nb_,\[Delta]_,\[CapitalOmega]_,maxn_]
	- BinIt[Ddata_,NO_]
	- LoadPaulaData[Dpath_,Dfile_,DropLast_,Prt_]
	- FLoadDataQC[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,Dy_,DropLast_]
	- FLoadDataQCHist[Dpath_,Dprefix_,Ddate_,Dfile_,DataRange_,HistStart_]
	- FitPeak[Fdata_,Ffitfun_,Ffitpara_]
	- v[Ekin_,Mass_]
	- WavenumberInEnergy[k_]
	- WavenumberInWavelength[k_]
	- \[Omega][f_]
	- dBmtomW[dBm_]
	- mWtodBm[mW_]
	- rotvec[\[Theta]_,t_]
	- AnalysisBenchmark[{filename_,seqLength_}]
	- GiveBinnedData[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,loadrows_,drop_,Nbins_]
	- SidebandFit[{dataRSB_,dataBSB_},MWTime_,{\[Pi]tgroS_,nbarS_,centerS_,IntensS_,offsetS_}]
	- SidebandFitNEW[{dataRSB_,dataBSB_},MWTime_,{\[Pi]tgroS_,nbarS_,centerS_,IntensS_,offsetS_}]
	- SidebandFlopFit[{dataRSB_,dataBSB_},det_,{\[Pi]tgroS_,nbarS_,IntensS_,offsetS_}]\n
	- pdc[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]b_,\[CapitalGamma]dec_,{\[Omega]_,t_}], pdb[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]b_,\[CapitalGamma]dec_,{\[Omega]_,t_}] und pdr[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]r_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:
Furthermore it sets some options for Plots and physics constants.. "


GaussProfil::usage=
"so you can use a Gaussian profil\n 
GaussProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]:
Int_: sets the intensity (nomalized for Int = 1)
\[Nu]center_: sets the peak center position
\[Sigma]_: sets the peak width"


LorentzProfil::usage=
"so you can use a Lorentzian profil\n 
LorentzProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]:
Int_: sets the intensity (nomalized for Int = 1)
\[Nu]center_: sets the peak center position
\[Sigma]_: sets the peak width"


FLoadDataQC::usage=
"Loads data taken with the QCDAC!\n
FLoadDataQC[Dpath_,Dprefix_,Ddate_,Dfile_,Dy_,DropLast_]:
	Dpath_: sets path of saved data (String-- Example: \\\847anyon\\qc\\MagTrap\\out\\)
	Dprefix_: name of used prog (String-- Example: sb_test%FieldIndep_sb_red_both_freq_drift --)
	Ddate_: relevant date (String-- Example: 20100527)
	Dfile_: relevant file (String-- Example: 213419.458)
	Dx_: # row in data representing the x values
	Dy_: # row in data representing the y values
	DropLast_: # of points to drop at the end, because of lost ion"


FitPeak::usage=
"Fits data...\n
FitPeak[Fdata_,Ffitfun_,Ffitpara_,x_]:
	Fdata_: list of {{x,y},..} values
	Ffitfun_: is the appropriate fit function (Example: a+b*x)
	Ffitpara_: sets the fitparameter with start values (Example: {{a,0},{b,1}})
	x_: is the variable"


AnalyzeTrapWM::usage=
"AnalyzeTrapWM[pot_,Mass_,x_,y_,z_,ions_]:
	pot_: total trapping potential
	x_,y_,z_: coordinate system
	ions_: eneter mass and position for each ion (Example: {{Mass,Pos},{Mass,Pos},..})
	NOTE: THIS TREATS DIFFERENT MASSES!
"


AnalyzeTrap::usage=
"AnalyzeTrap[pot_,x_,y_,z_,ions_]:
	pot_: total trapping potential
	x_,y_,z_: coordinate system
	ions_: eneter mass and position for each ion (Example: {{Mass,Pos},{Mass,Pos},..})
"


AnalysisBenchmark::usage=
"Benchmark analysis...
"


(* ::Section:: *)
(*start package code*)


Off[Plot::"plnr"];
Off[General::"spell"];
Off[General::"spell1"];
Off[Partition::pdep];
Needs["GUIKit`"];
Needs["ErrorBarPlots`"];


(* ::Subsection:: *)
(*set some operating system parameters*)


If[$OperatingSystem=="MacOSX",
{QCdatapath="/Users/uw/Documents/NIST_data/";BERdatapath="/Users/uw/Documents/Uni Freiburg/Daten/Bermuda Daten/";PAULAdatapath="/Users/uw/Documents/Uni Freiburg/Daten/Paula Daten/";fillpath="/"},
{QCdatapath="\\\847anyon\\qc\\MagTrap\\out\\";fillpath="\\"}]


(* ::Subsection:: *)
(*physical constants*)


\[Alpha]FS = 7.2973525698*10^(-3);
kB = 1.3806488*10^(-23);
g = 9.80665;
G = 6.67384*10^(-11);
h = 6.62606957*10^(-34);
hbar=h/(2\[Pi]);
e = 1.602176565*10^(-19);
c = 299792458;
mN = 1.674927351*10^(-27);
mP = 1.672621777*10^(-27);
mE = 9.10938291*10^(-31);
amu = 1.660538921*10^(-27);
\[Mu]B = 9.2740 * 10^(-24);
\[Epsilon]0 = 8.854187817*10^(-12);
\[Mu]0 = 4 \[Pi] 10^-7;

deg=\[Pi]/180;

ms=10^(-3);
\[Mu]s=10^(-6);
ns=10^(-9);

\[Mu]m=10^-6;
nm=10^-9;
mm=10^-3;
cm=10^-2;
in=0.0254;
ft=0.3048;

mA=10^-3;

kHz=10^3;
MHz=10^6;
GHz=10^9;

nF=10^-9;
pF=10^-12;
nH=10^-9;
\[Mu]H=10^-6;

kOhm=10^3;
MOhm=10^6;


mW=10^-3;
gramm=10^-3;

gauss=10^-4;
mT=10^-3;

Torr=133.3223;


(* ::Subsection:: *)
(*simple and handy functions*)


WavenumberInEnergy[k_]:= k/SetPrecision[8065.54477,9];
WavenumberInWavelength[k_]:= c/(WavenumberInEnergy[k]*e/(h))*10^(9);
dBmtomW[dBm_]:=N[10^(dBm/10)];
mWtodBm[mW_]:=N[10*Log[10,mW]];

\[Omega][f_]:=2*\[Pi]*f;
rotvec[\[Theta]_,t_]:=t {0,Cos[\[Theta]], Sin[\[Theta]]};



GaussProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]:=Int/(2\[Pi] \[Sigma]^2) Exp[-((x-\[Nu]center)^2/(2\[Sigma]^2))];
LorentzProfil[{Int_,\[Nu]center_,\[Sigma]_},x_]:=(2*Int/\[Pi])*(\[Sigma]/(4*(x-\[Nu]center)^2+\[Sigma]^2));


(* ::Subsection:: *)
(*some physics formulas*)


v[Ekin_,Mass_]:= Sqrt[(2*Ekin*e)/Mass];


"Doppler Temperature"
TDoppler[\[CapitalGamma]nat_,s_,\[Xi]_]:=(h \[CapitalGamma]nat)/(8\[Pi] kB) Sqrt[1+s](1+\[Xi]); 


ThermalRSB[t_,nb_,\[Delta]_,\[CapitalOmega]_,maxn_]:=\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(maxn\)]\(If[n > 0, 
\*SuperscriptBox[\(nb\), \(n\)], 1]/
\*SuperscriptBox[\((nb + 1)\), \(n + 1\)] \((
\*SuperscriptBox[\(Cos[t 
\*SqrtBox[\(
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4 \((1 + n)\)\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)]\)]/2]\), \(2\)] + \ 
\*SuperscriptBox[\(\[Delta]\), \(2\)]\ 
\*SuperscriptBox[\(Sin[t\ 
\*SqrtBox[\(
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4 \((1 + n)\)\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)]\)]/2]\), \(2\)]/\((
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4 \((1 + n)\)\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)])\))\)\)\);
ThermalBSB[t_,nb_,\[Delta]_,\[CapitalOmega]_,maxn_]:=1/(nb+1)+\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 1\), \(maxn\)]\(
\*SuperscriptBox[\(nb\), \(n\)]/
\*SuperscriptBox[\((nb + 1)\), \(n + 1\)] \((
\*SuperscriptBox[\(Cos[t 
\*SqrtBox[\(
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4  n\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)]\)]/2]\), \(2\)] + \ 
\*SuperscriptBox[\(\[Delta]\), \(2\)]\ 
\*SuperscriptBox[\(Sin[t 
\*SqrtBox[\(
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4  n\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)]\)]/2]\), \(2\)]/\((
\*SuperscriptBox[\(\[Delta]\), \(2\)] + 4  n\ 
\*SuperscriptBox[\(\[CapitalOmega]\), \(2\)])\))\)\)\);


"LD parameter for 25Mg+"
\[Eta]LD[\[Nu]tr_,\[Alpha]_]:=(Sqrt[2]*Abs[Cos[\[Alpha] Degree]] 2\[Pi])/(279.6 nm) Sqrt[(h/(2\[Pi]))/(2 25 amu (2\[Pi] \[Nu]tr))];


"for CC:";
pdcc[\[CapitalOmega]0_,\[Omega]c_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-\[CapitalOmega]0^2/(\[CapitalOmega]0^2+(\[Omega]-\[Omega]c)^2) Sin[Sqrt[\[CapitalOmega]0^2+(\[Omega]-\[Omega]c)^2]/2 t]^2)*Exp[-\[CapitalGamma]dec t]+.5;


"for OC:";
makeThermalPop[nbar_,maxFock_]:=Table[{1/(1+nbar) (nbar/(1+nbar))^i},{i,0,maxFock}]

makeThermal[nbar_,maxFock_]:=Sqrt[makeThermalPop[nbar,maxFock]]

P[nb_,n_]:=Flatten[makeThermalPop[nb,10(nb+1.)]][[n+1]]


"define different OC rabi rates:";
\[CapitalOmega]car[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*LaguerreL[n,0,\[Eta]^2]

"1st SBs:";
\[CapitalOmega]bsb[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[n!/(n+1)!]\[Eta] LaguerreL[n,1,\[Eta]^2]
\[CapitalOmega]rsb[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[(n-1)!/n!]\[Eta] LaguerreL[n-1,1,\[Eta]^2]

"2nd SBs:";
\[CapitalOmega]bsb2[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[n!/(n+2)!] \[Eta]^2 LaguerreL[n,2,\[Eta]^2]
\[CapitalOmega]rsb2[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[(n-2)!/n!] \[Eta]^2 LaguerreL[n-2,2,\[Eta]^2]

"3rd SBs:";
\[CapitalOmega]bsb3[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[n!/(n+3)!] \[Eta]^3 LaguerreL[n,3,\[Eta]^2]
\[CapitalOmega]rsb3[\[Eta]_,n_,\[CapitalOmega]0_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*Sqrt[(n-3)!/n!] \[Eta]^3 LaguerreL[n-3,3,\[Eta]^2]

"all in one (single mode!)";
\[CapitalOmega]oc[\[Eta]_,n_,\[CapitalOmega]0_,\[CapitalDelta]n_]:=\[CapitalOmega]0*Exp[-(\[Eta]^2/2)]*
Sqrt[
If[(n+\[CapitalDelta]n)>n,n!,(n+\[CapitalDelta]n)!]/
If[(n+\[CapitalDelta]n)>n,(n+\[CapitalDelta]n)!,n!]
]*\[Eta]^Abs[\[CapitalDelta]n]*
LaguerreL[If[(n+\[CapitalDelta]n)>n,n,(n+\[CapitalDelta]n)],Abs[\[CapitalDelta]n],\[Eta]^2];


"all in one multpile modes!:";
\[CapitalOmega]ocMM[\[Eta]_,n_,\[CapitalOmega]0_,\[CapitalDelta]n_]:=\[CapitalOmega]0*\!\(
\*SubsuperscriptBox[\(\[Product]\), \(i = 1\), \(Length[\[Eta]]\)]\(Exp[\(-\((\[Eta][\([i]\)]^2/2)\)\)]*\[IndentingNewLine]Sqrt[\[IndentingNewLine]If[\((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\) > n[\([i]\)], \(n[\([i]\)]!\), \(\((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\)!\)]/\[IndentingNewLine]If[\((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\) > n[\([i]\)], \(\((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\)!\), \(n[\([i]\)]!\)]\[IndentingNewLine]]*
\*SuperscriptBox[\(\[Eta][\([i]\)]\), \(Abs[\[CapitalDelta]n[\([i]\)]]\)]*LaguerreL[If[\((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\) > n[\([i]\)], n[\([i]\)], \((n[\([i]\)] + \[CapitalDelta]n[\([i]\)])\)], Abs[\[CapitalDelta]n[\([i]\)]], \[Eta][\([i]\)]^2]\)\);




(*pdocthermMM[nb_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Omega]oc_,{\[Omega]_,t_}]:=If[
Length[nb]==0,
\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(10\((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)]t]\), \(2\)]\)\),
If[
Length[nb]==2,
\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(k[1] = If[\[CapitalDelta]n[\([1]\)] > 0, 0, Abs[\[CapitalDelta]n[\([1]\)]]]\), \(10\((nb[\([1]\)] + 1)\)\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(k[2] = If[\[CapitalDelta]n[\([2]\)] > 0, 0, Abs[\[CapitalDelta]n[\([2]\)]]]\), \(10\((nb[\([2]\)] + 1)\)\)]
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*P[nb[\([1]\)], k[1]]*P[nb[\([2]\)], k[2]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)]t]\), \(2\)]\)\),
\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(k[1] = If[\[CapitalDelta]n[\([1]\)] > 0, 0, Abs[\[CapitalDelta]n[\([1]\)]]]\), \(10\((nb[\([1]\)] + 1)\)\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(k[2] = If[\[CapitalDelta]n[\([2]\)] > 0, 0, Abs[\[CapitalDelta]n[\([2]\)]]]\), \(10\((nb[\([2]\)] + 1)\)\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(k[3] = If[\[CapitalDelta]n[\([3]\)] > 0, 0, Abs[\[CapitalDelta]n[\([3]\)]]]\), \(10\((nb[\([3]\)] + 1)\)\)]
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2], k[3]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2], k[3]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*P[nb[\([1]\)], k[1]]*P[nb[\([2]\)], k[2]]*P[nb[\([3]\)], k[3]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]ocMM[\[Eta], {k[1], k[2], k[3]}, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)]t]\), \(2\)]\)\)\)]];
*)


"___oc CAR:___";
pdc[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]c_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]car[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]car[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]c)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]car[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]c)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;

pdocthermSM[nb_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Omega]oc_,{\[Omega]_,t_}]:=
\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(10 \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n, \[CapitalOmega]0, \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)


"___1st SBs:___";

pdb[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]b_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]bsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]bsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]bsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;

pdr[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]r_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 1\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]rsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]rsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]rsb[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;



"___2nd SBs:___";

pdb2[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]b_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]bsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]bsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]bsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;

pdr2[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]r_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 2\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]rsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \(
\*SuperscriptBox[\(\[CapitalOmega]rsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)]\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]rsb2[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;


"___3nd SBs:___";

pdb3[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]b_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]bsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]bsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)])\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]bsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]b)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;

pdr3[nb_,\[CapitalOmega]0_,\[Eta]_,\[Omega]r_,\[CapitalGamma]dec_,{\[Omega]_,t_}]:=(.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 3\), \(10\ \((nb + 1)\)\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]rsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)], \(
\*SuperscriptBox[\(\[CapitalOmega]rsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)]\)]*P[nb, n]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]rsb3[\[Eta], n, \[CapitalOmega]0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]r)\), \(2\)]\)], \(2\)] t]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec t]+.5;


"...and all in one!";


"trap-trap coupling:";
\[CapitalOmega]ex[qa_,qb_,dx_,ma_,mb_,\[Nu]x_]:=(qa e qb e)/(4\[Pi] \[Epsilon]0 dx^3 Sqrt[ma amu mb amu]2\[Pi] \[Nu]x);


PlotRabiRates[T\[Eta]_,max_]:=
ListPlot[{
  Table[\[CapitalOmega]oc[T\[Eta], n, 1., 0.], {n, 0., 100., 1.}],
  Table[\[CapitalOmega]oc[T\[Eta], n, 1., 1.], {n, 0., 100., 1.}],
  Table[\[CapitalOmega]oc[T\[Eta], n, 1., -1.], {n, 0., 100., 1.}]
  }
 , PlotRange -> {{0, max}, {-.5, 1.0}}
 , PlotStyle -> {Black, GrayLevel[.4], GrayLevel[.6], GrayLevel[.8]}
 , FrameLabel -> {"# Fock state", "\!\(\*SubscriptBox[\(\[CapitalOmega]\), \(n, n + \
s\)]\)/\!\(\*SubscriptBox[\(\[CapitalOmega]\), \(cc\)]\)"}
, PlotLegends -> {"OC","1.BSB","1.RSB"} 
	 ] 
	


PlotRabiRatesMM[Tc_,Ts_,max_]:=
ListPlot[{
  Table[\[CapitalOmega]ocMM[{Tc,Ts}, {n,n}, 1., {0.,0.}], {n, 0., 100., 1.}],
  Table[\[CapitalOmega]ocMM[{Tc,Ts}, {n,n}, 1., {1.,0.}], {n, 0., 100., 1.}],
  Table[\[CapitalOmega]ocMM[{Tc,Ts}, {n,n}, 1., {0.,1.}], {n, 0., 100., 1.}],
  Table[\[CapitalOmega]ocMM[{Tc,Ts}, {n,n}, 1., {2.,0.}], {n, 0., 100., 1.}]
  }
 , PlotRange -> {{0, max}, {-.5, 1.0}}
 , PlotStyle -> {Black, GrayLevel[.4], GrayLevel[.6], GrayLevel[.8]}
 , FrameLabel -> {"# Fock state", "\!\(\*SubscriptBox[\(\[CapitalOmega]\), \(n, n + \
s\)]\)/\!\(\*SubscriptBox[\(\[CapitalOmega]\), \(cc\)]\)"}
, PlotLegends -> {"OC","1.COM","1.STR","2.COM"} 
	 ] 


(* ::Subsection:: *)
(*Fit fcts.*)


FitOCandBSB[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[P[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],P[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		ffpdocthermSM[Pn1,\[Eta],1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,amp,off,\[Omega]oc,\[Delta]t,{\[Omega],t}],
		amp*(1-(ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]))+off
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];
(*
Print[
Show[
ErrorListPlot[combineDataErrSM[{dataBSB,dataOC},t1]],
Plot[sffpdocthermSM[TPnM1,.32,2\[Pi] T\[CapitalOmega]cc kHz,T\[CapitalGamma]dec,TAmp,Toff,0,{0.,t \[Mu]s},{t1 \[Mu]s}]
,{t,0,70.}
,PlotRange->{All}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
]
]
];
Print[
sffpdocthermSM[TPnM1,.32,2\[Pi] T\[CapitalOmega]cc kHz,T\[CapitalGamma]dec,TAmp,Toff,0,{0.,5 \[Mu]s},{t1 \[Mu]s}]
];
*)
FitPeak[combineDataSM[{dataBSB,dataOC},t1],
sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,If[ToggleFree[[6]]==1,Amp,TAmp],
If[ToggleFree[[7]]==1,off,Toff],If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,0.,{0.,x \[Mu]s},{t1 \[Mu]s}],
fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,1.,0.,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"],
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,1/(1+nbarM1) (nbarM1/(1+nbarM1))^x,{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeThermalPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];


FitOCandBSBvar[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[P[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],P[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		Amp*(1-ffpdocthermSM[Pn1,\[Eta],1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],t}])+off,
		Amp*(1-(ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]))+off
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,1.,0.,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"],
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,1,0,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,1/(1+nbarM1) (nbarM1/(1+nbarM1))^x,{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeThermalPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];



FitOCandBSBvar2[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[P[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],P[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		ffpdocthermSM[Pn1,\[Eta],1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],t}],
		ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"],
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,1/(1+nbarM1) (nbarM1/(1+nbarM1))^x,{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeThermalPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];


FitOC[dataOC_, ExpPara_, StartValues_, ToggleFree_, NFock_] :=
  
  Module[{nbarM1, FPop, Pres, \[Nu]M1, Tnb1, T\[Alpha]1x,T\[CapitalOmega]cc, T\[CapitalGamma]dec, TAmp, Toff, T\[Delta]t,TPnM1, PnM1, \[Alpha]1x, \[CapitalOmega]cc, \[CapitalGamma]dec, Amp, off, \[Delta]t, t1,Tfres, fitpara, ffpdocthermSM, combineDataSM},
   {\[Nu]M1 = Abs[ExpPara[[1]] - ExpPara[[2]]];
      Print[ "mode freq. SubscriptBox[\[Omega],M1]/(2\[Pi])(MHz): ", \[Nu]M1];
      Tnb1 = StartValues[[1]];
    Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
	Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
	T\[CapitalOmega]cc = StartValues[[3]];
	T\[CapitalGamma]dec = StartValues[[4]];
	TAmp = StartValues[[5]];
	Toff = StartValues[[6]];
	T\[Delta]t = StartValues[[7]];
      
      "prepare fock population...";
      TPnM1 = Table[P[Tnb1, n], {n, 0, NFock - 1}];
      PnM1 = Table[PM1[i], {i, 0, NFock - 1}];
      
      "join fit parameters...";
      fitpara = Join[Table[{PM1[i], P[Tnb1, i]}, {i, 0, NFock - 1}], {{\[Alpha]1x, T\[Alpha]1x}, {\[CapitalOmega]cc, T\[CapitalOmega]cc}, {\[CapitalGamma]dec,T\[CapitalGamma]dec}, {Amp, TAmp}, {off, Toff}, {\[Delta]t,T\[Delta]t}}];
      
      
      Print[fitpara];
      
t1=IntegerPart[Max[dataOC[[All,1]]]]+1;
 
      "define fit fct.:";
      ffpdocthermSM[Pn1_, \[Eta]_, \[CapitalOmega]0_, \[CapitalGamma]dec_, Amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}] :=
       
       Amp*(1 - ((0.5 - (\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = 0\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)] + 0.5)) + off;
      
        
      combineDataSM[data_] := Transpose[Drop[Transpose[data[[All]]],{3}]];
      
      FitPeak[combineDataSM[dataOC],
				ffpdocthermSM[If[ToggleFree[[1]] == 1, PnM1, TPnM1],
        					\[Eta]LD[\[Nu]M1 MHz, If[ToggleFree[[2]] == 1, \[Alpha]1x, T\[Alpha]1x]],
        					2 \[Pi] If[ToggleFree[[3]] == 1, \[CapitalOmega]cc, T\[CapitalOmega]cc] kHz, 
        					If[ToggleFree[[4]] == 1, \[CapitalGamma]dec, T\[CapitalGamma]dec] kHz,
        					If[ToggleFree[[5]] == 1, Amp, TAmp],
        					If[ToggleFree[[6]] == 1, off, Toff], 
							0.,
        					If[ToggleFree[[7]] == 1, \[Delta]t, T\[Delta]t] \[Mu]s,	
        					{0., x \[Mu]s}],
       		fitpara, x];
      
      
      Print[Tfres = Ffitres["ParameterConfidenceIntervalTable"]];
	
		Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];      

      Pres = Show[{
         ErrorListPlot[dataOC(*, PlotStyle -> Orange*)],
         Plot[          
          ffpdocthermSM[PnM1, \[Eta]LD[\[Nu]M1 MHz, \[Alpha]1x], 2 \[Pi] \[CapitalOmega]cc kHz, \[CapitalGamma]dec kHz, Amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}]/.Ffitres["BestFitParameters"]
          , {t, 0, t1}
          , PlotRange -> {All, {0, TAmp + Toff}}
          , PlotPoints -> IntegerPart[250]
          , MaxRecursion -> 0
         (* , PlotStyle -> {Thick, Orange}*)
          ]
         }, AspectRatio -> 1/4, ImageSize -> {Automatic, 300.}, 
        FrameLabel -> {"oc pulse duration (\[Mu]s)", "BDx fluo. cts. (200 (\[Mu]s ^-1)"}];
      
      Print["Fock state population of mode 1: "];
      FPop = Table[{i - 1, Tfres[[1, 1, i + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop, 1/(1 + nbarM1) (nbarM1/(1 + nbarM1))^x, {{nbarM1, Tnb1}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[
          makeThermalPop[nbarM1, NFock - 1] /. 
            Ffitres["BestFitParameters"] // Flatten, 
          FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[          
          Table[{Tfres[[1, 1, i + 1, 2]], 
            Tfres[[1, 1, i + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];
      
      Print[Pres];
      }
     Return[];
   
   ];



FitOChell[dataOC_, ExpPara_, StartValues_, ToggleFree_, NFock_] :=
  
  Module[{nbarM1, FPop, Pres, \[Nu]M1, Tnb1, T\[Alpha]1x,T\[CapitalOmega]cc, T\[CapitalGamma]dec, TAmp, Toff, T\[Delta]t,TPnM1, PnM1, \[Alpha]1x, \[CapitalOmega]cc, \[CapitalGamma]dec, Amp, off, \[Delta]t, t1,Tfres, fitpara, ffpdocthermSM, sffpdocthermSM, combineDataSM},
   {\[Nu]M1 = Abs[ExpPara[[1]] - ExpPara[[2]]];
      Print[ "mode freq. SubscriptBox[\[Omega],M1]/(2\[Pi])(MHz): ", \[Nu]M1];
      Tnb1 = StartValues[[1]];
    Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
	Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
	T\[CapitalOmega]cc = StartValues[[3]];
	T\[CapitalGamma]dec = StartValues[[4]];
	TAmp = StartValues[[5]];
	Toff = StartValues[[6]];
	T\[Delta]t = StartValues[[7]];
      
      "prepare fock population...";
      TPnM1 = Table[P[Tnb1, n], {n, 0, NFock - 1}];
      PnM1 = Table[PM1[i], {i, 0, NFock - 1}];
      
      "join fit parameters...";
      fitpara = Join[Table[{PM1[i], P[Tnb1, i]}, {i, 0, NFock - 1}], {{\[Alpha]1x, T\[Alpha]1x}, {\[CapitalOmega]cc, T\[CapitalOmega]cc}, {\[CapitalGamma]dec,T\[CapitalGamma]dec}, {Amp, TAmp}, {off, Toff}, {\[Delta]t,T\[Delta]t}}];
      
      
      Print[fitpara];
      
t1=IntegerPart[Max[dataOC[[All,1]]]]+1;
 
      "define fit fct.:";
      ffpdocthermSM[Pn1_, \[Eta]_, \[CapitalOmega]0_, \[CapitalGamma]dec_, Amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}] :=
       
       Amp*(1 - ((0.5 - (\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = 0\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \[CapitalOmega]0, 0]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)] + 0.5)) + off;
      
     sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=Amp*(1-ffpdocthermSM[Pn1,\[Eta],\[CapitalOmega]0,\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],t}])+off;
      
	 combineDataSM[data_] := Transpose[Drop[Transpose[data[[All]]],{3}]];
      
      FitPeak[combineDataSM[dataOC],
				sffpdocthermSM[If[ToggleFree[[1]] == 1, PnM1, TPnM1],
        					\[Eta]LD[\[Nu]M1 MHz, If[ToggleFree[[2]] == 1, \[Alpha]1x, T\[Alpha]1x]],
        					2 \[Pi] If[ToggleFree[[3]] == 1, \[CapitalOmega]cc, T\[CapitalOmega]cc] kHz, 
        					If[ToggleFree[[4]] == 1, \[CapitalGamma]dec, T\[CapitalGamma]dec] kHz,
        					If[ToggleFree[[5]] == 1, Amp, TAmp],
        					If[ToggleFree[[6]] == 1, off, Toff], 
							0.,
        					If[ToggleFree[[7]] == 1, \[Delta]t, T\[Delta]t] \[Mu]s,	
        					{0., x \[Mu]s}],
       		fitpara, x];
      
      
      Print[Tfres = Ffitres["ParameterConfidenceIntervalTable"]];
	
		Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];      

      Pres = Show[{
         ErrorListPlot[dataOC(*, PlotStyle -> Orange*)],
         Plot[          
          Amp*(1-ffpdocthermSM[PnM1, \[Eta]LD[\[Nu]M1 MHz, \[Alpha]1x], 2 \[Pi] \[CapitalOmega]cc kHz, \[CapitalGamma]dec kHz, 1., 0., 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}])+off/.Ffitres["BestFitParameters"]
          , {t, 0, t1}
          , PlotRange -> {All, {0, TAmp + Toff}}
          , PlotPoints -> IntegerPart[200]
          , MaxRecursion -> 0
         (* , PlotStyle -> {Thick, Orange}*)
          ]
         }, AspectRatio -> 1/4, ImageSize -> {Automatic, 300.}, 
        FrameLabel -> {"oc pulse duration (\[Mu]s)", "BDx fluo. cts. (200 (\[Mu]s ^-1)"}];
      
      Print["Fock state population of mode 1: "];
      FPop = Table[{i - 1, Tfres[[1, 1, i + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop, 1/(1 + nbarM1) (nbarM1/(1 + nbarM1))^x, {{nbarM1, Tnb1}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[
          makeThermalPop[nbarM1, NFock - 1] /. 
            Ffitres["BestFitParameters"] // Flatten, 
          FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[          
          Table[{Tfres[[1, 1, i + 1, 2]], 
            Tfres[[1, 1, i + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];
      
      Print[Pres];
      }
     Return[];
   
   ];



"for OC: coherent state";
makeCoherentPop[nbar_,maxFock_]:=Table[{(nbar^i/ (i!)) Exp[-nbar]},{i,0,maxFock}]

makeCoherent[nbar_,maxFock_]:=Sqrt[makeCoherentPop[nbar,maxFock]]

Pcoh[nb_,n_]:=Flatten[makeCoherentPop[nb,15(nb+1.)]][[n+1]]


FitOCandBSBpoi[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[Pcoh[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],Pcoh[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		Amp*(1-ffpdocthermSM[Pn1,\[Eta],1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],t}])+off,
		Amp*(1-(ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]))+off
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,1.,0.,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"],
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,1,0,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,(nbarM1^x/x!) Exp[-nbarM1],{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeCoherentPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];





FitOCandRSBpoi[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[Pcoh[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],Pcoh[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)] +0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		Amp*(1-ffpdocthermSM[Pn1,\[Eta],-1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],t}])+off,
		Amp*(1-(ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,1.,0.,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]))+off
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,1.,0.,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"],
Amp*(1-(ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],-1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,1,0,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]))+off/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,(nbarM1^x/x!) Exp[-nbarM1],{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeCoherentPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];




FitOCandBSBpoi2[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[Pcoh[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],Pcoh[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		ffpdocthermSM[Pn1,\[Eta],1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],t}],
		ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"],
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,(nbarM1^x/x!) Exp[-nbarM1],{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeCoherentPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];




FitOCandRSBpoi2[{dataOC_,dataBSB_},ExpPara_,StartValues_,ToggleFree_,NFock_]:=
Module[{nbarM1,FPop,Pres,\[Nu]M1,Tnb1,T\[Alpha]1x,T\[CapitalOmega]cc,T\[Delta]\[CapitalOmega]sb,T\[CapitalGamma]dec,TAmp,Toff,T\[Delta]t,TPnM1,PnM1,\[Alpha]1x,\[CapitalOmega]cc,\[Delta]\[CapitalOmega]sb,\[CapitalGamma]dec,Amp,off,\[Delta]t,t1,Tfres,fitpara,ffpdocthermSM,sffpdocthermSM,combineDataErrSM,combineDataSM},
{\[Nu]M1=Abs[ExpPara[[1]]-ExpPara[[2]]];
Print["mode freq. \!\(\*SubscriptBox[\(\[Omega]\), \(M1\)]\)/(2\[Pi])(MHz): ",\[Nu]M1];
Tnb1=StartValues[[1]];
Print["mode1 angles to \!\(\*SubscriptBox[\(\[CapitalDelta]k\), \(raman\)]\) (Deg.): ",T\[Alpha]1x=StartValues[[2]]];
Print["LD-parameters: ",\[Eta]LD[\[Nu]M1 MHz,T\[Alpha]1x]];
T\[CapitalOmega]cc= StartValues[[3]];
T\[Delta]\[CapitalOmega]sb=StartValues[[4]];
T\[CapitalGamma]dec=StartValues[[5]];
TAmp=StartValues[[6]];
Toff=StartValues[[7]];
T\[Delta]t=StartValues[[8]];


"prepare fock population...";
TPnM1=Table[Pcoh[Tnb1,n],{n,0,NFock-1}];
PnM1=Table[PM1[i],{i,0,NFock-1}];

"join fit parameters...";
fitpara=Join[Table[{PM1[i],Pcoh[Tnb1,i]},{i,0,NFock-1}],
{{\[Alpha]1x,T\[Alpha]1x},
{\[CapitalOmega]cc,T\[CapitalOmega]cc},
{\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb},
{\[CapitalGamma]dec,T\[CapitalGamma]dec},
{Amp,TAmp},
{off,Toff},
{\[Delta]t,T\[Delta]t}
}];


Print[fitpara];

t1=IntegerPart[Max[dataBSB[[All,1]]]]+1;

"define fit fct.:";
ffpdocthermSM[Pn1_,\[Eta]_,\[CapitalDelta]n_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_}]:=
amp*(1-((0.5-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n1 = If[\[CapitalDelta]n > 0, 0, Abs[\[CapitalDelta]n]]\), \(Length[Pn1] - 1\)]\(
\*FractionBox[
SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)], \((
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)])\)]*Abs[Pn1[\([n1 + 1]\)]]*
\*SuperscriptBox[\(Sin[
\*FractionBox[
SqrtBox[\(
\*SuperscriptBox[\(\[CapitalOmega]oc[\[Eta], n1, \((\[CapitalOmega]0 + \[Delta]\[CapitalOmega]sb)\), \[CapitalDelta]n]\), \(2\)] + 
\*SuperscriptBox[\((\[Omega] - \[Omega]oc)\), \(2\)]\)], \(2\)] \((t + \[Delta]t)\)]\), \(2\)]\)\)))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;

sffpdocthermSM[Pn1_,\[Eta]_,\[CapitalOmega]0_,\[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_,Amp_,off_,\[Omega]oc_,\[Delta]t_,{\[Omega]_,t_},{t1_}]:=
		If[t<t1,
		ffpdocthermSM[Pn1,\[Eta],-1.,\[CapitalOmega]0,\[Delta]\[CapitalOmega]sb,\[Eta]*\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],t}],
		ffpdocthermSM[Pn1,\[Eta],0.,\[CapitalOmega]0,0.,\[CapitalGamma]dec,Amp,off,\[Omega]oc,\[Delta]t,{\[Omega],(t-t1)}]
		];


combineDataErrSM[data_,t1_]:=Join[
		Transpose[Transpose[data[[1]]]+{0.,0.,0.}],
		Transpose[Transpose[data[[2]]]+{t1,0.,0.}]
			];

combineDataSM[data_,t1_]:=Join[
		Transpose[Drop[Transpose[data[[1]]],{3}]+{0.,0.}],
		Transpose[Drop[Transpose[data[[2]]],{3}]+{t1,0.}]
			];

FitPeak[combineDataSM[{dataBSB,dataOC},t1],
	sffpdocthermSM[If[ToggleFree[[1]]==1,PnM1,TPnM1],
	\[Eta]LD[\[Nu]M1 MHz,If[ToggleFree[[2]]==1,\[Alpha]1x,T\[Alpha]1x]],
	2\[Pi] If[ToggleFree[[3]]==1,\[CapitalOmega]cc,T\[CapitalOmega]cc] kHz, 
	2\[Pi] If[ToggleFree[[4]]==1,\[Delta]\[CapitalOmega]sb,T\[Delta]\[CapitalOmega]sb] kHz,
	If[ToggleFree[[5]]==1,\[CapitalGamma]dec,T\[CapitalGamma]dec] kHz,
	If[ToggleFree[[6]]==1,Amp,TAmp],
	If[ToggleFree[[7]]==1,off,Toff],
	If[ToggleFree[[8]]==1,\[Delta]t,T\[Delta]t]\[Mu]s,
	0.,
	{0.,x \[Mu]s},
	{t1 \[Mu]s}],
	fitpara,x];


Print[Tfres=Ffitres["ParameterConfidenceIntervalTable"]];

Print["Sum:  ",Sum[PM1[i],{i,0,NFock-1}]/.Ffitres["BestFitParameters"]];

Pres=Show[{
ErrorListPlot[{dataOC,dataBSB},PlotStyle->{Gray,Orange}],
Plot[{
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],0,2\[Pi] \[CapitalOmega]cc kHz,0.,\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"],
ffpdocthermSM[PnM1,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x],-1,2\[Pi] \[CapitalOmega]cc kHz,2\[Pi] \[Delta]\[CapitalOmega]sb kHz,\[Eta]LD[\[Nu]M1 MHz,\[Alpha]1x]*\[CapitalGamma]dec kHz,Amp,off,0.,\[Delta]t \[Mu]s,{0.,t \[Mu]s}]/.Ffitres["BestFitParameters"]
}
,{t,0,t1}
,PlotRange->{All,{0,TAmp+Toff}}
,PlotPoints->IntegerPart[220]
,MaxRecursion->0
,PlotStyle->{{Thick,Gray},{Thick,Orange}}
]
},AspectRatio->1/4,ImageSize->{Automatic,300.},FrameLabel->{"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s^-1)"}];

Print["Fock state population of mode 1: "];
FPop=Table[{i-1,Tfres[[1,1,i+1,2]]},{i,1,NFock}];
FitPeak[FPop,(nbarM1^x/x!) Exp[-nbarM1],{{nbarM1,Tnb1}},x];
Print[Ffitres["ParameterConfidenceIntervalTable"]];

Print[
Show[
{
BarChart[makeCoherentPop[nbarM1,NFock-1]/.Ffitres["BestFitParameters"]//Flatten,FrameLabel->{"fock state #+1","motional pop."}],
ErrorListPlot[
Table[{Tfres[[1,1,i+1,2]],Tfres[[1,1,i+1,3]]},{i,1,NFock}]
]
}
,PlotRange->{All,{0,1.2*Max[FPop[[All,2]]]}}
,ImageSize->{Automatic,200}
]
];

Print[Pres];
}
Return[];
];







\[Eta]lamb[\[Nu]tr_] := 
  Sqrt[2]*2 \[Pi]/(279.6 nm) Sqrt[hbar/(2*25 amu (2 \[Pi] \[Nu]tr))];
\[Eta]c[\[Nu]tr_] := \[Eta]lamb[\[Nu]tr]/Sqrt[2];
\[Eta]s[\[Nu]tr_] := \[Eta]lamb[\[Nu]tr]/Sqrt[2*Sqrt[3]]



FitOCCOMSTR[{dataOC_, dataCOM_, dataSTR_}, ExpPara_, StartValues_,ToggleFree_, NFock_] :=
  Module[{nbarM1, nbarM2, FPop1, FPop2, Pres, Pres1, Pres2 ,\[Nu]M1, \[Nu]M2, Tnb1, Tnb2, T\[CapitalOmega]cc, T\[Delta]\[CapitalOmega]sb, T\[CapitalGamma]dec, Tamp, Toff, T\[Delta]t, TPnM1, TPnM2, PnM1, PnM2, \[CapitalOmega]cc, \[Delta]\[CapitalOmega]sb, \[CapitalGamma]dec, amp, off, \[Delta]t, t0, t1, t2, tcom, tstr, Tfres,data1, fitpara, ffpdocthermMM, sffpdocthermMM, combineDataErrMM, combineDataMM},
   {\[Nu]M1 = Abs[ExpPara[[1]] - ExpPara[[2]]];
	Print["mode freq. \[Omega]_M1/(2\[Pi])(MHz): ", \[Nu]M1]; 
	Print["LD-parameter-COM: ", \[Eta]c[\[Nu]M1 MHz]];
	\[Nu]M2 = Abs[ExpPara[[1]] - ExpPara[[3]]]; 
    Print["mode freq. \[Omega]_M2/(2\[Pi])(MHz): ", \[Nu]M2];
    Print["LD-parameter-STR: ", \[Eta]s[\[Nu]M1 MHz]];
    Tnb1 = StartValues[[1]]; Tnb2 = StartValues[[2]];
    T\[CapitalOmega]cc = StartValues[[3]];
    T\[Delta]\[CapitalOmega]sb = StartValues[[4]];
    T\[CapitalGamma]dec = StartValues[[5]];
    Tamp = StartValues[[6]];
    Toff = StartValues[[7]];
    T\[Delta]t = StartValues[[8]];
      
 "prepare fock population...";
      TPnM1 = Table[P[Tnb1, n], {n, 0, NFock - 1}];
      PnM1 = Table[PM1[i], {i, 0, NFock - 1}];
      TPnM2 = Table[P[Tnb2, n], {n, 0, NFock - 1}];
      PnM2 = Table[PM2[i], {i, 0, NFock - 1}];
      
 "join fit parameters...";
     fitpara = Join[Table[{PM1[i], P[Tnb1, i]}, {i, 0, NFock - 1}],
				    Table[{PM2[i], P[Tnb2, i]}, {i, 0, NFock - 1}],
        			{{\[CapitalOmega]cc, T\[CapitalOmega]cc},
         			{\[Delta]\[CapitalOmega]sb, T\[Delta]\[CapitalOmega]sb},
         			{\[CapitalGamma]dec, T\[CapitalGamma]dec},
         			{amp, Tamp},
         			{off, Toff},
         			{\[Delta]t, T\[Delta]t}}
                    ];
      
      
Print[fitpara];
t0 =  IntegerPart[Max[dataOC[[All, 1]]]]+1;    
t1 = IntegerPart[Max[dataCOM[[All, 1]]]]+1;  
t2 = IntegerPart[Max[dataSTR[[All, 1]]]]+1;
      
"define fit fct.:";

ffpdocthermMM[Pn1_, Pn2_, \[Eta]_, \[CapitalDelta]n_, \[CapitalOmega]cc_, \[Delta]\[CapitalOmega]sb_, \[CapitalGamma]dec_, amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}]:=

amp*(1.-((0.5-(Sum[Sum[\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2/(\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2+(\[Omega]-\[Omega]oc)^2)*Abs[Pn1[[n1+1]]]*Abs[Pn2[[n2+1]]]*Sin[Sqrt[\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2 +(\[Omega]-\[Omega]oc)^2]*(t+\[Delta]t)/2]^2,{n2,0,Length[Pn2]-1}],{n1,0,Length[Pn1]-1}]))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;


sffpdocthermMM[Pn1_, Pn2_, \[Eta]_, \[CapitalOmega]cc_, \[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_, amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}, {t1_, t2_}] := 
If[t < t1,
        ffpdocthermMM[Pn1,Pn2, \[Eta], {1,0}, \[CapitalOmega]cc, \[Delta]\[CapitalOmega]sb, \[CapitalGamma]dec, amp, off, \[Omega]oc, \[Delta]t, {\[Omega], t}],
        If[t < (t1 + t2),
         	ffpdocthermMM[Pn1, Pn2, \[Eta], {0, 1}, \[CapitalOmega]cc, 0, \[CapitalGamma]dec, amp, off, \[Omega]oc, \[Delta]t, {\[Omega], (t - t1)}],
             amp*(1 - ffpdocthermMM[Pn1, Pn2, \[Eta], {0, 0}, \[CapitalOmega]cc, 0., \[CapitalGamma]dec, 1, 0, \[Omega]oc, \[Delta]t, {\[Omega], (t - (t1 +t2))}]) + off]];
      
combineDataErrMM[data_, t1_, t2_]:= Join[
        		Transpose[Transpose[data[[2]]] + {0., 0., 0.}],
        		Transpose[Transpose[data[[3]]] + {t1, 0., 0.}],
                Transpose[Transpose[data[[1]]] + {(t1 + t2), 0., 0.}]
        			];
      
      		
combineDataMM[data_, t1_, t2_]:= Join[
        		Transpose[Drop[Transpose[data[[2]]], {3}] + {0., 0.}],
        		Transpose[Drop[Transpose[data[[3]]], {3}] + {t1, 0.}],
                Transpose[Drop[Transpose[data[[1]]], {3}] + {(t1 + t2), 0.}]
        			];
      

FitPeak[combineDataMM[{dataOC, dataCOM, dataSTR}, t1, t2], 
		sffpdocthermMM[If[ToggleFree[[1]] == 1, PnM1, TPnM1],
        			   If[ToggleFree[[2]] == 1, PnM2, TPnM2],
        			   {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M2 MHz]},
        			   2 \[Pi] If[ToggleFree[[3]] == 1, \[CapitalOmega]cc, T\[CapitalOmega]cc] kHz, 
        			   2 \[Pi] If[ToggleFree[[4]] == 1, \[Delta]\[CapitalOmega]sb, T\[Delta]\[CapitalOmega]sb] kHz,
        			   If[ToggleFree[[5]] == 1, \[CapitalGamma]dec, T\[CapitalGamma]dec] kHz,
        			   If[ToggleFree[[6]] == 1, amp, Tamp],
        			   If[ToggleFree[[7]] == 1, off, Toff],
        			   0.,
        			   If[ToggleFree[[8]] == 1, \[Delta]t, T\[Delta]t] \[Mu]s,
        			   {0., x \[Mu]s},
        			   {t1 \[Mu]s, t2 \[Mu]s}
        			  ],
       fitpara, 
	   x];
      
      
Print[Tfres = Ffitres["ParameterConfidenceIntervalTable"]];
      
      
Pres = Show[{	
        ErrorListPlot[{dataOC}, 
           PlotStyle -> {Gray, Orange, Red}],
        Plot[{amp*(1. - ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M1 MHz]}, {0, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 0., \[CapitalGamma]dec kHz, 1, 0, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}]) + off /.Ffitres["BestFitParameters"]},  
			{t,0,t0},    
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {Thick, Gray}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];
 
Pres1 = Show[{	
        ErrorListPlot[{dataCOM}, 
           PlotStyle -> { Orange}],
        Plot[{
              ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M1 MHz]}, {1, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 2 \[Pi] \[Delta]\[CapitalOmega]sb kHz, \[Eta]lamb[\[Nu]M1 MHz]*\[CapitalGamma]dec kHz, amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}] /. Ffitres["BestFitParameters"]},
          	{t, 0, t1}, 
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {Thick, Orange}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"COM pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];

Pres2 = Show[{	
        ErrorListPlot[{dataSTR}, 
           PlotStyle -> { Red}],
        Plot[{ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M1 MHz]}, {0, 1}, 2 \[Pi] \[CapitalOmega]cc kHz, 0., \[Eta]lamb[\[Nu]M1 MHz]*\[CapitalGamma]dec kHz, amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}] /. Ffitres["BestFitParameters"]},
          	{t, 0, t2}, 
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {Thick,Red}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"STR pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];     

Print[Pres];
Print[Pres1];
Print[Pres2];

Print["Fock state population of mode 1: "];
      FPop1 = Table[{i - 1, Tfres[[1, 1, i + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop1, 1/(1 + nbarM1) (nbarM1/(1 + nbarM1))^x, {{nbarM1, Tnb1}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[makeThermalPop[nbarM1, NFock - 1] /. Ffitres["BestFitParameters"] // Flatten, FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[ Table[{Tfres[[1, 1, i + 1, 2]], Tfres[[1, 1, i + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop1[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];      

Print["Fock state population of mode 2: "];
      FPop2 = Table[{i - 1, Tfres[[1, 1, i + NFock + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop2, 1/(1 + nbarM2) (nbarM2/(1 + nbarM2))^x, {{nbarM2, Tnb2}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[
          makeThermalPop[nbarM2, NFock - 1] /. Ffitres["BestFitParameters"] // Flatten, FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[ Table[{Tfres[[1, 1, i +  NFock + 1, 2]], Tfres[[1, 1, i +  NFock + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop2[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];      
    

     
	 }
     Return[];
   ];



FitOCCOM[{dataOC_, dataCOM_}, ExpPara_, StartValues_,ToggleFree_, NFock_] :=
  Module[{nbarM1, nbarM2, FPop1, FPop2, Pres, Pres1 ,\[Nu]M1, \[Nu]M2, Tnb1, Tnb2, T\[CapitalOmega]cc, T\[Delta]\[CapitalOmega]sb, T\[CapitalGamma]dec, Tamp, Toff, T\[Delta]t, TPnM1, TPnM2, PnM1, PnM2, \[CapitalOmega]cc, \[Delta]\[CapitalOmega]sb, \[CapitalGamma]dec, amp, off, \[Delta]t, t1, tcom, tstr, Tfres,data1, fitpara, ffpdocthermMM, sffpdocthermMM, combineDataErrMM, combineDataMM},
   {\[Nu]M1 = Abs[ExpPara[[1]] - ExpPara[[2]]];
	Print["mode freq. \[Omega]_M1/(2\[Pi])(MHz): ", \[Nu]M1]; 
	Print["LD-parameter-COM: ", \[Eta]c[\[Nu]M1 MHz]];
	\[Nu]M2 = Abs[ExpPara[[1]] - ExpPara[[3]]]; 
    Print["mode freq. \[Omega]_M2/(2\[Pi])(MHz): ", \[Nu]M2];
    Print["LD-parameter-STR: ", \[Eta]s[\[Nu]M1 MHz]];
    Tnb1 = StartValues[[1]]; Tnb2 = StartValues[[2]];
    T\[CapitalOmega]cc = StartValues[[3]];
    T\[Delta]\[CapitalOmega]sb = StartValues[[4]];
    T\[CapitalGamma]dec = StartValues[[5]];
    Tamp = StartValues[[6]];
    Toff = StartValues[[7]];
    T\[Delta]t = StartValues[[8]];
      
 "prepare fock population...";
      TPnM1 = Table[P[Tnb1, n], {n, 0, NFock - 1}];
      PnM1 = Table[PM1[i], {i, 0, NFock - 1}];
      TPnM2 = Table[P[Tnb2, n], {n, 0, NFock - 1}];
      PnM2 = Table[PM2[i], {i, 0, NFock - 1}];
      
 "join fit parameters...";
     fitpara = Join[Table[{PM1[i], P[Tnb1, i]}, {i, 0, NFock - 1}],
				    Table[{PM2[i], P[Tnb2, i]}, {i, 0, NFock - 1}],
        			{{\[CapitalOmega]cc, T\[CapitalOmega]cc},
         			{\[Delta]\[CapitalOmega]sb, T\[Delta]\[CapitalOmega]sb},
         			{\[CapitalGamma]dec, T\[CapitalGamma]dec},
         			{amp, Tamp},
         			{off, Toff},
         			{\[Delta]t, T\[Delta]t}}
                    ];
      
      
Print[fitpara];
      
t1 = IntegerPart[Max[dataCOM[[All, 1]]]]+1;  
      
"define fit fct.:";

ffpdocthermMM[Pn1_, Pn2_, \[Eta]_, \[CapitalDelta]n_, \[CapitalOmega]cc_, \[Delta]\[CapitalOmega]sb_, \[CapitalGamma]dec_, amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}]:=

amp*(1.-((0.5-(Sum[Sum[\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2/(\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2+(\[Omega]-\[Omega]oc)^2)*Abs[Pn1[[n1+1]]]*Abs[Pn2[[n2+1]]]*Sin[Sqrt[\[CapitalOmega]ocMM[\[Eta],{n1,n2},(\[CapitalOmega]cc+\[Delta]\[CapitalOmega]sb),\[CapitalDelta]n]^2 +(\[Omega]-\[Omega]oc)^2]*(t+\[Delta]t)/2]^2,{n2,0,Length[Pn2]-1}],{n1,0,Length[Pn1]-1}]))*Exp[-\[CapitalGamma]dec*(t+\[Delta]t)]+0.5))+off;


sffpdocthermMM[Pn1_, Pn2_, \[Eta]_, \[CapitalOmega]cc_, \[Delta]\[CapitalOmega]sb_,\[CapitalGamma]dec_, amp_, off_, \[Omega]oc_, \[Delta]t_, {\[Omega]_, t_}, t1_] := 
If[t < t1,
        ffpdocthermMM[Pn1,Pn2, \[Eta], {1,0}, \[CapitalOmega]cc, \[Delta]\[CapitalOmega]sb, \[CapitalGamma]dec, amp, off, \[Omega]oc, \[Delta]t, {\[Omega], t}],
         amp*(1 - ffpdocthermMM[Pn1, Pn2, \[Eta], {0, 0}, \[CapitalOmega]cc, 0., \[CapitalGamma]dec, 1, 0, \[Omega]oc, \[Delta]t, {\[Omega], (t - t1)}]) + off];
      
combineDataErrMM[data_, t1_]:= Join[
        		Transpose[Transpose[data[[2]]] + {0., 0., 0.}],
                Transpose[Transpose[data[[1]]] + {t1 , 0., 0.}]
        			];
      
      		
combineDataMM[data_, t1_]:= Join[
        		Transpose[Drop[Transpose[data[[2]]], {3}] + {0., 0.}],
        	    Transpose[Drop[Transpose[data[[1]]], {3}] + {t1 , 0.}]
        			];
      

FitPeak[combineDataMM[{dataOC, dataCOM}, t1], 
		sffpdocthermMM[If[ToggleFree[[1]] == 1, PnM1, TPnM1],
        			   If[ToggleFree[[2]] == 1, PnM2, TPnM2],
        			   {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M2 MHz]},
        			   2 \[Pi] If[ToggleFree[[3]] == 1, \[CapitalOmega]cc, T\[CapitalOmega]cc] kHz, 
        			   2 \[Pi] If[ToggleFree[[4]] == 1, \[Delta]\[CapitalOmega]sb, T\[Delta]\[CapitalOmega]sb] kHz,
        			   If[ToggleFree[[5]] == 1, \[CapitalGamma]dec, T\[CapitalGamma]dec] kHz,
        			   If[ToggleFree[[6]] == 1, amp, Tamp],
        			   If[ToggleFree[[7]] == 1, off, Toff],
        			   0.,
        			   If[ToggleFree[[8]] == 1, \[Delta]t, T\[Delta]t] \[Mu]s,
        			   {0., x \[Mu]s},
        			   t1 \[Mu]s
        			  ],
       fitpara, 
	   x];
      
      
Print[Tfres = Ffitres["ParameterConfidenceIntervalTable"]];
      
      
Pres = Show[{	
        ErrorListPlot[{dataOC}, 
           PlotStyle -> {Gray, Orange, Red}],
        Plot[{amp*(1. - ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M1 MHz]}, {0, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 0., \[CapitalGamma]dec kHz, 1, 0, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}]) + off /.Ffitres["BestFitParameters"]},  
			{t,0,t1/2},    
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {Thick, Gray}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];
 
Pres1 = Show[{	
        ErrorListPlot[{dataCOM}, 
           PlotStyle -> { Orange}],
        Plot[{
              ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M1 MHz]}, {1, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 2 \[Pi] \[Delta]\[CapitalOmega]sb kHz, \[Eta]lamb[\[Nu]M1 MHz]*\[CapitalGamma]dec kHz, amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}] /. Ffitres["BestFitParameters"]},
          	{t, 0, t1}, 
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {Thick, Orange}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"COM pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];

Print[Pres];
Print[Pres1];

Print["Fock state population of mode 1: "];
      FPop1 = Table[{i - 1, Tfres[[1, 1, i + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop1, 1/(1 + nbarM1) (nbarM1/(1 + nbarM1))^x, {{nbarM1, Tnb1}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[makeThermalPop[nbarM1, NFock - 1] /. Ffitres["BestFitParameters"] // Flatten, FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[ Table[{Tfres[[1, 1, i + 1, 2]], Tfres[[1, 1, i + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop1[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];      

Print["Fock state population of mode 2: "];
      FPop2 = Table[{i - 1, Tfres[[1, 1, i + NFock + 1, 2]]}, {i, 1, NFock}];
      FitPeak[FPop2, 1/(1 + nbarM2) (nbarM2/(1 + nbarM2))^x, {{nbarM2, Tnb2}}, x];
      Print[Ffitres["ParameterConfidenceIntervalTable"]];
      
      Print[
       Show[{
         BarChart[
          makeThermalPop[nbarM2, NFock - 1] /. Ffitres["BestFitParameters"] // Flatten, FrameLabel -> {"fock state #+1", "motional pop."}],
         ErrorListPlot[ Table[{Tfres[[1, 1, i +  NFock + 1, 2]], Tfres[[1, 1, i +  NFock + 1, 3]]}, {i, 1, NFock}]
          ]
         }
        , PlotRange -> {All, {0, 1.2*Max[FPop2[[All, 2]]]}}
        , ImageSize -> {Automatic, 200}
        ]
       ];      
    

     
	 }
     Return[];
   ];



(*
Pres = Show[{	
        ErrorListPlot[{dataOC, dataCOM, dataSTR}, 
           PlotStyle -> {Gray, Orange, Red}],
        Plot[{amp*(1. - ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M2 MHz]}, {0, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 0., \[CapitalGamma]dec kHz, 1, 0, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}]) + off /.Ffitres["BestFitParameters"],
              ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M2 MHz]}, {1, 0}, 2 \[Pi] \[CapitalOmega]cc kHz, 2 \[Pi] \[Delta]\[CapitalOmega]sb kHz, \[Eta]lamb[\[Nu]M1 MHz]*\[CapitalGamma]dec kHz, amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}] /. Ffitres["BestFitParameters"],
              ffpdocthermMM[PnM1, PnM2, {\[Eta]c[\[Nu]M1 MHz], \[Eta]s[\[Nu]M2 MHz]}, {0, 1}, 2 \[Pi] \[CapitalOmega]cc kHz, 2 \[Pi] \[Delta]\[CapitalOmega]sb kHz, \[Eta]lamb[\[Nu]M1 MHz]*\[CapitalGamma]dec kHz, amp, off, 0., \[Delta]t \[Mu]s, {0., t \[Mu]s}] /. Ffitres["BestFitParameters"]},
          	{t, 0, t2}, 
		PlotPoints -> IntegerPart[300], 
        MaxRecursion -> 0, 
        PlotStyle -> {{Thick, Gray}, {Thick, Orange}, {Thick,Red}}]},
        PlotRange -> {All, {0, Tamp + Toff}},
        AspectRatio -> 1/2,
        ImageSize -> {Automatic, 300.},
        FrameLabel -> {"oc pulse duration (\[Mu]s)","BDx fluo. cts. (200 \[Mu]s)^(-1)"}];*)


(* ::Subsection::Closed:: *)
(*Uni FR QS DAQ data treatment*)


LoadPaulaData[Dpath_,Dfile_,DropLast_,Prt_]:=Module[{dataRAW,dataErr},
dataRAW=Import[Dpath<>Dfile];
dataErr=Table[SetPrecision[{dataRAW[[i,1]],dataRAW[[i,2]]/1000,dataRAW[[i,3]]/1000},8],{i,48,Length[dataRAW]-DropLast}];
If[Prt==1,
Print[ErrorListPlot[dataErr,FrameLabel->{"scan para.","fluo. rate (kHz)"}]]
];
Return[dataErr]
]

LoadBermudaData[BERdatapath_,DateFolder_,ExpFolder_,times_]:=Module[{data,dataErr,fileNames,numFiles,dataStart,dataEnd,fileTimesString,variable,varPos,legend},{
fileNames=Table[BERdatapath<>DateFolder<>ExpFolder<>times[[i]]<>".dat",{i,1,Length[times]}];
numFiles=Length[fileNames];

dataStart=Table[1+Length[FindList[fileNames[[i]],"#"]],{i,1,numFiles}];
dataEnd=Table[dataStart[[i]]-1+ToExpression[StringDrop[StringDrop[FindList[fileNames[[i]],"#<datasets>"],11],-11][[1]]],{i,1,numFiles}];


fileTimesString=Table[{StringTake[Import[fileNames[[i]]][[3,2]],{8,15}]},{i,1,numFiles}];

variable=Flatten[Table[FindList[fileNames[[i]],"<int_points>"],{i,1,numFiles}]];
varPos=Table[Flatten[StringPosition[variable[[1]],"\""]],{i,1,numFiles}];
variable=Table[StringDrop[StringTake[variable[[i]],varPos[[i,3]]-1],varPos[[i,1]]],{i,1,numFiles}];

legend=DateFolder<>"\n"<>ExpFolder<>"\n"<>Table["file #"<>ToString[i]<>": "<>fileTimesString[[i,1]]<>"\n",{i,1,numFiles}];

data=Table[(Import[fileNames[[i]]])[[dataStart[[i]]+1;;dataEnd[[i]],1;;2]],{i,1,numFiles}];

dataErr=Table[{data[[i,k,1]],data[[i,k,2]],1/Sqrt[data[[i,k,2]]]},{i,1,numFiles},{k,1,Length[data[[i]]]}];
Print[
ErrorListPlot[
Table[
dataErr[[i]]
,{i,1,numFiles}
]
,PlotStyle->{Black,Red,Blue}
,PlotLegends->legend
,FrameLabel->{variable[[1]],"BDx cts"}
]];
Return[{dataErr,data}]
}];


(* ::Subsection::Closed:: *)
(*NIST QC DAQ data treatment*)


ImpExpData[dir_]:=Module[
{impdat,impdatgrp,impdatave,groupval,i,j,k,datastart,datafile,paramdir,paramfile,paramdat,numexpline,
numexps,numnormdetsline,numnormdets,histsizeline,histsize,groupave,groupavs,thisgroup,thisgroupsum,groupnumexps},
datafile=FileNames["histData*.csv",{dir}][[1]];
paramdir=FileNameJoin[{dir,"params"}];
paramfile=FileNames["*.txt",{paramdir}][[1]];
paramdat=Import[paramfile,"TSV"];

numexpline=Select[paramdat,StringTake[#1[[1]],8]=="{Num Exp"&][[1]];
numexps=ToExpression[StringCases[numexpline,NumberString][[1]][[1]]<>".0"];

numnormdetsline=Select[paramdat,StringTake[#1[[1]],15]=="{Num Normal Det"&][[1]];
numnormdets=ToExpression[StringCases[numnormdetsline,NumberString][[1]][[1]]];

histsizeline=Select[paramdat,StringTake[#1[[1]],10]=="{Hist Size"&][[1]];
histsize=ToExpression[StringCases[histsizeline,NumberString][[1]][[1]]];

impdat=Import[datafile];
datastart=Length[impdat[[1]]]-histsize*numnormdets+1;
impdat=impdat[[2;;,Flatten[{3,Range[datastart,datastart+histsize*numnormdets-1]}]]];
impdatgrp=GatherBy[impdat,#[[1]]&];
impdatave={};
For[
j=1,j<=Length[impdatgrp],j++,
groupval=impdatgrp[[j]][[1,1]];
thisgroup=impdatgrp[[j]][[All,2;;]];
thisgroupsum=Total[thisgroup];
groupavs={};
groupnumexps={};
For[
k=1,k<=numnormdets,k++,
thishistogram=thisgroupsum[[(k-1)*histsize+1;;k*histsize]];
AppendTo[groupnumexps,Total[thishistogram]];
AppendTo[groupavs,Range[0,histsize-1].thishistogram/groupnumexps[[k]]];
];
groupavs=N[groupavs];
groupave={};
For[
k=1,k<=numnormdets,k++,
thishistogram=thisgroupsum[[(k-1)*histsize+1;;k*histsize]];
AppendTo[groupave,Sqrt[thishistogram.(Range[0,histsize-1]-groupavs[[k]])^2/(groupnumexps[[k]]-1)]/Sqrt[groupnumexps[[k]]]];
];
impdatave=Append[impdatave,Flatten[{groupval,Transpose[{groupavs,groupave,groupnumexps}]}]];
];
Return[impdatave];
];


GetIPfile[Dpath_,Dprefix_,Ddate_,Dfile_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath<>"dc"<>fillpath;
Dfilename= "IonProperties_inc.dc";
(*Print[">> You just loaded data from that file:\n"<>Dfilepath<>Dfilename<>"\n   ->The data is accessible via 'DData' "];*)
Import[Dfilepath<>Dfilename,"TEXT"]
];


ClickLoadDataQC[Dpath_,l_]:=Module[
{paramdir,paramfile,paramdat,numnormdetsline,numnormdets,ELP,Dy,i,m},
Off[StringTake::"take"];
Off[Power::"infy"];
Off[\[Infinity]::"indet"];
paramdir=FileNameJoin[{Dpath,"params"}];
paramfile=FileNames["*.txt",{paramdir}][[1]];
paramdat=Import[paramfile,"TSV"];

numnormdetsline=Select[paramdat,StringTake[#1[[1]],15]=="{Num Normal Det"&][[1]];
numnormdets=ToExpression[StringCases[numnormdetsline,NumberString][[1]][[1]]];
Dy=Table[2+(i-1)*3,{i,1,numnormdets}];

DData=ImpExpData[Dpath];
For[i=1,i<Length[Dy]+1,
{
m=Dy[[i]];
If[ToString[DData[[1,m]]/DData[[1,m]]]!="1.",
{Dy=Drop[Dy,{i}]}
];
i++
}]
DataErr=Table[DData[[1;;(Length[DData]),{1,Dy[[i]],Dy[[i]]+1}]],{i,1,Length[Dy]}];

ELP={};

ELP=Table[
ErrorListPlot[DataErr[[i]],PlotStyle->RGBColor[1-(i/numnormdets),0,0],
PlotRange->{{DData[[1,1]],Last[DData][[1]]},All}],
{i,1,numnormdets}];
Show[ELP,
PlotLabel->Style["file #"<>ToString[l]<>": "<>StringDrop[StringTake[Dpath,-25],-1]]]
];

BrowseFiles[DPath_]:=FlipView[Quiet[Table[ClickLoadDataQC[ToString[FileNames[DPath<>"*"][[i]]<>"/"],i],{i,1,Length[FileNames[DPath<>"*"]]}]]];



FLoadDataQC[Dpath_,Dprefix_,Ddate_,Dfile_,Dy_,DropLast_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath;
Dfilename= "histData." <>StringDrop[Dprefix<>Ddate<>"_"<>Dfile,StringLength[Dprefix]] <> ".csv";
(*Print[">> You just loaded data from that file:\n"<>Dfilepath<>Dfilename<>"\n   ->The data is accessible via 'DData' "];*)
Off[StringTake::"take"];
Off[Power::"infy"];
Off[\[Infinity]::"indet"];

DData=ImpExpData[Dfilepath];
DataErr=Table[DData[[1;;(Length[DData]-DropLast),{1,Dy[[i]],Dy[[i]]+1}]],{i,1,Length[Dy]}];
ErrorListPlot[DataErr]
];


FLoadDataQCOLD[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,Dy_,DropLast_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath;
Dfilename= "histData." <>StringDrop[Dprefix<>Ddate<>"_"<>Dfile,StringLength[Dprefix]] <> ".csv";
(*Print[">> You just loaded data from that file:\n"<>Dfilepath<>Dfilename<>"\n   ->The data is accessible via 'DData' "];*)
Dcache=Import[Dfilepath<>Dfilename];
DData=Drop[Drop[Transpose[{ToExpression[Dcache[[All, Dx]]], ToExpression[Dcache[[All,Dy]]]}],{1}],-DropLast];
(*Print[">> Exp. Setting:"];
For[i=1, i<(Dy-Dx),i++,{
If[Dcache[[1, i+Dx]]!=Dcache[[1, Dx]],
Print[Dcache[[1, i+Dx]]<>": "<>ToString[ToExpression[Dcache[[2, i+Dx]]]]];
]
}];*)
ListPlot[Sort[DData],PlotRange->All,Joined->True,FrameLabel->{Dcache[[1, Dx]],"counts (arb. units)"}]
];


FLoadDataQCHist[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,Dy_,HistStart_,DropLast_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i,j},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath;
Dfilename= "histData." <>StringDrop[Dprefix<>Ddate<>"_"<>Dfile,StringLength[Dprefix]] <> ".csv";
FLoadDataQCOLD[Dpath,Dprefix,Ddate,Dfile,Dx,Dy,DropLast];
DDataAve=DData;
Dcache=Import[Dfilepath<>Dfilename];
DData=Table[Table[ToExpression[Dcache[[j,i]]],{i,HistStart,HistStart+49}],{j,2,Length[DDataAve]+1}];
Manipulate[BarChart[DData[[i]]],{i,1,Length[DData],1}]
];


FLoadDataQCHistOLD[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,Dy_,Dz_,DataRange_,HistStart_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i,j},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath;
Dfilename= "histData." <>StringDrop[Dprefix<>Ddate<>"_"<>Dfile,StringLength[Dprefix]] <> ".csv";
Dcache=Import[Dfilepath<>Dfilename];
DData=Table[Table[ToExpression[Dcache[[j,i]]],{i,HistStart,HistStart+49}],{j,DataRange[[1]],DataRange[[2]]}];
DDataAve={ToExpression[Dcache[[2, Dx]]], ToExpression[Dcache[[2,Dy]]],ToExpression[Dcache[[2,Dz]]]};
Manipulate[BarChart[DData[[i]]],{i,1,Length[DData],1}]
];


FLoadDataQCHistFits[Dpath_,Dprefix_,Ddate_,Dfile_,Dx_,Dy_,DropLast_]:=Module[
{Dpathfile,Dfilepath,Dfilename,Dcache,i},
Dpathfile=Dprefix<>Ddate<>"_"<>Dfile;
Dfilepath=Dpath<>
StringInsert[StringInsert[Ddate,"-",5],"-",8]<>fillpath
<>StringDrop[Dpathfile,-StringLength[Ddate<>"_"<>Dfile]-3]
<>fillpath
<>StringInsert[StringInsert[StringInsert[StringDrop[StringInsert[StringInsert[StringDrop[Dpathfile,StringLength[Dprefix]],"-",5],"-",8],{11}],"--",11],".",15],".",18]
<>fillpath;
Dfilename= "histFitData." <>StringDrop[Dprefix<>Ddate<>"_"<>Dfile,StringLength[Dprefix]] <> ".csv";
(*Print[">> You just loaded data from that file:\n"<>Dfilepath<>Dfilename<>"\n   ->The data is accessible via 'DData' "];*)
Dcache=Import[Dfilepath<>Dfilename];
DData0=Drop[Drop[Transpose[{ToExpression[Dcache[[All, Dx]]], ToExpression[Dcache[[All,Dy]]], ToExpression[Dcache[[All,Dy+1]]]}],{1}],-DropLast];
DData1=Drop[Drop[Transpose[{ToExpression[Dcache[[All, Dx]]], ToExpression[Dcache[[All,Dy+2]]], ToExpression[Dcache[[All,Dy+3]]]}],{1}],-DropLast];
DData2=Drop[Drop[Transpose[{ToExpression[Dcache[[All, Dx]]], ToExpression[Dcache[[All,Dy+4]]], ToExpression[Dcache[[All,Dy+5]]]}],{1}],-DropLast];
(*Print[">> Exp. Setting:"];
For[i=1, i<(Dy-Dx),i++,{
If[Dcache[[1, i+Dx]]!=Dcache[[1, Dx]],
Print[Dcache[[1, i+Dx]]<>": "<>ToString[ToExpression[Dcache[[2, i+Dx]]]]];
]
}];*)
ErrorListPlot[{Sort[DData0],Sort[DData1],Sort[DData2]},PlotRange->{All,{0,1}},FrameLabel->{Dcache[[1, Dx]],"pop (arb. units)"}]
];


(* ::Subsection:: *)
(*some handy Modules*)


BinIt[Ddata_,NO_]:=Module[
{SDdata},
SDdata=Sort[Ddata];
BinnedData=Table[{Sum[SDdata[[i+n,1]],{n,0,NO-1}]/NO,Sum[SDdata[[i+n,2]],{n,0,NO-1}]/NO},{i,1,Length[SDdata],NO}];
];


FitPeak[Fdata_,Ffitfun_,Ffitpara_,x_]:=Module[
{},
Ffitres=NonlinearModelFit[Fdata,Ffitfun,Ffitpara,x,Method->"LevenbergMarquardt"];
(*Print["You just fitted the data with: "<>ToString[Ffitres[x],StandardForm]<>"\n   ->The fit result is accessible via 'Ffitres' "];
Print[Ffitres["ParameterConfidenceIntervalTable"]];*)
Plot[Ffitres[x],{x,Min[Fdata[[All,1]]], Max[Fdata[[All,1]]]},Epilog->{Point@Fdata},PlotRange->All]
];


AnalyzeTrapWM[pot_,Mass_,x_,y_,z_,ions_] := Module[
	{MinPosMicron,MinPosMeter,MinPosMicronRuleRule,MinPosMeterRuleRule,\[Omega]modes,es,evs,NIons,
	PotAllIonsMeter,PotAllIonsMicron,PotAllIons,PotMat,PotMatNew,
	i,j,k,l,
	xiTable,xiTableFlat,xi,
	InitialConditions,LengthUnit,mass,
	buff,
	outstring
	},

	(* How many ions? *)
	NIons=Length[ions];

	(* Table with cartesian coordinates for each ion *)
	xiTable=Table[xi[i,j],{i,1,NIons},{j,1,3}];

	(* Flat table *)
	xiTableFlat=Flatten[Table[xi[i,j],{i,1,NIons},{j,1,3}],1];
	
	(* Initial positions for the ions *)
	InitialConditions=Flatten[
		Table[
			{xi[i,j],ions[[i,2,j]]/\[Mu]m},
			{i,1,NIons},
			{j,1,3}
		],1
	];

	(* Potential for all ions together, length unit still open *)
	PotAllIons=Sum[
		(* External potential at the position of each ion *)
		(pot/.{
			x->xi[i,1] LengthUnit,
			y->xi[i,2] LengthUnit,
			z->xi[i,3] LengthUnit,
      Mass ->ions[[i,1]]
		}) +
		(* Coulomb repulsion between ions *)
		Sum[e/(4\[Pi] \[Epsilon]0)/Sqrt[(xiTable[[i]] LengthUnit - xiTable[[j]] LengthUnit).(xiTable[[i]] LengthUnit - xiTable[[j]] LengthUnit)],{j,1,i-1}],
		{i,1,NIons}
	];
	
	(* Lenth unit microns *)
	PotAllIonsMicron=PotAllIons/.{LengthUnit->\[Mu]m};

	(* Length unit meters *)
	PotAllIonsMeter=PotAllIons/.{LengthUnit->1};

	(* Find the ion positions that minimize the total energy *)
	MinPosMicronRuleRule = FindMinimum[
		PotAllIonsMicron,
		InitialConditions
	][[2]];

	(* Minimum positions in other units *)
	MinPosMicron=xiTable/.MinPosMicronRuleRule;
	MinPosMeter=MinPosMicron*\[Mu]m;
	MinPosMeterRuleRule=Thread[Rule[xiTableFlat,Flatten[MinPosMeter]]];
	outstring="";
	
	For[i=1,i<=NIons,i++,
		Print["Position ion number ",i," (\[Mu]m): (",MinPosMicron[[i,1]],",",MinPosMicron[[i,2]],",",MinPosMicron[[i,3]],")"];
	];
	
mass=ions[[All,1]];

	(* Matrix with all the second order derivatives at the minimum position *)
	PotMat=Table[e/Sqrt[mass[[IntegerPart[(i+2)/3]]]mass[[IntegerPart[(j+2)/3]]]]*
		D[PotAllIonsMeter,xiTableFlat[[i]],xiTableFlat[[j]]]/.MinPosMeterRuleRule,
		{i,1,3*NIons},
		{j,1,3*NIons}
	];
	
	(* work around the bug in Eigensystem[] *)
	PotMatNew=PotMat;
	For[i=1,i<=3,i++,
		For[j=1,j<=3,j++,
			For[k=1,k<=NIons,k++,
				For[l=1,l<=NIons,l++,
					PotMatNew[[ (k-1)*3+i, (l-1)*3+j ]]=PotMat[[ (k-1)*3+Mod[i+1,3,1], (l-1)*3+Mod[j+1,3,1] ]];
				];
			];
		];
	];
	PotMat=PotMatNew;
	
	(* Find normal modes *)
	es = Eigensystem[PotMat];
	\[Omega]modes = Sqrt[es[[1]]];

	(* work around the bug in Eigensystem[] *)
	evs=es[[2]];
	For[i=1,i<=3*NIons,i++,
		For[j=1,j<=3,j++,
			For[l=1,l<=NIons,l++,
				evs[[ i, (l-1)*3+j ]]=es[[ 2, i, (l-1)*3+Mod[j-1,3,1] ]];
			];
		];
	];

	Print["Mode frequencies (MHz): ",Table[\[Omega]modes[[i]]/(2\[Pi] MHz),{i,1,3*NIons}],"\n"];
	
	Print["Normal mode vectors:\n",MatrixForm[SetPrecision[Chop[evs,10^-3],3]]];
	
	Return[
		{
			MinPosMeter,
			\[Omega]modes,
			evs
		}
	];

];


AnalyzeTrap[pot_,x_,y_,z_,ions_]:= Module[
	{MinPosMicron,MinPosMeter,MinPosMicronRuleRule,MinPosMeterRuleRule,\[Omega]modes,es,NIons,
	PotAllIonsMeter,PotAllIonsMicron,PotAllIons,PotMat,
	i,j,
	xiTable,xiTableFlat,xi,
	InitialConditions,LengthUnit,mass},

	(* How many ions? *)
	NIons=Length[ions];

	(* Table with cartesian coordinates for each ion *)
	xiTable=Table[xi[i,j],{i,1,NIons},{j,1,3}];

	(* Flat table *)
	xiTableFlat=Flatten[Table[xi[i,j],{i,1,NIons},{j,1,3}],1];

	(* Initial positions for the ions *)
	InitialConditions=Flatten[
		Table[
			{xi[i,j],ions[[i,2,j]]/\[Mu]m},
			{i,1,NIons},
			{j,1,3}
		],1
	];

	(* Potential for all ions together, length unit still open *)
	PotAllIons=Sum[
		(* External potential at the position of each ion *)
		(pot/.{
			x->xi[i,1] LengthUnit,
			y->xi[i,2] LengthUnit,
			z->xi[i,3] LengthUnit
		}) +
		(* Coulomb repulsion between ions *)
		Sum[e/(4\[Pi] \[Epsilon]0)/Sqrt[(xiTable[[i]] LengthUnit - xiTable[[j]] LengthUnit).(xiTable[[i]] LengthUnit - xiTable[[j]] LengthUnit)],{j,1,i-1}],
		{i,1,NIons}
	];
	
	(* Lenth unit microns *)
	PotAllIonsMicron=PotAllIons/.{LengthUnit->\[Mu]m};

	(* Length unit meters *)
	PotAllIonsMeter=PotAllIons/.{LengthUnit->1};

	(* Find the ion positions that minimize the total energy *)
	MinPosMicronRuleRule = FindMinimum[
		PotAllIonsMicron,
		InitialConditions
	][[2]];

	(* Minimum positions in other units *)
	MinPosMicron=xiTable/.MinPosMicronRuleRule;
	MinPosMeter=MinPosMicron*\[Mu]m;
	MinPosMeterRuleRule=Thread[Rule[xiTableFlat,Flatten[MinPosMeter]]];
	For[i=1,i<=NIons,i++,
		Print["Position ion number ",i," [\[Mu]m]: ",MinPosMicron[[i]]];
	];

	(* TODO For now, let's assume for now that all ions have the same mass. *)
	mass=ions[[1,1]];
	
	(* Matrix with all the second order derivatives at the minimum position *)
	PotMat=e/mass*Table[
		D[PotAllIonsMeter,xiTableFlat[[i]],xiTableFlat[[j]]]/.MinPosMeterRuleRule,
		{i,1,3*NIons},
		{j,1,3*NIons}
	];

	(* Find normal modes *)
	es = Eigensystem[PotMat];
	\[Omega]modes = Sqrt[es[[1]]];
	Print["Mode frequencies [MHz]: ",\[Omega]modes/(2\[Pi] MHz)];
	Print["Normal mode vectors:    ",TableForm[Chop[es[[2]]]]];

	Return[
		{
			MinPosMeter,
			\[Omega]modes,
			es[[2]]
		}
	];
];


SidebandFit[{dataRSB_,dataBSB_},MWTime_,{\[Pi]tgroS_,nbarS_,centerS_,IntensS_,offsetS_}]:=Module[
{fitfunB,fitfunR,chisquare,fitres,p1,p2,\[Pi]tgro,nbar,center,Intens,offset,limitLow,limitHigh,TextStartX,TextStartY},{
Off[FindMinimum::"lstol"];
fitfunB[{\[Pi]tgro_,nbar_,center_,Intens_,offset_},x_]:=Abs[Intens]*ThermalBSB[MWTime \[Mu]s,nbar,2\[Pi] (x-center)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),15]+Abs[offset];
fitfunR[{\[Pi]tgro_,nbar_,center_,Intens_,offset_},x_]:=Abs[Intens]*ThermalRSB[MWTime \[Mu]s,nbar,2\[Pi] (x-center)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),15]+Abs[offset];

chisquare[{\[Pi]tgro_,nbar_,center_,Intens_,offset_}]:=1/(Length[dataRSB]+Length[dataBSB]-6)*
(Sum[((fitfunR[{\[Pi]tgro,nbar,center,Intens,offset},dataRSB[[k,1]]]-dataRSB[[k,2]])/dataRSB[[k,3]])^2,{k,1,Length[dataRSB]}]
+Sum[((fitfunB[{\[Pi]tgro,nbar,center,Intens,offset},dataBSB[[k,1]]]-dataBSB[[k,2]])/dataBSB[[k,3]])^2,{k,1,Length[dataBSB]}]);

fitres=FindMinimum[chisquare[{\[Pi]tgro,nbar,center,Intens,offset}],{{\[Pi]tgro,\[Pi]tgroS},{nbar,nbarS},{center,centerS},{Intens,IntensS},{offset,offsetS}}];

limitLow=dataBSB[[1,1]]-0.1*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
limitHigh=dataBSB[[Length[dataBSB],1]]+0.1*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartX=dataBSB[[Length[dataBSB],1]]-0.3*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartY=Max[dataBSB[[All,2]]]/3;

p1=Show[
ErrorListPlot[dataRSB,PlotStyle->Directive[PointSize[Large],Red],PlotRange->All],
ErrorListPlot[dataBSB,PlotStyle->Directive[PointSize[Large],Blue],PlotRange->All]
];

p2=Plot[{
fitfunB[{\[Pi]tgro,nbar,center,Intens,offset},x]/.fitres[[2]],fitfunR[{\[Pi]tgro,nbar,center,Intens,offset},x]/.fitres[[2]]
},
{x,limitLow,limitHigh},
PlotRange->{All,{0,Max[dataBSB[[All,2]]]+1.5}},
FrameLabel->{"\!\(\*SubscriptBox[\"\[Nu]\", \"MW\"]\)-\!\(\*SubscriptBox[\"\[Nu]\", 
RowBox[{\"field\", \"-\", \"indep\"}]]\) (MHz)","fluo.fluo./400 (\!\(\*SuperscriptBox[\"\[Mu]s\", 
RowBox[{\"-\", \"1\"}]]\))"},
PlotStyle->{Directive[Blue,Thick,Opacity[.6]],Directive[Red,Thick,Opacity[.6]]},
Epilog->{
Text[Style["nbar = "<>ToString[SetPrecision[nbar/.fitres[[2]],2],TraditionalForm],Small],{TextStartX,TextStartY},{-1,0}],
Text[Style["\[Pi]-time(nbar=0)/\[Mu]s = "<>ToString[SetPrecision[\[Pi]tgro/.fitres[[2]],4],TraditionalForm],Small],{TextStartX,TextStartY-.5},{-1,0}](*,
Text[Style["\!\(\*SubscriptBox[\"\[Nu]\", \"SB\"]\)/MHz = "<>ToString[SetPrecision[center/.fitres[[2]],4],TraditionalForm],Small],{TextStartX,TextStartY-1.},{-1,0}],
Text[Style["dark count offset = "<>ToString[SetPrecision[offset/.fitres[[2]],3],TraditionalForm],Small],{TextStartX,TextStartY-1.5},{-1,0}],
Text[Style["red. \!\(\*SuperscriptBox[\"\[Chi]\", \"2\"]\) = "<>ToString[SetPrecision[fitres[[1]],2],TraditionalForm],Small],{TextStartX,TextStartY-2.},{-1,0}]
*)}];

Show[p2,p1,ImageSize->{600,400}]
}];


SidebandFitNEW[{dataRSB_,dataBSB_},MWTime_,{\[Pi]tgroS_,nbarS_,centerS_,IntensS_,offsetS_}]:=Module[
{fitfunB,fitfunR,chisquare,fitres,p1,p2,\[Pi]tgro,nbar,center,Intens,offset,limitLow,limitHigh,TextStartX,TextStartY},{
Off[FindMinimum::"lstol"];
fitfunB[{\[Pi]tgro_,nbar_,center_,Intens_,offset_},x_]:=Intens*(1-ThermalBSB[MWTime \[Mu]s,nbar,2\[Pi] (x-center)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),20])+Abs[offset];
fitfunR[{\[Pi]tgro_,nbar_,center_,Intens_,offset_},x_]:=Intens*(1-ThermalRSB[MWTime \[Mu]s,nbar,2\[Pi] (x-center)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),20])+Abs[offset];

chisquare[{\[Pi]tgro_,nbar_,center_,Intens_,offset_}]:=1/(Length[dataRSB]+Length[dataBSB]-6)*
(Sum[((fitfunR[{\[Pi]tgro,nbar,center,Intens,offset},dataRSB[[k,1]]]-dataRSB[[k,2]])/dataRSB[[k,3]])^2,{k,1,Length[dataRSB]}]
+Sum[((fitfunB[{\[Pi]tgro,nbar,center,Intens,offset},dataBSB[[k,1]]]-dataBSB[[k,2]])/dataBSB[[k,3]])^2,{k,1,Length[dataBSB]}]);

fitres=FindMinimum[chisquare[{\[Pi]tgro,nbar,center,IntensS,offsetS}],{{\[Pi]tgro,\[Pi]tgroS},{center,centerS},{nbar,nbarS}}];

limitLow=dataBSB[[1,1]]-0.1*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
limitHigh=dataBSB[[Length[dataBSB],1]]+0.1*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartX=dataBSB[[Length[dataBSB],1]]-0.3*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartY=2*Max[dataBSB[[All,2]]]/3;

p1=Show[
ErrorListPlot[dataRSB,PlotStyle->Directive[PointSize[Large],Red],PlotRange->All],
ErrorListPlot[dataBSB,PlotStyle->Directive[PointSize[Large],Blue],PlotRange->All]
];

p2=Plot[{
fitfunR[{\[Pi]tgro,nbar,center,IntensS,offsetS},x]/.fitres[[2]],
fitfunB[{\[Pi]tgro,nbar,center,IntensS,offsetS},x]/.fitres[[2]]
},
{x,limitLow,limitHigh},
PlotRange->{All,{0,Max[dataBSB[[All,2]]]+1.5}},
FrameLabel->{"\!\(\*SubscriptBox[\"\[Nu]\", \"MW\"]\)-\!\(\*SubscriptBox[\"\[Nu]\", 
RowBox[{\"field\", \"-\", \"indep\"}]]\) (MHz)","fluo./400 (\!\(\*SuperscriptBox[\"\[Mu]s\", 
RowBox[{\"-\", \"1\"}]]\))"},
PlotStyle->{Directive[Blue,Thick,Opacity[.6]],Directive[Red,Thick,Opacity[.6]]}
];
Print[ToString[fitres]];
Show[p2,p1,ImageSize->{600,400}]
}];


SidebandFlopFit[{dataRSB_,dataBSB_},det_,{\[Pi]tgroS_,nbarS_,IntensS_,offsetS_}]:=Module[
{fitfunB,fitfunR,chisquare,fitres,p1,p2,\[Pi]tgro,nbar,center,Intens,offset,limitLow,limitHigh,TextStartX,TextStartY},{
Off[FindMinimum::"lstol"];
fitfunB[{\[Pi]tgro_,nbar_,Intens_,offset_},x_]:=Intens*ThermalBSB[x \[Mu]s,nbar,2\[Pi] (det)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),15]+offset;
fitfunR[{\[Pi]tgro_,nbar_,Intens_,offset_},x_]:=Intens*ThermalRSB[x \[Mu]s,nbar,2\[Pi] (det)MHz,\[Pi]/(2 \[Pi]tgro \[Mu]s),15]+offset;

chisquare[{\[Pi]tgro_,nbar_,Intens_,offset_}]:=1/(Length[dataRSB]+Length[dataBSB]-6)*
(Sum[((fitfunR[{\[Pi]tgro,nbar,Intens,offset},dataRSB[[k,1]]]-dataRSB[[k,2]])/dataRSB[[k,3]])^2,{k,1,Length[dataRSB]}]
+Sum[((fitfunB[{\[Pi]tgro,nbar,Intens,offset},dataBSB[[k,1]]]-dataBSB[[k,2]])/dataBSB[[k,3]])^2,{k,1,Length[dataBSB]}]);

fitres=FindMinimum[chisquare[{\[Pi]tgro,nbar,Intens,offset}],{{\[Pi]tgro,\[Pi]tgroS},{nbar,nbarS},{Intens,IntensS},{offset,offsetS}}];

limitLow=0.;
limitHigh=dataBSB[[Length[dataBSB],1]]+0.1*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartX=dataBSB[[Length[dataBSB],1]]-0.3*(dataBSB[[Length[dataBSB],1]]-dataBSB[[1,1]]);
TextStartY=Max[dataBSB[[All,2]]]/3;

p1=Show[
ErrorListPlot[dataRSB,PlotStyle->Directive[PointSize[Large],Red],PlotRange->All],
ErrorListPlot[dataBSB,PlotStyle->Directive[PointSize[Large],Blue],PlotRange->All]
];

p2=Plot[{
fitfunB[{\[Pi]tgro,nbar,Intens,offset},x]/.fitres[[2]],fitfunR[{\[Pi]tgro,nbar,Intens,offset},x]/.fitres[[2]]
},
{x,limitLow,limitHigh},
PlotRange->{All,{0,Max[dataBSB[[All,2]]]+1.5}},
PlotStyle->{Directive[Blue,Thick,Opacity[.6]],Directive[Red,Thick,Opacity[.6]]},
FrameLabel->{"MW time (\[Mu]s)","fluo. (arb. units)"},
Epilog->{
Text[Style["nbar = "<>ToString[nbar/.fitres[[2]]],Small],{TextStartX,TextStartY},{-1,0}],
Text[Style["\[Pi]-time(nbar=0)/\[Mu]s = "<>ToString[\[Pi]tgro/.fitres[[2]]],Small],{TextStartX,TextStartY-.5},{-1,0}],
Text[Style["dark count offset = "<>ToString[offset/.fitres[[2]]],Small],{TextStartX,TextStartY-1.},{-1,0}],
Text[Style["red. \!\(\*SuperscriptBox[\"\[Chi]\", \"2\"]\) = "<>ToString[fitres[[1]]],Small],{TextStartX,TextStartY-1.5},{-1,0}]
}];

Show[p2,p1,ImageSize->{600,400}]

}];


(* ::Subsection::Closed:: *)
(*some stuff for randomized benchmarking*)


AnalysisBenchmark[{filename_, seqLengths_}]:=Module[
	{dataBench,avedark,avebright,rescaleddata,goaldata,LPa,LPb,AveSeqLength,
	AveSeqLengthError,fitfun,Errors,ErrorWeights,OneSigma,err,x,Result,ResultErr},
		{dataBench=Drop[Import[filename,"Table"],0];
		avedark=Sum[dataBench[[i,4]]/Length[dataBench],{i,1,Length[dataBench]}];
		avebright=Sum[dataBench[[i,5]]/Length[dataBench],{i,1,Length[dataBench]}];

		rescaleddata=Table[{i,(dataBench[[i,3]]-dataBench[[i,4]])/(dataBench[[i,5]]-dataBench[[i,4]])},{i,1,Length[dataBench]}];
		goaldata=Table[{i+.5,1-dataBench[[i,6]]},{i,1,Length[dataBench]}];

		LPa=ListPlot[goaldata,FrameLabel->{"exp. #"," result"}];
		LPb=ListPlot[rescaleddata,FrameLabel->{"exp. #"," result"},Joined->True,InterpolationOrder->0,PlotStyle->{RGBColor[1,0,0]},PlotRange->{All,All}];
		Show[LPb,LPa],

		fidelData=Table[{i,
			If [goaldata[[i,2]]==0,1-(goaldata[[i,2]]+rescaleddata[[i,2]]),1-(goaldata[[i,2]]-rescaleddata[[i,2]])]},{i,1,Length[dataBench]}];
		ListLogPlot[fidelData,FrameLabel->{"exp #","fidelity"},PlotRange->{All,All},Joined->False],

		fidelData=Table[{dataBench[[i,2]],
			If [goaldata[[i,2]]==0,1-(goaldata[[i,2]]+rescaleddata[[i,2]]),1-(goaldata[[i,2]]-rescaleddata[[i,2]])]},{i,1,Length[dataBench]}];
		
		SeqLength = seqLengths;
		
		For[i=1,i<Length[SeqLength]+1,{
			Subscript[SeqLength, i]=Select[fidelData,Total[#]>seqLengths[[i]]&&Total[#]<seqLengths[[i]]+2&];
			Subscript[AveSeqLength, i]=Mean[Subscript[SeqLength, i]][[2]];
			Subscript[AveSeqLengthError, i]=StandardDeviation[Subscript[SeqLength, i]][[2]];
			SeqLength[[i]]={{seqLengths[[i]],Subscript[AveSeqLength, i]},ErrorBar[Subscript[AveSeqLengthError, i]]};
		i++}];
		
		"\nResults: \n"<>ToString[Grid[Table[{SeqLength[[i,1,1]], SeqLength[[i,1,2]], SeqLength[[i,2,1]]},{i,1, Length[SeqLength]}]]]<>"\n",
		fitfun[err_,x_]=Exp[-err*x];
		Errors=SeqLength[[All,2,1]];
		ErrorWeights=Min[Errors]/Errors;
		Ffitres=NonlinearModelFit[SeqLength[[All,1]],fitfun[err,x],{err},x,Weights->ErrorWeights];
		OneSigma=Ffitres["MeanPredictionBands",ConfidenceLevel->.99];
		Show[		
			ListPlot[fidelData,PlotStyle->GrayLevel[.75],FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}],
			ErrorListPlot[SeqLength,FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}, PlotStyle->{RGBColor[0,0,0]}],
			Plot[{OneSigma,fitfun[err,x]/.Ffitres[[1,2]]},{x,0,60},Filling->{1->{2}}, PlotStyle->{RGBColor[1,0,0],RGBColor[0,0,1],RGBColor[0,0,0]}]
		],
		Result = SetAccuracy[err/.Ffitres["BestFitParameters"], 6];
		ResultErr = SetAccuracy[Ffitres["ParameterConfidenceIntervalTable"][[1,1,2,3]], 6];
		"one-qubit error probability: "<>ToString[Result]<>" +/- "<>ToString[ResultErr]<>"\n"
	}]


AnalysisBenchmarkNEW[{fideldata_, SeqLengths_}]:=Module[
	{dataBench,avedark,avebright,rescaleddata,goaldata,LPa,LPb,AveSeqLength,
	AveSeqLengthError,fitfun,Errors,ErrorWeights,OneSigma,err,x,Result,ResultErr,dif, d},
		{SeqLength = SeqLengths;
		For[i=1,i<Length[SeqLength]+1,{
			Subscript[SeqLength, i]=Select[fidelData,Total[#]>SeqLength[[i]]&&Total[#]<SeqLength[[i]]+2&];
			Subscript[AveSeqLength, i]=Mean[Subscript[SeqLength, i]][[2]];
			Subscript[AveSeqLengthError, i]=StandardDeviation[Subscript[SeqLength, i]][[2]];
			SeqLength[[i]]={{SeqLengths[[i]],Subscript[AveSeqLength, i]},ErrorBar[Subscript[AveSeqLengthError, i]]};
		i++}];

		"\nResults: \n"<>ToString[Grid[Table[{SeqLength[[i,1,1]], SeqLength[[i,1,2]], SeqLength[[i,2,1]]},{i,1, Length[SeqLength]}]]]<>"\n",

	fitfun[dif_,d_,x_]:=1/2+1/2(1-dif) (1-d)^x;
	Errors=SeqLength[[All,2,1]];
	ErrorWeights=Min[Errors]/Errors;
	Ffitres=NonlinearModelFit[SeqLength[[All,1]],fitfun[dif,d,x],{{dif,0.08},{d,0.001}},x,Weights->ErrorWeights];
	OneSigma=Ffitres["MeanPredictionBands",ConfidenceLevel->.99];
	Show[
		ListPlot[fidelData,PlotStyle->GrayLevel[.75],FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}],
		ErrorListPlot[SeqLength,FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}, PlotStyle->{RGBColor[0,0,0]}],Plot[{OneSigma,fitfun[dif,d,x]/.Ffitres[[1,2]]},{x,0,60},Filling->{1->{2}}, PlotStyle->{RGBColor[1,0,0],RGBColor[0,0,1],RGBColor[0,0,0]}]
	],
	Result = SetAccuracy[dif/.Ffitres["BestFitParameters"], 6];
	ResultErr = SetAccuracy[Ffitres["ParameterConfidenceIntervalTable"][[1,1,2,3]], 6];
	"probability of depolarization due to initialization and read-out: "<>ToString[Result]<>" +/- "<>ToString[ResultErr],

	Result = SetAccuracy[d/.Ffitres["BestFitParameters"], 6];
	ResultErr = SetAccuracy[Ffitres["ParameterConfidenceIntervalTable"][[1,1,3,3]], 6];
	"average probability of depolarization of a single randomized computational gate: "<>ToString[Result]<>" +/- "<>ToString[ResultErr]<>"\n"
	}]


AnalysisBenchmarkWHisto[{fideldata_, SeqLengths_}]:=Module[
	{dataBench,avedark,avebright,rescaleddata,goaldata,LPa,LPb,AveSeqLength,
	AveSeqLengthError,fitfun,Errors,ErrorWeights,OneSigma,err,x,Result,ResultErr,dif, d},
	{SeqLength = SeqLengths;
	For[i=1,i<Length[SeqLength]+1,{
		Subscript[SeqLength, i]=Select[fidelData,Total[#]>SeqLength[[i]]&&Total[#]<SeqLength[[i]]+2&];
		Subscript[AveSeqLength, i]=Mean[Subscript[SeqLength, i]][[2]];
		Subscript[AveSeqLengthError, i]=StandardDeviation[Subscript[SeqLength, i]][[2]];
		SeqLength[[i]]={{SeqLengths[[i]],Subscript[AveSeqLength, i]},ErrorBar[Subscript[AveSeqLengthError, i]]};
	i++}];

	"\nResults: \n"<>ToString[Grid[Table[{SeqLength[[i,1,1]], SeqLength[[i,1,2]], SeqLength[[i,2,1]]},{i,1, Length[SeqLength]}]]]<>"\n",

	fitfun[dif_,d_,x_]:=1/2+1/2(1-dif) (1-d)^x;
	Errors=SeqLength[[All,2,1]];
	ErrorWeights=Min[Errors]/Errors;
	Ffitres=NonlinearModelFit[SeqLength[[All,1]],fitfun[dif,d,x],{{dif,0.08},{d,0.001}},x,Weights->ErrorWeights];
	OneSigma=Ffitres["MeanPredictionBands",ConfidenceLevel->.99];
	Show[
	ListPlot[fidelData,PlotStyle->GrayLevel[.75],FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}],
	ErrorListPlot[SeqLength,FrameLabel->{"# of gates","fidelity"},PlotRange->{All,All}, PlotStyle->{RGBColor[0,0,0]}],Plot[{OneSigma,fitfun[dif,d,x]/.Ffitres[[1,2]]},{x,0,60},Filling->{1->{2}}, PlotStyle->{RGBColor[1,0,0],RGBColor[0,0,1],RGBColor[0,0,0]}]
	],
	Result = SetAccuracy[dif/.Ffitres["BestFitParameters"], 6];
	ResultErr = SetAccuracy[Ffitres["ParameterConfidenceIntervalTable"][[1,1,2,3]], 6];
	"probability of depolarization due to initialization and read-out: "<>ToString[Result]<>" +/- "<>ToString[ResultErr],

	Result = SetAccuracy[d/.Ffitres["BestFitParameters"], 6];
	ResultErr = SetAccuracy[Ffitres["ParameterConfidenceIntervalTable"][[1,1,3,3]], 6];
	"average probability of depolarization of a single randomized computational gate: "<>ToString[Result]<>" +/- "<>ToString[ResultErr]<>"\n"
}]


(* ::Subsection::Closed:: *)
(*here some options for plots are set*)


SetOptions[Plot, PlotRange->All,PlotStyle->Directive[Thick],Axes->False,Frame->True,BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[BarChart, PlotRange->All, Axes->False,Frame->True,BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[LogPlot, PlotRange->All, Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[LogLogPlot, PlotRange->All, Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ErrorListPlot, PlotRange->All, PlotMarkers->Automatic, PlotStyle->Directive[PointSize[Large]],Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ListPlot, PlotRange->All, PlotMarkers->Automatic,PlotStyle->Directive[PointSize[Large]],Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ListLogPlot, PlotRange->All, PlotStyle->Directive[PointSize[Large]],Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ParametricPlot, PlotRange->All, Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ParametricPlot3D, PlotRange->All, Axes->False, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ListLogLogPlot, PlotRange->All, PlotStyle->Directive[PointSize[Large]],Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];
SetOptions[ListLinePlot, PlotRange->All, PlotStyle->Directive[Thick],Axes->False,Frame->True, BaseStyle->{FontFamily->"Arial",FontWeight->"Bold",FontSize->18},ImageSize->{500, Automatic}];



(* ::Section::Closed:: *)
(*epilog*)


EndPackage[]
