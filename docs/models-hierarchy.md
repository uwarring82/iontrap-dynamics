# Hierarchy of models and realisations: trapped ion, quantum Rabi, spin–boson, JC / AJC

!!! note "Provenance — vendored companion note (WP-03 Dispatch RLD)"
    **Source.** Content-faithfully vendored from `ajc-provenance/docs/hierarchy.md` **v0.4** (lock
    candidate — "content review converged; external referee rounds 1–3 and Scout; the §4/§5
    supporting references are verified; awaiting Harbourmaster lock decision").

    **Source commit / DOI.** _Pending lock_ — to be recorded here once the upstream note is
    locked. This vendored copy tracks the v0.4 lock-candidate text. WP-03 R7 named this
    field as a precondition, so R7 stays **open** (upstream-provenance-pending) until it lands.

    **Licence.** **CC BY-SA 4.0** (Coastline), preserved from the source — distinct from the
    Sail tutorials (CC BY-NC-SA 4.0) and the code (MIT). Stewardship: U. Warring
    (Harbourmaster), AG Schätz.

    **Encoding.** The upstream Markdown reached this repository with a UTF-8/Latin-1
    transit corruption; the prose punctuation (section signs, en/em dashes, ×, −, →) has
    been repaired and the embedded GitHub-flavoured LaTeX preserved verbatim. Three source-repo
    relative links (`../LICENCE`, `../cases/bibliography.bib`, `../task-card/locked-wording.md`)
    are rendered as plain-text references, since their targets live in `ajc-provenance`, not here.

    **Rendering.** Mathematics uses GitHub-flavoured LaTeX. This docs site does not load a
    MathJax/KaTeX extension, so `$…$` expressions display literally (subscripts and underscores
    shown raw) — the note is designed to degrade gracefully this way; it typesets on GitHub.

    **Cited by.** `CONVENTIONS.md` §25 and Tutorial 18 (_Reduced models vs full dynamics_)
    reference this note by section.

> Endorsement Marker · T(h)reehouse +EC — Local candidate framework (no parity implied with externally validated laws).
> Stewardship: U. Warring (Harbourmaster). Licence: CC BY-SA 4.0.
> Status: lock candidate (v0.4). Content review converged — external referee rounds 1–3 and Scout; the §4/§5 supporting references are now verified (§11). Awaiting Harbourmaster lock decision. Versioning in git.
> Rendering: mathematics uses GitHub-flavoured LaTeX (`$…$`, `$$…$$`); it renders on GitHub and in any MathJax/KaTeX viewer. Plain markdown previewers without math support show subscripts and underscores literally.

## External constraints (citation-only)

This note relies on externally validated physics, referenced as constraints and never re-derived here:

- the trapped-ion light–matter interaction and its sideband structure — Leibfried, Blatt, Monroe & Wineland, *Rev. Mod. Phys.* 75, 281 (2003);
- the Jaynes–Cummings model — Jaynes & Cummings, *Proc. IEEE* 51, 89 (1963);
- integrability of the quantum Rabi model — Braak, *Phys. Rev. Lett.* 107, 100401 (2011).

Full entries in `cases/bibliography.bib` (source repository).

## Novel boundaries (local, versioned)

The document's own revisable commitments — falsifiable by counterexample, not by external authority, and carrying no parity with the constraints above:

- the two-axis separation (containment vs realisation) as the move that resolves the JC/AJC "label" question;
- the R0–R3 rung assignment and the validity conditions collected in §10;
- the claim that what Q4 calls "loss" is captured, for the Lamb–Dicke truncation specifically, by the Debye–Waller × Laguerre renormalisation of the sideband coupling (§5). Other rungs of the descent from R(−1) drop other things (off-resonant carrier terms, further modes, micromotion, leakage levels, Stark shifts, laser noise); this claim is about the Lamb–Dicke step only.

---

## 1. Purpose

This note fixes the mathematical relationships between four objects — the full trapped-ion Hamiltonian, the quantum Rabi model (QRM), the spin–boson model, and the Jaynes–Cummings / anti-Jaynes–Cummings (JC / AJC) pair — and the hierarchy that connects them. It is the spine for two case questions: Q3 (where "only a label" holds for JC vs AJC) and Q4 (what is lost when the approximated sideband Hamiltonian is treated as primary). Both are statements about *position in this hierarchy*, so the hierarchy is made explicit first and the cases cite it by rung.

The single most important point is that there are two hierarchies, not one, and conflating them is the error the project guards against:

- Axis A — *model containment*: which Hamiltonian is a special case of (or reducible to) which, as abstract operators, independent of any hardware.
- Axis B — *physical realisation*: which experimental operation on a real ion turns the full ion Hamiltonian into each model.

"Is X a special case of Y?" can have different answers on the two axes. Keeping them apart is exactly what lets Q3 state cleanly *where* the JC–AJC relabelling is innocent and where it is not.

## 2. Conventions

- $\hbar = 1$.
- Qubit: bare splitting on $\sigma_z$, coupling on $\sigma_x$ (equivalently $\sigma_\pm = \tfrac{1}{2}(\sigma_x \pm i\sigma_y)$).
- Three frequencies are kept distinct (see the laboratory engineering note in §5): the physical atomic splitting $\omega_{\mathrm{at}} > 0$; the laser detuning $\delta = \omega_L - \omega_{\mathrm{at}}$ (either sign); and the effective/model qubit frequency $\omega_0$ (written $\omega_0^{\mathrm{eff}}$ in a simulator), whose sign is a frame/basis convention. In the abstract models of §3, $\omega_0$ is this effective frequency, not the physical splitting.
- "RWA" means *drop the counter-rotating terms* — the terms left fast-oscillating after moving to the relevant interaction picture.
- Two distinct rotating-wave approximations appear and must not be blurred: the *optical* RWA (drops terms at the optical sum frequency $\omega_{\mathrm{at}}+\omega_L$) and the *vibrational / resolved-sideband* RWA (drops terms oscillating at the trap frequency $\nu$). In the ion, JC arises from the second of these, after the Lamb–Dicke expansion — not from the optical one.
- Pauli identities used below: $\sigma_x\sigma_z\sigma_x = -\sigma_z$ and $\sigma_x\sigma_\pm\sigma_x = \sigma_\mp$.

## 3. The objects (and an implicit fifth)

R(−1) · the real ion. Multilevel internal structure, several motional modes, a *quantised* field, micromotion. Named here and not formalised, so that "full trapped-ion Hamiltonian" is never read as fundamental. Everything below is already an idealisation of this.

R0 · full trapped-ion Hamiltonian (one ion, one mode, two internal levels, classical drive), lab frame, with $\omega_{\mathrm{at}} > 0$ the physical atomic splitting:

$$H = \frac{\omega_{\mathrm{at}}}{2}\sigma_z + \nu a^\dagger a + \Omega \sigma_x \cos\big(\eta(a+a^\dagger) - \omega_L t + \phi\big).$$

"Full" means full *within* the two-level / single-mode / classical-field idealisation. The coupling is nonlinear in $(a+a^\dagger)$ through the exponential hidden in the cosine; this nonlinearity is the whole subject of Q4.

Quantum Rabi model:

$$H_{\mathrm{QRM}} = \frac{\omega_0}{2}\sigma_z + \omega_f a^\dagger a + g \sigma_x(a+a^\dagger).$$

Transverse coupling, no RWA. Symmetry: a discrete $\mathbb{Z}_2$ parity $\Pi = \sigma_z (-1)^{a^\dagger a}$ (so $a \to -a$, $\sigma_x \to -\sigma_x$ leave $H_{\mathrm{QRM}}$ invariant). Excitation number is not conserved. Integrable (Braak 2011).

Spin–boson model:

$$H_{\mathrm{SB}} = \frac{\epsilon}{2}\sigma_z + \frac{\Delta}{2}\sigma_x + \sigma_z\sum_k g_k(a_k + a_k^\dagger) + \sum_k \omega_k a_k^\dagger a_k,$$

one spin coupled to *many* modes, characterised by a spectral density $J(\omega) = \pi\sum_k g_k^2 \delta(\omega-\omega_k)$. The single-mode, unbiased ($\epsilon=0$) case is unitarily equivalent to the quantum Rabi model after a spin-basis rotation (a $\pi/2$ rotation about $y$ maps $\sigma_z \leftrightarrow \sigma_x$), up to absorbable signs — the rotation gives $\omega_0 = -\Delta$, the sign reabsorbed by convention. A non-zero bias $\epsilon$ maps to the *asymmetric* (biased) QRM. The genuine content beyond Rabi is the multimode bath, not the single-mode coupling.

JC and AJC:

$$H_{\mathrm{JC}} = \frac{\omega_0}{2}\sigma_z + \omega_f a^\dagger a + g(a\sigma_+ + a^\dagger\sigma_-), \qquad H_{\mathrm{AJC}} = \frac{\omega_0}{2}\sigma_z + \omega_f a^\dagger a + g(a^\dagger\sigma_+ + a\sigma_-).$$

Each has a continuous $U(1)$ symmetry: JC conserves $N = a^\dagger a + \sigma_+\sigma_-$; AJC conserves $a^\dagger a - \sigma_+\sigma_-$. Both are exactly solvable in $2\times 2$ blocks (dressed states). Neither is a special case of the other.

## 4. Axis A — model containment

As abstract operators:

$$\text{spin–boson } (N \text{ modes}) \xrightarrow{\text{ one mode; unbiased + spin rotation }} \text{QRM} \xrightarrow{\text{ RWA (two complementary halves) }} \{H_{\mathrm{JC}}, H_{\mathrm{AJC}}\}.$$

The first arrow is a *unitary equivalence* in the single-mode unbiased case, not a literal term-by-term containment in the displayed basis: the standard spin–boson coupling is longitudinal ($\sigma_z$) and the QRM coupling transverse ($\sigma_x$), related by the spin rotation of §3. The biased single-mode case maps to the asymmetric QRM; the multimode case is the genuine generalisation.

The non-obvious content of this axis is what the RWA does to the *symmetry*:

| Model | Symmetry | Conserved quantity | Solvability |
|-------|----------|--------------------|-------------|
| Spin–boson (multimode) | none generic ($\mathbb{Z}_2$ if unbiased single-mode) | — | generally hard |
| Quantum Rabi | $\mathbb{Z}_2$ parity $\Pi=\sigma_z(-1)^{a^\dagger a}$ | parity (discrete) | integrable |
| JC | $U(1)$ | $N=a^\dagger a+\sigma_+\sigma_-$ | block-diagonal $2\times2$ |
| AJC | $U(1)$ | $a^\dagger a-\sigma_+\sigma_-$ | block-diagonal $2\times2$ |

In the QRM → JC/AJC reduction the approximation is visible algebraically as a symmetry enhancement: the exact QRM has only the discrete $\mathbb{Z}_2$ parity, whereas each rotating-wave half acquires a continuous $U(1)$ (a conserved excitation-like number), of which the parity is the subgroup $e^{i\pi N}$. This enhancement is the clean signature that an approximation has been made; it is exact in neither coupling strength nor detuning beyond the regime that justifies dropping the counter-rotating half. (Braak's integrability of the QRM rests on that $\mathbb{Z}_2$; JC and AJC are easier precisely because one coupling half has been removed.)

JC and AJC are the two complementary RWAs of the same parent: JC keeps $(a\sigma_+ + a^\dagger\sigma_-)$, AJC keeps $(a^\dagger\sigma_+ + a\sigma_-)$. Neither is "the" RWA; they are a pair selected by which half survives.

## 5. Axis B — the trapped-ion rung ladder

Each step is one named operation with one cost. This is the realisation ladder, and it is the *mirror image* of the cavity-QED intuition (see §7).

R0 · two-level + single-mode truncation — the $H$ of §3. Cost: other internal levels and motional modes assumed off-resonant; field treated classically.

R1 · optical RWA + interaction picture (with respect to $H_0=\tfrac{\omega_{\mathrm{at}}}{2}\sigma_z+\nu a^\dagger a$) —

$$H_I = \frac{\Omega}{2}\sigma_+ \exp\big(i\eta(a e^{-i\nu t} + a^\dagger e^{i\nu t})\big) e^{-i\delta t} + \text{h.c.}, \qquad \delta \equiv \omega_L - \omega_{\mathrm{at}}.$$

The optically counter-rotating terms (at $\omega_{\mathrm{at}}+\omega_L$) are gone, but the spin-dependent displacement $e^{i\eta(a+a^\dagger)}$ — the full sideband nonlinearity — is intact. Structurally this is a driven spin–motion coupling, not yet Rabi and not yet JC; it becomes spin–boson-like only once many modes or many tones are engineered into an effective bath.

R2 · Lamb–Dicke expansion ($\eta\sqrt{\langle(a+a^\dagger)^2\rangle}\ll 1$), first order in $\eta$ — three resonances,

$$\underbrace{\frac{\Omega}{2}\sigma_+ e^{-i\delta t}}_{\text{carrier, } \delta=0} + \underbrace{\frac{\Omega}{2} i\eta \sigma_+ a e^{-i(\delta+\nu)t}}_{\text{red sideband, } \delta=-\nu} + \underbrace{\frac{\Omega}{2} i\eta \sigma_+ a^\dagger e^{-i(\delta-\nu)t}}_{\text{blue sideband, } \delta=+\nu} + \mathrm{h.c.}$$

This is the step that linearises the coupling, and so is the first point at which "Rabi" becomes available. Everything dropped here is the content of Q4.

R3 · drive choice + vibrational (resolved-sideband) RWA — the effective model is *selected by the detuning*. In every row the resolved-sideband selection is applied (carrier and far-off-resonant terms dropped); the rows differ in which sideband terms are retained:

| Drive | Resolved-sideband selection | Terms retained | Effective model | Coupling |
|-------|-----------------------------|----------------|-----------------|----------|
| monochromatic, $\delta=-\nu$ (red) | yes (drop carrier + blue) | red-sideband term only | JC | $g=\eta\Omega/2$ |
| monochromatic, $\delta=+\nu$ (blue) | yes (drop carrier + red) | blue-sideband term only | AJC | $g=\eta\Omega/2$ |
| bichromatic (red + blue) | yes (drop carrier + far-off-resonant) | both sideband terms | quantum Rabi (*simulated*) | $g=\eta\Omega/2$; $\omega_0^{\mathrm{eff}},\omega_f^{\mathrm{eff}}$ from the two detunings |
| many tones / many modes | yes | many sideband terms | spin–boson (engineered $J(\omega)$) | per-tone |

The QRM row is a *simulated* QRM: resolved-sideband selection is still applied, but both sideband terms are retained and no further RWA is applied to the simulated qubit–field coupling. JC and AJC differ from it precisely by retaining only one of the two sideband terms — equivalently, by applying that further (simulated-QRM) RWA. This is the route to ultrastrong / deep-strong coupling on a single ion (Pedernales et al. 2015, proposal; Lv et al. 2018, experiment). Many tones and modes engineer a phonon bath, i.e. spin–boson (Porras et al. 2008; Lemmer et al. 2018). The full ion sits above all of them because it still carries the higher sidebands and the nonlinear corrections that R2 discards.

### Laboratory engineering note: physical splitting, detuning, and effective negative frequency

The symbol $\omega_0$ has so far played two roles that must not be silently identified. In the real ion the internal transition is a positive physical splitting, $\omega_{\mathrm{at}} > 0$; the abstract $\omega_0$ of §3, and the effective frequency reached in a simulator, are something else. Three quantities, only one of which is the physical splitting:

| Quantity | Meaning | Can it be negative? |
|----------|---------|---------------------|
| physical atomic splitting $\omega_{\mathrm{at}}$ | energy gap between the two real ion levels | no — positive by convention |
| laser detuning $\delta = \omega_L - \omega_{\mathrm{at}}$ | experimental knob selecting carrier / red / blue sideband | yes |
| effective qubit frequency $\omega_0^{\mathrm{eff}}$ | rotating-frame parameter of the engineered (simulated) model | yes — sign is frame/basis dependent |

Trapped-ion interaction engineering turns the lab frame into an effective model through four knobs:

- laser detuning — selects carrier, red sideband, blue sideband, or bichromatic combinations;
- laser phase — sets the spin quadrature ($\sigma_x$ vs $\sigma_y$);
- Rabi frequency — sets $g = \eta\Omega/2$ in the Lamb–Dicke sideband approximation;
- rotating-frame residual detunings — set the effective $\omega_0^{\mathrm{eff}}$ and $\omega_f^{\mathrm{eff}}$.

For a bichromatic drive, write the two laser detunings as

$$\delta_r = -\nu + \Delta_r, \qquad \delta_b = +\nu + \Delta_b.$$

After the Lamb–Dicke expansion and sideband selection, the retained interaction is, schematically,

$$H_I \simeq g\big(\sigma_+ a e^{-i\Delta_r t} + \sigma_+ a^\dagger e^{-i\Delta_b t} + \mathrm{h.c.}\big), \qquad g = \eta\Omega/2.$$

A further rotating-frame transformation makes this time-independent; in that final frame the effective qubit frequency is the sum, and the effective oscillator frequency the difference, of the two residual detunings $\Delta_r$ and $\Delta_b$ (Pedernales et al. 2015). So $\omega_0^{\mathrm{eff}}$ can be tuned through zero and made effectively negative, while the physical splitting $\omega_{\mathrm{at}}$ stays positive throughout.

The negative frequency in the algebraic relation $H_{\mathrm{AJC}}(\omega_0) = \sigma_x H_{\mathrm{JC}}(-\omega_0)\sigma_x$ (§6) is therefore never a physical inversion of the atomic transition. It arises in one of two ways: a basis relabelling — exchanging $|\downarrow\rangle$ and $|\uparrow\rangle$ sends $\sigma_z \to -\sigma_z$ and $\sigma_\pm \to \sigma_\mp$, describing the same physical system as if the spin term had changed sign — or a rotating-frame effective detuning, as above.

A nuance worth stating, because it is a real trap: preparing $|\uparrow\rangle$ instead of $|\downarrow\rangle$ does not by itself change the Hamiltonian. Preparing the opposite spin state can realise the partner dynamics when combined with the matching basis/readout convention, but the Hamiltonian sign change itself is a relabelling or rotating-frame effect, not a physical inversion of the energy splitting.

The trichotomy to carry into Q3: red versus blue sideband selection is a physical detuning knob; the JC – AJC algebraic equivalence is a basis relation; and a negative spin frequency is an effective rotating-frame or relabelling convention — not a negative bare transition. Confusing the first with the third is the main source of the "only a label" ambiguity.

### The nonlinear branch (not a footnote — Q4 lives here)

Without the Lamb–Dicke truncation, the blue-sideband coupling on $|g,n\rangle \to |e,n{+}1\rangle$ has Rabi frequency

$$\Omega_{n,n+1} = \Omega e^{-\eta^2/2} \eta \frac{L_n^{(1)}(\eta^2)}{\sqrt{n+1}},$$

with $L_n^{(1)}$ an associated Laguerre polynomial, $e^{-\eta^2/2}$ the Debye–Waller factor, and $\Omega$ the carrier Rabi frequency. In the Lamb–Dicke limit $L_n^{(1)}(\eta^2)\to n+1$ and $e^{-\eta^2/2}\to 1$, so

$$\Omega_{n,n+1} \to \Omega \eta \sqrt{n+1} = 2g\sqrt{n+1} \quad (g=\eta\Omega/2),$$

i.e. the corresponding coupling matrix element reduces to $g\sqrt{n+1}$ — the $\sqrt{n+1}$ Fock-state dependence of the bare AJC term (the sideband Rabi frequency being twice this). The Debye–Waller suppression and the full Laguerre dependence are real and $n$-dependent: they deform the collapse/revival comb built on the reduced picture. The beyond-Lamb–Dicke nonlinear sideband dynamics is documented experimentally and theoretically (Cheng et al. 2018).

Within the Lamb–Dicke regime these are *systematic, computable corrections* to R3 — higher orders of the same $\eta$-expansion, recoverable perturbatively — not a different framework. They cease to be perturbative when the regime itself fails: $\eta^2(2n+1) \gtrsim 1$ (large $\eta$, or high Fock $n$ even at small $\eta$), where the exact Laguerre form is required and no finite $\eta$-truncation suffices.

So the R3 Hamiltonian is not wrong; it is a controlled local chart. The error is to forget that its constant coupling $g$ is the Lamb–Dicke shadow of an $n$-dependent matrix element.

## 6. JC – AJC, precisely

The relation, stated four ways at four levels:

- Abstract (Axis A): the two complementary RWAs of the quantum Rabi model.
- Algebraic (LOCK-3 — a locked-wording item; see the locked-wording task card in the source repository): $H_{\mathrm{AJC}}(\omega_0) = \sigma_x H_{\mathrm{JC}}(-\omega_0)\sigma_x$. Here $\omega_0$ is the effective (model / rotating-frame) qubit frequency, not the physical splitting $\omega_{\mathrm{at}}$; negating it is a basis relabel or a rotating-frame sign, never an inversion of $\omega_{\mathrm{at}}$ (see the laboratory engineering note in §5). AJC is JC with the effective qubit frequency negated and the spin relabelled, using $\sigma_x\sigma_z\sigma_x=-\sigma_z$ and $\sigma_x\sigma_\pm\sigma_x=\sigma_\mp$.
- Operational (Axis B, ion): the red–blue switch — the *sign of the detuning* $\delta=\mp\nu$. This is the sense in which "only a label" is closest to true.
- Interpretive (Sail, not locked): the negated-frequency partner is what touches the Dirac-oscillator picture (Bermudez 2007; handbook §9.4) — the antiparticle sector maps to the AJC-like term. A sentence, not a claim.

## 7. The boson is not the same boson

A standing caution for Q1/Q2, to be stated rather than smuggled: in cavity QED the boson is the photon and JC follows directly from the *optical* RWA; in the trapped ion the boson is the phonon and JC follows from the *sideband* RWA, after the $\eta$-expansion (R2). Same algebra, different physical origin, different small parameter. This is precisely why the etymological question — did the trapped-ion community coin "anti-JC"? — is not settled by the algebra alone.

## 8. The three-regime boundary (LOCK-2) as positions in the hierarchy

LOCK-2 is the locked-wording item that scopes *when* "only a label" holds (see the locked-wording task card in the source repository). It falls straight out of the ladder:

| Regime | Position | Status of "only a label" |
|--------|----------|--------------------------|
| Isolated interaction-frame Hamiltonian, studied in its own right | the abstract JC or AJC operator (post-R3, taken alone) | true — JC and AJC are unitarily equivalent via §6 |
| A single selected sideband on a real ion | one row of the R3 table | a physical knob — red vs blue is a detuning choice with distinct dynamics |
| Full quantum Rabi (abstract, or the simulated QRM at R3 bichromatic) | both coupling halves present | false — the two halves are non-commuting and physically inequivalent |

## 9. What this fixes for Q3 and Q4

- Q3 (isomorphism boundary) is the one-paragraph statement of the table in §8, anchored on the §6 identity and the trichotomy in the laboratory engineering note (§5): knob vs basis-relation vs rotating-frame convention. It can now cite rungs: equivalence holds at the level of the *isolated* post-R3 operator; it fails once R3 is read as a selection on R2 (a knob) or undone toward the QRM.
- Q4 (approximation status) is the R2 gap: the error licensed by treating an R3 effective Hamiltonian as primary is the loss of the nonlinear branch in §5 — the Debye–Waller / Laguerre renormalisation is the worked counterexample, and the beyond-Lamb–Dicke / beyond-RWA literature (Cheng et al. 2018) is the evidence that the discarded structure is physical, not formal. The corrections are perturbative within the Lamb–Dicke regime and non-perturbative once $\eta^2(2n+1)\gtrsim 1$.

## 10. Collected validity conditions

| Arrow | Operation | What is dropped | Validity condition | Symmetry change |
|-------|-----------|-----------------|--------------------|-----------------|
| R(−1) → R0 | two-level + single-mode truncation | other levels/modes; field quantisation | spectral isolation; classical, intense field | — |
| R0 → R1 | optical RWA + interaction picture | terms at $\omega_{\mathrm{at}}+\omega_L$ | $\Omega, \delta, \eta\Omega \ll \omega_{\mathrm{at}},\omega_L$ | — |
| R1 → R2 | Lamb–Dicke expansion (1st order) | $O(\eta^2)$ and higher sidebands | $\eta\sqrt{\langle(a+a^\dagger)^2\rangle}\ll 1$, i.e. $\eta^2(2n+1)\ll1$ | coupling linearised |
| R2 → R3 (one tone) | resolved-sideband selection + retain one sideband | carrier, other sideband, terms at $\nu,2\nu,\dots$ | resolved sidebands: $\eta\Omega \ll \nu$; $\delta=\mp\nu$ | $\mathbb{Z}_2 \to U(1)$ |
| R2 → R3 (two tones) | resolved-sideband selection only (no further RWA) | carrier + far-off-resonant | resolved sidebands | $\mathbb{Z}_2$ retained — simulated QRM |
| QRM → JC/AJC | RWA on the (simulated) qubit–field coupling — retain one sideband, not both | one coupling half | $g \ll \omega_0,\omega_f$, near-resonance | $\mathbb{Z}_2 \leftrightarrow U(1)$ (in this reduction) |
| single-mode SB → QRM | spin rotation $\sigma_x\leftrightarrow\sigma_z$ | — (exact; bias $\epsilon$ → asymmetric QRM) | exact, up to absorbable signs | basis change only |

## 11. Supporting references

The references behind §4 and §5 are verified and recorded in `cases/bibliography.bib` (source repository):

- backbone — Jaynes–Cummings 1963; Leibfried et al. 2003; Bermudez et al. 2007; Lamata et al. 2007; Solano–Agarwal–Walther 2003; Bocanegra-Garay et al. 2024;
- quantum Rabi integrability — Braak 2011 (*Phys. Rev. Lett.* 107, 100401);
- trapped-ion QRM via bichromatic sideband driving, including ultrastrong/deep-strong regimes — Pedernales et al. 2015 (*Sci. Rep.* 5, 15472, proposal) and Lv et al. 2018 (*Phys. Rev. X* 8, 021027, experiment);
- beyond-Lamb–Dicke nonlinear sideband dynamics (Q4 evidence) — Cheng et al. 2018 (*Phys. Rev. A* 97, 023624);
- trapped-ion spin–boson / engineered baths — Porras et al. 2008 (*Phys. Rev. A* 78, 010101) and Lemmer et al. 2018 (*New J. Phys.* 20, 073002).

One item remains open and belongs to the Q1/Q2 etymology track, not to this hierarchy: pinning the foundational ion-trap review that introduced the paired $H_{\mathrm{JC}\pm}$ / $H_{\mathrm{AJC}\pm}$ notation. Report "earliest located", never "first ever".
