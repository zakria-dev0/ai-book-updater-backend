"""
omml_to_latex.py
Converts OMML (Office Math Markup Language) XML to LaTeX.

OMML is the math format used by Microsoft Word (.docx).
Each <m:oMath> element is parsed recursively and mapped to LaTeX commands.
"""

from __future__ import annotations
from lxml import etree
import re

# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _m(local: str) -> str:
    """Return Clark-notation tag for OMML math namespace."""
    return f"{{{MATH_NS}}}{local}"


def _w(local: str) -> str:
    """Return Clark-notation tag for WordprocessingML namespace."""
    return f"{{{W_NS}}}{local}"


def _get_attr(el, local_name: str, default: str = "") -> str:
    """Fetch an attribute that may be namespaced (m:val) or bare (val)."""
    val = el.get(_m(local_name))
    if val is None:
        val = el.get(local_name, default)
    return val if val is not None else default


def _local(tag: str) -> str:
    """Strip namespace from a Clark-notation tag string."""
    return tag.split("}")[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# Unicode → LaTeX symbol maps
# ---------------------------------------------------------------------------

_SPECIAL: dict[str, str] = {
    # Greek lowercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # Greek uppercase
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # Arrows
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↑": r"\uparrow", "↓": r"\downarrow",
    # Relations
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "≅": r"\cong", "∼": r"\sim", "≪": r"\ll", "≫": r"\gg",
    "∝": r"\propto", "⊥": r"\perp", "∥": r"\parallel",
    # Sets
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊃": r"\supset",
    "⊆": r"\subseteq", "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap",
    "∅": r"\emptyset", "∀": r"\forall", "∃": r"\exists",
    # Operators
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "·": r"\cdot", "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    # Calculus / analysis
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    # Misc
    "√": r"\sqrt", "…": r"\ldots", "⋮": r"\vdots", "⋱": r"\ddots",
    "°": r"^{\circ}", "′": r"'", "″": r"''",
    "ℝ": r"\mathbb{R}", "ℤ": r"\mathbb{Z}", "ℕ": r"\mathbb{N}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}",
    # LaTeX special chars that need escaping in text
    "#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&",
    "_": r"\_", "{": r"\{", "}": r"\}",
}

_NARY_OP: dict[str, str] = {
    "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod",
    "⋃": r"\bigcup", "⋂": r"\bigcap",
    "⋁": r"\bigvee", "⋀": r"\bigwedge",
    "⨄": r"\biguplus", "⨁": r"\bigoplus", "⨂": r"\bigotimes",
    "⨀": r"\bigodot", "⨆": r"\bigsqcup",
}

_ACCENT_MAP: dict[str, str] = {
    "\u0302": r"\hat",    # combining circumflex (̂)
    "\u0303": r"\tilde",  # combining tilde (̃)
    "\u0304": r"\bar",    # combining macron (̄)
    "\u0307": r"\dot",    # combining dot above (̇)
    "\u0308": r"\ddot",   # combining diaeresis (̈)
    "\u0306": r"\breve",  # combining breve (̆)
    "\u030c": r"\check",  # combining caron (̌)
    "\u20d7": r"\vec",    # combining right arrow above (⃗)
    "̂": r"\hat", "̃": r"\tilde", "̄": r"\bar",
    "̇": r"\dot", "̈": r"\ddot", "⃗": r"\vec",
}

_BEG_DELIM: dict[str, str] = {
    "(": r"\left(", "[": r"\left[", "{": r"\left\{",
    "|": r"\left|", "‖": r"\left\|", "⌈": r"\left\lceil",
    "⌊": r"\left\lfloor", "〈": r"\left\langle", "": r"\left.",
}
_END_DELIM: dict[str, str] = {
    ")": r"\right)", "]": r"\right]", "}": r"\right\}",
    "|": r"\right|", "‖": r"\right\|", "⌉": r"\right\rceil",
    "⌋": r"\right\rfloor", "〉": r"\right\rangle", "": r"\right.",
}


# ---------------------------------------------------------------------------
# Core conversion dispatcher
# ---------------------------------------------------------------------------

def _convert(el) -> str:
    """Dispatch an lxml element to its specific converter."""
    tag = _local(el.tag)

    dispatch = {
        "oMath":     _conv_omath,
        "oMathPara": _conv_children,
        "r":         _conv_run,
        "t":         _conv_text,
        "f":         _conv_fraction,
        "sSup":      _conv_ssup,
        "sSub":      _conv_ssub,
        "sSubSup":   _conv_ssubsup,
        "sqrt":      _conv_sqrt,
        "rad":       _conv_rad,
        "nary":      _conv_nary,
        "func":      _conv_func,
        "d":         _conv_delimiter,
        "m":         _conv_matrix,
        "acc":       _conv_accent,
        "bar":       _conv_bar,
        "limLow":    _conv_limlower,
        "limUpp":    _conv_limupper,
        "eqArr":     _conv_eqarr,
        "groupChr":  _conv_groupchr,
        "borderBox": _conv_borderbox,
        "box":       _conv_box,
        "phant":     _conv_phant,
    }

    # Property / metadata elements — produce no LaTeX output
    _SKIP_TAGS = {
        "rPr", "fPr", "sPr", "naryPr", "dPr", "mPr", "accPr", "barPr",
        "groupChrPr", "pPr", "oMathParaPr", "radPr", "eqArrPr", "limLowPr",
        "limUppPr", "funcPr", "boxPr", "phantPr", "ctrlPr",
        "mcs", "mc", "mcPr", "count",
    }
    if tag in _SKIP_TAGS:
        return ""

    fn = dispatch.get(tag)
    if fn:
        return fn(el)
    # Unknown element — recurse into children
    return _conv_children(el)


def _conv_children(el) -> str:
    return "".join(_convert(child) for child in el)


def _conv_omath(el) -> str:
    return _conv_children(el)


# ---------------------------------------------------------------------------
# Run / text
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Map Unicode math characters to LaTeX equivalents."""
    result = []
    for ch in text:
        result.append(_SPECIAL.get(ch, ch))
    return "".join(result)


def _conv_text(el) -> str:
    return _escape(el.text or "")


def _conv_run(el) -> str:
    """<m:r> — may carry style info in <m:rPr>."""
    rpr = el.find(_m("rPr"))
    is_normal_text = False
    is_bold = False

    if rpr is not None:
        # <m:nor/> means normal (non-math) text
        if rpr.find(_m("nor")) is not None:
            is_normal_text = True
        sty = rpr.find(_m("sty"))
        if sty is not None:
            sty_val = _get_attr(sty, "val", "")
            if sty_val in ("b", "bi"):
                is_bold = True
            elif sty_val == "p":
                is_normal_text = True

    t_el = el.find(_m("t"))
    if t_el is None or not t_el.text:
        return ""
    text = _escape(t_el.text)

    if is_normal_text:
        return r"\text{" + t_el.text + "}"
    if is_bold:
        return r"\mathbf{" + text + "}"
    return text


# ---------------------------------------------------------------------------
# Fraction
# ---------------------------------------------------------------------------

def _conv_fraction(el) -> str:
    """<m:f> — detect linear/skewed fractions via <m:fPr><m:type>."""
    fpr = el.find(_m("fPr"))
    ftype = ""
    if fpr is not None:
        type_el = fpr.find(_m("type"))
        if type_el is not None:
            ftype = _get_attr(type_el, "val", "")

    num = _conv_children(el.find(_m("num"))) if el.find(_m("num")) is not None else ""
    den = _conv_children(el.find(_m("den"))) if el.find(_m("den")) is not None else ""

    if ftype == "lin":
        return f"{num}/{den}"
    if ftype == "skw":
        return "{}^{" + num + "}/_{" + den + "}"
    if ftype == "noBar":
        return r"\binom{" + num + "}{" + den + "}"
    return r"\frac{" + num + "}{" + den + "}"


# ---------------------------------------------------------------------------
# Super / subscript
# ---------------------------------------------------------------------------

def _conv_ssup(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    sup = _conv_children(el.find(_m("sup"))) if el.find(_m("sup")) is not None else ""
    return "{" + e + "}^{" + sup + "}"


def _conv_ssub(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    sub = _conv_children(el.find(_m("sub"))) if el.find(_m("sub")) is not None else ""
    return "{" + e + "}_{" + sub + "}"


def _conv_ssubsup(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    sub = _conv_children(el.find(_m("sub"))) if el.find(_m("sub")) is not None else ""
    sup = _conv_children(el.find(_m("sup"))) if el.find(_m("sup")) is not None else ""
    return "{" + e + "}_{" + sub + "}^{" + sup + "}"


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def _conv_sqrt(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    return r"\sqrt{" + e + "}"


def _conv_rad(el) -> str:
    deg_el = el.find(_m("deg"))
    e_el = el.find(_m("e"))
    e = _conv_children(e_el) if e_el is not None else ""

    # Check if degree is hidden
    rad_pr = el.find(_m("radPr"))
    if rad_pr is not None:
        deg_hide = rad_pr.find(_m("degHide"))
        if deg_hide is not None:
            val = _get_attr(deg_hide, "val", "1")
            if val not in ("0", "false"):
                return r"\sqrt{" + e + "}"

    degree = _conv_children(deg_el) if deg_el is not None else ""
    if not degree.strip():
        return r"\sqrt{" + e + "}"
    return r"\sqrt[" + degree + "]{" + e + "}"


# ---------------------------------------------------------------------------
# N-ary operators (integrals, sums, products)
# ---------------------------------------------------------------------------

def _conv_nary(el) -> str:
    nary_pr = el.find(_m("naryPr"))
    sub_el = el.find(_m("sub"))
    sup_el = el.find(_m("sup"))
    e_el = el.find(_m("e"))

    op = r"\int"  # default operator
    sub_hide = False
    sup_hide = False

    if nary_pr is not None:
        chr_el = nary_pr.find(_m("chr"))
        if chr_el is not None:
            char_val = _get_attr(chr_el, "val", "∫")
            op = _NARY_OP.get(char_val, r"\int")
        # Check hide flags
        sh = nary_pr.find(_m("subHide"))
        if sh is not None and _get_attr(sh, "val", "0") not in ("0", "false"):
            sub_hide = True
        sph = nary_pr.find(_m("supHide"))
        if sph is not None and _get_attr(sph, "val", "0") not in ("0", "false"):
            sup_hide = True

    result = op
    if not sub_hide and sub_el is not None:
        s = _conv_children(sub_el)
        if s:
            result += "_{" + s + "}"
    if not sup_hide and sup_el is not None:
        s = _conv_children(sup_el)
        if s:
            result += "^{" + s + "}"
    if e_el is not None:
        result += " " + _conv_children(e_el)
    return result


# ---------------------------------------------------------------------------
# Function application
# ---------------------------------------------------------------------------

def _conv_func(el) -> str:
    fname_el = el.find(_m("fName"))
    e_el = el.find(_m("e"))
    fname = _conv_children(fname_el) if fname_el is not None else ""
    arg = _conv_children(e_el) if e_el is not None else ""
    # If fname is a known trig/log function, wrap arg in parens
    _KNOWN_FUNCS = {
        r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
        r"\arcsin", r"\arccos", r"\arctan",
        r"\log", r"\ln", r"\exp", r"\lim", r"\max", r"\min",
        r"\det", r"\ker", r"\dim", r"\deg",
    }
    if fname.strip() in _KNOWN_FUNCS:
        return fname + r"\left(" + arg + r"\right)"
    return fname + "{" + arg + "}"


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------

def _conv_delimiter(el) -> str:
    dpr = el.find(_m("dPr"))
    beg_chr = "("
    end_chr = ")"

    if dpr is not None:
        beg_el = dpr.find(_m("begChr"))
        end_el = dpr.find(_m("endChr"))
        if beg_el is not None:
            beg_chr = _get_attr(beg_el, "val", "(")
        if end_el is not None:
            end_chr = _get_attr(end_el, "val", ")")

    beg_latex = _BEG_DELIM.get(beg_chr, r"\left" + beg_chr)
    end_latex = _END_DELIM.get(end_chr, r"\right" + end_chr)

    contents = [_conv_children(e) for e in el.findall(_m("e"))]
    # Multiple <m:e> in a delimiter means separated by separator char
    sep_el = dpr.find(_m("sepChr")) if dpr is not None else None
    sep_char = _get_attr(sep_el, "val", "|") if sep_el is not None else "|"
    sep_latex = r" \middle" + sep_char + " " if len(contents) > 1 else ""

    inner = sep_latex.join(contents)
    return beg_latex + " " + inner + " " + end_latex


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

def _conv_matrix(el) -> str:
    """<m:m> matrix — detect matrix type from column alignment."""
    mpr = el.find(_m("mPr"))
    env = "pmatrix"

    if mpr is not None:
        # mcs/mc/mcPr/mJc can tell us alignment but we default to pmatrix
        pass

    rows = []
    for mr_el in el.findall(_m("mr")):
        cells = [_conv_children(e) for e in mr_el.findall(_m("e"))]
        rows.append(" & ".join(cells))
    body = r" \\ ".join(rows)
    return r"\begin{" + env + "}" + body + r"\end{" + env + "}"


# ---------------------------------------------------------------------------
# Accent
# ---------------------------------------------------------------------------

def _conv_accent(el) -> str:
    acc_pr = el.find(_m("accPr"))
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""

    accent_char = "\u0302"  # default: combining hat
    if acc_pr is not None:
        chr_el = acc_pr.find(_m("chr"))
        if chr_el is not None:
            accent_char = _get_attr(chr_el, "val", "\u0302")

    cmd = _ACCENT_MAP.get(accent_char, r"\hat")
    return cmd + "{" + inner + "}"


# ---------------------------------------------------------------------------
# Overline / underline
# ---------------------------------------------------------------------------

def _conv_bar(el) -> str:
    bar_pr = el.find(_m("barPr"))
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""

    pos = "top"
    if bar_pr is not None:
        pos_el = bar_pr.find(_m("pos"))
        if pos_el is not None:
            pos = _get_attr(pos_el, "val", "top")

    return r"\underline{" + inner + "}" if pos == "bot" else r"\overline{" + inner + "}"


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def _conv_limlower(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    lim = _conv_children(el.find(_m("lim"))) if el.find(_m("lim")) is not None else ""
    return r"\underset{" + lim + "}{" + e + "}"


def _conv_limupper(el) -> str:
    e = _conv_children(el.find(_m("e"))) if el.find(_m("e")) is not None else ""
    lim = _conv_children(el.find(_m("lim"))) if el.find(_m("lim")) is not None else ""
    return r"\overset{" + lim + "}{" + e + "}"


# ---------------------------------------------------------------------------
# Equation array (aligned)
# ---------------------------------------------------------------------------

def _conv_eqarr(el) -> str:
    rows = [_conv_children(e) for e in el.findall(_m("e"))]
    body = r" \\ ".join(rows)
    return r"\begin{aligned}" + body + r"\end{aligned}"


# ---------------------------------------------------------------------------
# Group character (overbrace / underbrace)
# ---------------------------------------------------------------------------

def _conv_groupchr(el) -> str:
    gchr_pr = el.find(_m("groupChrPr"))
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""

    pos = "top"
    if gchr_pr is not None:
        pos_el = gchr_pr.find(_m("pos"))
        if pos_el is not None:
            pos = _get_attr(pos_el, "val", "top")

    return r"\underbrace{" + inner + "}" if pos == "bot" else r"\overbrace{" + inner + "}"


# ---------------------------------------------------------------------------
# Box / border box / phantom
# ---------------------------------------------------------------------------

def _conv_borderbox(el) -> str:
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""
    return r"\boxed{" + inner + "}"


def _conv_box(el) -> str:
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""
    return r"\boxed{" + inner + "}"


def _conv_phant(el) -> str:
    """Phantom — renders invisible; use \phantom{}."""
    e_el = el.find(_m("e"))
    inner = _conv_children(e_el) if e_el is not None else ""
    return r"\phantom{" + inner + "}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def omml_to_latex(omml_xml: str) -> str:
    """
    Convert an OMML XML string (content of <m:oMath>…</m:oMath>) to LaTeX.

    Returns an empty string if conversion fails or input is empty.
    """
    if not omml_xml or not omml_xml.strip():
        return ""
    try:
        root = etree.fromstring(omml_xml.encode("utf-8"))
        latex = _convert(root)
        # Normalize whitespace
        latex = re.sub(r" {2,}", " ", latex).strip()
        return latex
    except Exception:
        return ""
