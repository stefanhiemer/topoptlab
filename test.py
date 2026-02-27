import sympy as sp

def _mul_gcd_general(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """Multiplicative 'gcd' for general expressions via power dictionaries."""
    a = sp.sympify(a)
    b = sp.sympify(b)

    # Split numeric coefficient and symbolic part
    ca, aa = a.as_coeff_Mul()
    cb, bb = b.as_coeff_Mul()

    # Numeric gcd for integer/rational coefficients; otherwise keep 1
    try:
        c = sp.gcd(ca, cb)
    except Exception:
        c = sp.Integer(1)

    pa = aa.as_powers_dict()
    pb = bb.as_powers_dict()

    common = sp.Integer(1)
    for base in set(pa) & set(pb):
        ea = pa[base]
        eb = pb[base]
        # take the "min" exponent when comparable (works for integers/rationals)
        try:
            e = min(ea, eb)
        except TypeError:
            # if exponents aren't comparable, require exact match
            e = ea if ea == eb else None

        if e is None:
            continue
        if e != 0:
            common *= base**e

    return sp.simplify(c * common)


def split_independent(expr: sp.Expr, forbidden_vars):
    """Return (independent_factor, remainder) where independent_factor has none of forbidden_vars."""
    expr = sp.factor_terms(sp.sympify(expr))
    indep, rem = expr.as_independent(*forbidden_vars, as_Add=False)
    # guard against 0/None
    if indep is None or indep == 0 or (getattr(indep, "is_zero", None) is True):
        indep = sp.Integer(1)
    return indep, rem


def common_factor_excluding_vars(expr1: sp.Expr, expr2: sp.Expr, forbidden_vars):
    """
    Find a common multiplicative factor shared by expr1 and expr2
    that does NOT contain forbidden_vars.
    Returns (common_factor, new1, new2) with expri = common_factor * newi.
    """
    f1, r1 = split_independent(expr1, forbidden_vars)
    f2, r2 = split_independent(expr2, forbidden_vars)

    common = _mul_gcd_general(f1, f2)
    if common == 0:
        common = sp.Integer(1)

    new1 = sp.simplify((f1/common) * r1)
    new2 = sp.simplify((f2/common) * r2)
    return common, new1, new2