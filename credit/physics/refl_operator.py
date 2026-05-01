import torch


# Constants (align with diag_nssl_refl.py)
PI = 3.141592653589793
RWDN = 1000.0
ALPHAR = 0.0
ALPHAH = 0.0
ALPHAHL = 1.0
SNU = -0.8
DBZMIN = 0.0


def _g1(alpha: float) -> float:
    return ((6 + alpha) * (5 + alpha) * (4 + alpha)) / ((3 + alpha) * (2 + alpha) * (1 + alpha))


def _safe_to_float(x):
    return float(x) if isinstance(x, (int, float)) else x


def ze_rain(qr: torch.Tensor, nr: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    g1 = _g1(ALPHAR)
    coeff = 1.0e18 * (6.0 / (PI * RWDN)) ** 2
    qr = qr.double(); nr = nr.double(); rho = rho.double()
    ze = torch.zeros_like(qr)
    mask = (qr >= 1.0e-5) & (nr > 1.0e-3)
    if mask.any():
        zx = g1 * (rho[mask] * qr[mask]) ** 2 / nr[mask]
        ze[mask] = coeff * zx
    return ze.float()


def ze_snow(qs: torch.Tensor, ns: torch.Tensor, rho: torch.Tensor, temp: torch.Tensor, qr: torch.Tensor | None = None) -> torch.Tensor:
    # Constants
    TFR = 273.16
    KSQ = 0.189
    qs = qs.double(); ns = ns.double(); rho = rho.double(); temp = temp.double()
    if qr is None:
        qr = torch.zeros_like(qs)
    else:
        qr = qr.double()
    ze = torch.zeros_like(qs)
    mask = (qs >= 1.0e-6) & (ns > 1.0e-7)
    if not mask.any():
        return ze.float()

    qs_m = qs[mask]; ns_m = ns[mask]; rho_m = rho[mask]; temp_m = temp[mask]; qr_m = qr[mask]
    bb = (temp_m > (TFR + 1.0)) & (qs_m > qr_m) & (qr_m > 1.0e-6)
    qxw = torch.zeros_like(qs_m)
    qxw[bb] = torch.minimum(0.5 * qs_m[bb], qr_m[bb])

    # Old (wet snow / bright band)
    ze_old = torch.zeros_like(qs_m)
    use_old = qxw > 1.0e-6
    if use_old.any():
        qs_wet = qs_m[use_old] + qxw[use_old]
        ze_old[use_old] = (
            3.6e18
            * (SNU + 2.0)
            * (0.224 * qs_wet + 0.776 * qxw[use_old])
            * qs_wet
            / (ns_m[use_old] * (SNU + 1.0) * RWDN**2)
            * rho_m[use_old] ** 2
        )

    # Cox (1988) dry snow formulation
    # Precompute gamma factors (constants): Gamma(0.2) and Gamma(1.5333...) folded into coefficient below
    # Use numeric constants from numpy reference: 323.3226 * 0.106214**2 / 917^2 / (1+SNU)^(4/3)
    coef = 1.0e18 * 323.3226 * (0.106214 ** 2) * KSQ / (917.0 ** 2) / ((1.0 + SNU) ** (4.0 / 3.0))
    ze_new = torch.zeros_like(qs_m)
    use_new = ~use_old
    if use_new.any():
        ze_new[use_new] = coef * qs_m[use_new] ** 2 * rho_m[use_new] ** 2 / ns_m[use_new]

    ze_out = torch.zeros_like(qs)
    ze_out[mask] = ze_old + ze_new
    return ze_out.float()


def _clamp_mean_volume_and_adjust_N(q: torch.Tensor, n: torch.Tensor, rho: torch.Tensor, dens: torch.Tensor, vmin: float, vmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    # xV = rho * q / (dens * max(n, small))
    small = 1.0e-9
    xV = rho * q / (dens * torch.clamp(n, min=small))
    need = (xV < vmin) | (xV > vmax)
    xV = torch.clamp(xV, min=vmin, max=vmax)
    n_eff = n.clone()
    if need.any():
        n_eff[need] = rho[need] * q[need] / (xV[need] * dens[need])
    return xV, n_eff


def ze_graupel(qg: torch.Tensor, ng: torch.Tensor, rho: torch.Tensor, volg: torch.Tensor | None = None) -> torch.Tensor:
    g1 = _g1(ALPHAH)
    coeff = 1.0e18 * (6.0 / (PI * RWDN)) ** 2
    qg = qg.double(); ng = ng.double(); rho = rho.double()
    ze = torch.zeros_like(qg)
    mask = (qg >= 1.0e-5) & (ng >= 1.0e-8)
    if not mask.any():
        return ze.float()
    qg_m = qg[mask]; ng_m = ng[mask]; rho_m = rho[mask]
    if volg is not None:
        volg = volg.double()
        volg_m = volg[mask]
        hwdn = torch.full_like(qg_m, 500.0)
        valid = volg_m > 0
        hwdn[valid] = rho_m[valid] * qg_m[valid] / volg_m[valid]
        hwdn = torch.clamp(hwdn, min=100.0, max=900.0)
    else:
        hwdn = torch.full_like(qg_m, 500.0)
    # Clamp mean volume and adjust N
    XVHMN = 0.523599 * (0.3e-3) ** 3
    XVHMX = 0.523599 * (20.0e-3) ** 3
    _, n_eff = _clamp_mean_volume_and_adjust_N(qg_m, ng_m, rho_m, hwdn, XVHMN, XVHMX)
    zx = g1 * rho_m ** 2 * 0.224 * qg_m * qg_m / n_eff
    out = torch.zeros_like(qg)
    out[mask] = coeff * zx
    return out.float()


def ze_hail(qh: torch.Tensor, nh: torch.Tensor, rho: torch.Tensor, volh: torch.Tensor | None = None) -> torch.Tensor:
    g1 = _g1(ALPHAHL)
    coeff = 1.0e18 * (6.0 / (PI * RWDN)) ** 2
    qh = qh.double(); nh = nh.double(); rho = rho.double()
    ze = torch.zeros_like(qh)
    mask = (qh >= 1.0e-5) & (nh > 0)
    if not mask.any():
        return ze.float()
    qh_m = qh[mask]; nh_m = nh[mask]; rho_m = rho[mask]
    if volh is not None:
        volh = volh.double()
        volh_m = volh[mask]
        hldn = torch.full_like(qh_m, 900.0)
        valid = volh_m > 0
        hldn[valid] = rho_m[valid] * qh_m[valid] / volh_m[valid]
        hldn = torch.clamp(hldn, min=300.0, max=900.0)
    else:
        hldn = torch.full_like(qh_m, 900.0)
    XVHLMN = 0.523599 * (0.3e-3) ** 3
    XVHLMX = 0.523599 * (40.0e-3) ** 3
    _, n_eff = _clamp_mean_volume_and_adjust_N(qh_m, nh_m, rho_m, hldn, XVHLMN, XVHLMX)
    zx = g1 * rho_m ** 2 * 0.224 * qh_m * qh_m / n_eff
    out = torch.zeros_like(qh)
    out[mask] = coeff * zx
    return out.float()


def ze_ice(qi: torch.Tensor, ni: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    qi = qi.double(); ni = ni.double(); rho = rho.double()
    ze = torch.zeros_like(qi)
    mask = (qi > 1.0e-9) & (ni > 1.0)
    if mask.any():
        vr = rho[mask] * qi[mask] / (900.0 * ni[mask])
        ze[mask] = 0.224 * 3.6e18 * (0.0 + 2.0) * ni[mask] * vr ** 2 / (0.0 + 1.0) * (900.0 / 1000.0) ** 2
    return ze.float()


def combine_to_dbz(parts: dict[str, torch.Tensor], dbz_floor: float = DBZMIN) -> torch.Tensor:
    total = None
    for v in parts.values():
        total = v if total is None else (total + v)
    total = total.clamp_min(0.0)
    dbz = torch.zeros_like(total)
    pos = total > 0
    dbz[pos] = 10.0 * torch.log10(total[pos])
    return torch.maximum(dbz, torch.scalar_tensor(dbz_floor, dtype=dbz.dtype, device=dbz.device))
