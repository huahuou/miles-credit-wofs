#!/usr/bin/env python3
"""
Diagnose 3D S-band radar reflectivity from MPAS NSSL 2-moment microphysics fields.

Reimplements the reflectivity calculation from radardd02() in module_mp_nssl_2mom.F
for the 2-moment case (ipconc=5, imurain=1).

Usage:
    python diag_nssl_refl.py INPUT_FILE [-o OUTPUT_FILE] [--member MEM] [--compare]

Examples:
    # Diagnose reflectivity and write to a new file
    python diag_nssl_refl.py ana/mem001.nc -o refl_diag_mem001.nc

    # Diagnose and compare with MPAS-computed refl10cm
    python diag_nssl_refl.py ana/mem001.nc --compare

    # Overwrite refl10cm in the input file (in-place)
    python diag_nssl_refl.py ana/mem001.nc --inplace

Author: auto-generated from MPAS NSSL source code analysis
"""

import argparse
import sys
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    print("ERROR: netCDF4 module not found. Load conda environment first:", file=sys.stderr)
    print("  source /etc/profile.d/modules.sh && module load rdhpcs-conda/25.3.1 && conda activate wofs_an", file=sys.stderr)
    sys.exit(1)

# =============================================================================
#  NSSL 2-moment default parameters (from module_mp_nssl_2mom.F)
# =============================================================================

PI = 3.141592653589793
PIINV = 1.0 / PI

# Shape parameters (gamma distribution)
ALPHAR  = 0.0    # rain  (imurain=1, gamma-diameter)
ALPHAH  = 0.0    # graupel
ALPHAHL = 1.0    # hail
SNU     = -0.8   # snow shape param (gamma-volume)
CINU    = 0.0    # ice crystal shape param

# Default densities [kg/m^3]
RHO_QH  = 500.0   # graupel
RHO_QHL = 900.0   # hail
RHO_QS  = 100.0   # snow
RWDN    = 1000.0   # water

# Density limits
HDNMN  = 170.0    # min graupel density
HLDNMN = 500.0    # min hail density

# Min mixing ratios for reflectivity calculation [kg/kg]
QRMIN = 1.0e-5
QSMIN = 1.0e-6
QHMIN = 1.0e-5
QIMIN = 1.0e-9   # qxmin(li) - very small

# Min number concentration threshold
CXMIN = 1.0e-8

# Min/max mean particle volumes [m^3] = (pi/6)*D^3
XVSMN  = 0.523599 * (0.01e-3)**3
XVSMX  = 0.523599 * (10.0e-3)**3
XVHMN  = 0.523599 * (0.3e-3)**3
XVHMX  = 0.523599 * (20.0e-3)**3
XVHLMN = 0.523599 * (0.3e-3)**3
XVHLMX = 0.523599 * (40.0e-3)**3

# Precomputed gamma values for snow (Cox 1988 formula)
from math import gamma as _gamma
GSNOW1  = _gamma(SNU + 1.0)        # Gamma(0.2)
GSNOW73 = _gamma(SNU + 7.0/3.0)    # Gamma(1.5333..)

# dBZ floor
DBZMIN = 0.0


def _g1(alpha):
    """Sixth-moment / third-moment ratio for gamma-diameter distribution."""
    return ((6+alpha)*(5+alpha)*(4+alpha) /
            ((3+alpha)*(2+alpha)*(1+alpha)))


def ze_rain(qr, nr, rho):
    """Rain reflectivity [mm^6/m^3], imurain=1 (gamma-diameter)."""
    g1 = _g1(ALPHAR)  # = 20.0 for ALPHAR=0
    coeff = 1.0e18 * (6.0 / (PI * RWDN))**2

    ze = np.zeros_like(qr)
    mask = (qr >= QRMIN) & (nr > 1.0e-3)
    zx = g1 * (rho[mask] * qr[mask])**2 / nr[mask]
    ze[mask] = coeff * zx
    return ze


def ze_snow(qs, ns, rho, temp, qr=None):
    """
    Snow reflectivity [mm^6/m^3], 2-moment.
    Uses Cox (1988) mass-size relation for dry snow, and the classical
    6th-moment formula with bright-band enhancement for wet snow
    (T > freezing with rain present, iusewetsnow=1).
    """
    TFR = 273.16
    KSQ = 0.189   # Smith (1984) dielectric for equiv. ice sphere

    ze = np.zeros_like(qs)
    mask = (qs >= QSMIN) & (ns > 1.0e-7)

    qs_m = qs[mask]
    ns_m = ns[mask]
    rho_m = rho[mask]
    temp_m = temp[mask]

    if qr is not None:
        qr_m = qr[mask]
    else:
        qr_m = np.zeros_like(qs_m)

    # Bright band: iusewetsnow=1, T > tfr+1, qs > qr, qr > qsmin
    bb = (temp_m > TFR + 1.0) & (qs_m > qr_m) & (qr_m > QSMIN)
    qxw = np.zeros_like(qs_m)
    qxw[bb] = np.minimum(0.5 * qs_m[bb], qr_m[bb])

    use_old = qxw > QSMIN

    # Old formula (wet snow / bright band):
    # Ze = 3.6e18 * (snu+2) * (0.224*(qs+qxw) + 0.776*qxw) * (qs+qxw)
    #      / (ns * (snu+1) * rwdn^2) * rho^2
    ze_old = np.zeros_like(qs_m)
    if use_old.any():
        qs_wet = qs_m[use_old] + qxw[use_old]
        ze_old[use_old] = (3.6e18 * (SNU + 2.0) *
                           (0.224 * qs_wet + 0.776 * qxw[use_old]) * qs_wet /
                           (ns_m[use_old] * (SNU + 1.0) * RWDN**2) * rho_m[use_old]**2)

    # Cox (1988) new formulation for dry snow: m = p * d^2 (p = 0.106214)
    ze_new = np.zeros_like(qs_m)
    use_new = ~use_old
    if use_new.any():
        ze_new[use_new] = (1.0e18 * 323.3226 * 0.106214**2 * KSQ *
                           qs_m[use_new]**2 * rho_m[use_new]**2 * GSNOW73 /
                           (ns_m[use_new] * 917.0**2 * GSNOW1 *
                            (1.0 + SNU)**(4.0/3.0)))

    ze[mask] = ze_old + ze_new
    return ze


def ze_ice(qi, ni, rho):
    """Ice crystal reflectivity [mm^6/m^3], assuming spherical ice (density 900)."""
    ze = np.zeros_like(qi)
    mask = (qi > QIMIN) & (ni > 1.0)

    vr = rho[mask] * qi[mask] / (900.0 * ni[mask])
    ze[mask] = 0.224 * 3.6e18 * (CINU + 2.0) * ni[mask] * vr**2 / (CINU + 1.0) * (900.0/1000.0)**2
    return ze


def ze_graupel(qg, ng, rho, volg=None):
    """
    Graupel reflectivity [mm^6/m^3], 2-moment.
    Uses volg to diagnose density if available, otherwise default RHO_QH.
    After clamping mean volume, recomputes effective N (matching radardd02).
    """
    g1 = _g1(ALPHAH)  # = 20.0 for ALPHAH=0
    coeff = 1.0e18 * (6.0 / (PI * RWDN))**2

    ze = np.zeros_like(qg)
    mask = (qg >= QHMIN) & (ng >= CXMIN)

    qg_m = qg[mask]
    ng_m = ng[mask]
    rho_m = rho[mask]

    if volg is not None:
        volg_m = volg[mask]
        valid_vol = volg_m > 0
        hwdn = np.full_like(qg_m, RHO_QH)
        hwdn[valid_vol] = rho_m[valid_vol] * qg_m[valid_vol] / volg_m[valid_vol]
        hwdn = np.clip(hwdn, 100.0, 900.0)
    else:
        hwdn = np.full_like(qg_m, RHO_QH)

    chw = ng_m.copy()
    xvh = rho_m * qg_m / (hwdn * np.maximum(chw, 1.0e-3))
    need_clamp = (xvh < XVHMN) | (xvh > XVHMX)
    xvh = np.clip(xvh, XVHMN, XVHMX)
    chw[need_clamp] = (rho_m[need_clamp] * qg_m[need_clamp] /
                       (xvh[need_clamp] * hwdn[need_clamp]))

    zx = g1 * rho_m**2 * 0.224 * qg_m * qg_m / chw
    ze[mask] = coeff * zx
    return ze


def ze_hail(qh, nh, rho, volh=None):
    """
    Hail reflectivity [mm^6/m^3], 2-moment.
    Uses volh to diagnose density if available, otherwise default RHO_QHL.
    After clamping mean volume, recomputes effective N (matching radardd02).
    """
    g1 = _g1(ALPHAHL)  # = 8.75 for ALPHAHL=1
    coeff = 1.0e18 * (6.0 / (PI * RWDN))**2

    ze = np.zeros_like(qh)
    mask = (qh >= QHMIN) & (nh > 0.0)

    qh_m = qh[mask]
    nh_m = nh[mask]
    rho_m = rho[mask]

    if volh is not None:
        volh_m = volh[mask]
        valid_vol = volh_m > 0
        hldn = np.full_like(qh_m, RHO_QHL)
        hldn[valid_vol] = rho_m[valid_vol] * qh_m[valid_vol] / volh_m[valid_vol]
        hldn = np.clip(hldn, 300.0, 900.0)
    else:
        hldn = np.full_like(qh_m, RHO_QHL)

    chl = nh_m.copy()
    xvhl = rho_m * qh_m / (hldn * np.maximum(chl, 1.0e-9))
    need_clamp = (xvhl < XVHLMN) | (xvhl > XVHLMX)
    xvhl = np.clip(xvhl, XVHLMN, XVHLMX)
    chl[need_clamp] = (rho_m[need_clamp] * qh_m[need_clamp] /
                       (xvhl[need_clamp] * hldn[need_clamp]))

    zx = g1 * rho_m**2 * 0.224 * qh_m * qh_m / chl
    ze[mask] = coeff * zx
    return ze


def compute_reflectivity(ds, time_idx=0):
    """
    Compute 3D reflectivity from an MPAS netCDF dataset.

    Parameters
    ----------
    ds : netCDF4.Dataset (opened for reading)
    time_idx : int, time index to read

    Returns
    -------
    dbz : np.ndarray, shape (nCells, nVertLevels), reflectivity in dBZ
    ze_components : dict of np.ndarray, linear Z contributions from each species
    """

    def _read(name, fallback=None):
        if name in ds.variables:
            arr = ds.variables[name][time_idx, :, :]
            return np.asarray(arr, dtype=np.float64)
        elif fallback is not None:
            return fallback
        else:
            return None

    rho = _read('rho')
    if rho is None:
        raise RuntimeError("'rho' (dry air density) not found in file")

    qr  = _read('qr',  np.zeros_like(rho))
    qi  = _read('qi',  np.zeros_like(rho))
    qs  = _read('qs',  np.zeros_like(rho))
    qg  = _read('qg',  np.zeros_like(rho))
    qh  = _read('qh',  np.zeros_like(rho))

    nr  = _read('nr',  np.zeros_like(rho))
    ni  = _read('ni',  np.zeros_like(rho))
    ns  = _read('ns',  np.zeros_like(rho))
    ng  = _read('ng',  np.zeros_like(rho))
    nh  = _read('nh',  np.zeros_like(rho))

    volg = _read('volg')
    volh = _read('volh')

    # Temperature from theta and pressure (for potential bright-band, reserved)
    theta = _read('theta')
    pressure = _read('pressure')
    if pressure is None:
        p_base = _read('pressure_base', np.zeros_like(rho))
        p_p    = _read('pressure_p',    np.zeros_like(rho))
        pressure = p_base + p_p

    P0 = 1.0e5
    RCP = 287.04 / 1004.5
    temp = theta * (pressure / P0)**RCP

    z_rain    = ze_rain(qr, nr, rho)
    z_snow    = ze_snow(qs, ns, rho, temp, qr=qr)
    z_ice     = ze_ice(qi, ni, rho)
    z_graupel = ze_graupel(qg, ng, rho, volg)
    z_hail    = ze_hail(qh, nh, rho, volh)

    z_total = z_rain + z_snow + z_ice + z_graupel + z_hail

    dbz = np.full_like(z_total, DBZMIN)
    pos = z_total > 0.0
    dbz[pos] = np.maximum(DBZMIN, 10.0 * np.log10(z_total[pos]))

    components = {
        'rain': z_rain, 'snow': z_snow, 'ice': z_ice,
        'graupel': z_graupel, 'hail': z_hail, 'total': z_total
    }
    return dbz, components


def compare_with_mpas(ds, dbz_diag, time_idx=0):
    """Print comparison statistics between diagnosed and MPAS refl10cm."""
    if 'refl10cm' not in ds.variables:
        print("  refl10cm not found in file, skipping comparison.")
        return

    refl_mpas = np.asarray(ds.variables['refl10cm'][time_idx, :, :], dtype=np.float64)

    diff = dbz_diag - refl_mpas
    mask_valid = (refl_mpas > 0) | (dbz_diag > 0)

    print(f"\n  === Comparison: diagnosed vs MPAS refl10cm ===")
    print(f"  MPAS refl10cm : min={refl_mpas.min():.2f}, max={refl_mpas.max():.2f}, mean={refl_mpas.mean():.4f}")
    print(f"  Diagnosed dBZ : min={dbz_diag.min():.2f}, max={dbz_diag.max():.2f}, mean={dbz_diag.mean():.4f}")
    print(f"  Difference    : min={diff.min():.2f}, max={diff.max():.2f}, mean={diff.mean():.4f}")

    if mask_valid.any():
        rmse = np.sqrt(np.mean(diff[mask_valid]**2))
        mae  = np.mean(np.abs(diff[mask_valid]))
        bias = np.mean(diff[mask_valid])
        corr = np.corrcoef(dbz_diag[mask_valid].ravel(), refl_mpas[mask_valid].ravel())[0,1]
        print(f"  Where either > 0 dBZ (N={mask_valid.sum()}):")
        print(f"    RMSE = {rmse:.3f} dBZ,  MAE = {mae:.3f} dBZ,  Bias = {bias:.3f} dBZ")
        print(f"    Correlation = {corr:.6f}")

    for thresh in [10, 20, 30, 40]:
        m1 = refl_mpas.max(axis=1) >= thresh
        m2 = dbz_diag.max(axis=1) >= thresh
        both = (m1 & m2).sum()
        print(f"  Composite >= {thresh} dBZ: MPAS={m1.sum()}, Diag={m2.sum()}, overlap={both}")


def write_output(input_file, output_file, dbz_diag, components, time_idx=0):
    """Write diagnosed reflectivity to a new netCDF file."""
    with nc.Dataset(input_file, 'r') as src, nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        # Copy coordinate variables
        for vname in ['xtime', 'latCell', 'lonCell', 'zgrid', 'zz']:
            if vname in src.variables:
                vsrc = src.variables[vname]
                vdst = dst.createVariable(vname, vsrc.datatype, vsrc.dimensions,
                                          zlib=True, complevel=4)
                vdst[:] = vsrc[:]
                for attr in vsrc.ncattrs():
                    vdst.setncattr(attr, vsrc.getncattr(attr))

        # Write diagnosed dBZ
        vout = dst.createVariable('refl10cm_diag', 'f4', ('Time', 'nCells', 'nVertLevels'),
                                  zlib=True, complevel=4, fill_value=-999.0)
        vout[0, :, :] = dbz_diag.astype(np.float32)
        vout.units = 'dBZ'
        vout.long_name = 'Diagnosed S-band reflectivity (NSSL 2-moment)'

        # Copy original refl10cm for comparison
        if 'refl10cm' in src.variables:
            vsrc = src.variables['refl10cm']
            vdst = dst.createVariable('refl10cm_mpas', vsrc.datatype, vsrc.dimensions,
                                      zlib=True, complevel=4)
            vdst[:] = vsrc[:]
            vdst.long_name = 'Original MPAS-computed reflectivity'

        # Write component contributions in dBZ
        for cname, zarr in components.items():
            if cname == 'total':
                continue
            vname = f'ze_{cname}'
            vout = dst.createVariable(vname, 'f4', ('Time', 'nCells', 'nVertLevels'),
                                      zlib=True, complevel=4, fill_value=0.0)
            dbz_c = np.zeros_like(zarr, dtype=np.float32)
            pos = zarr > 0
            dbz_c[pos] = 10.0 * np.log10(zarr[pos]).astype(np.float32)
            vout[0, :, :] = dbz_c
            vout.units = 'dBZ'
            vout.long_name = f'Reflectivity contribution from {cname} (dBZ)'

        dst.setncattr('history', 'Created by diag_nssl_refl.py')
        print(f"  Output written to: {output_file}")


def overwrite_inplace(input_file, dbz_diag, time_idx=0):
    """Overwrite refl10cm in the input file with diagnosed values."""
    with nc.Dataset(input_file, 'r+') as ds:
        if 'refl10cm' in ds.variables:
            ds.variables['refl10cm'][time_idx, :, :] = dbz_diag.astype(np.float32)
            print(f"  Overwrote refl10cm in: {input_file}")
        else:
            print(f"  WARNING: refl10cm not found in {input_file}, cannot overwrite.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose 3D reflectivity from MPAS NSSL 2-moment fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s ana/mem001.nc --compare
  %(prog)s ana/mem001.nc -o refl_diag.nc
  %(prog)s ana/mem001.nc --inplace
  %(prog)s ens/mem001.nc --compare
""")
    parser.add_argument('input', help='Input MPAS netCDF file')
    parser.add_argument('-o', '--output', help='Output netCDF file for diagnosed reflectivity')
    parser.add_argument('--compare', action='store_true',
                        help='Compare diagnosed reflectivity with MPAS refl10cm')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite refl10cm in the input file')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to process (default: 0)')
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    ds = nc.Dataset(args.input, 'r')

    nCells = len(ds.dimensions['nCells'])
    nLevels = len(ds.dimensions['nVertLevels'])
    print(f"  Grid: {nCells} cells x {nLevels} levels")

    # List available hydrometeor fields
    hydro_vars = ['qr','qi','qs','qg','qh','nr','ni','ns','ng','nh','volg','volh']
    present = [v for v in hydro_vars if v in ds.variables]
    missing = [v for v in hydro_vars if v not in ds.variables]
    print(f"  Available: {', '.join(present)}")
    if missing:
        print(f"  Missing (will use defaults): {', '.join(missing)}")

    dbz_diag, components = compute_reflectivity(ds, time_idx=args.time)

    # Summary statistics
    print(f"\n  Diagnosed reflectivity:")
    print(f"    min={dbz_diag.min():.2f}, max={dbz_diag.max():.2f}, mean={dbz_diag.mean():.4f} dBZ")
    for cname, zarr in components.items():
        if cname == 'total':
            continue
        npos = (zarr > 0).sum()
        if npos > 0:
            dbz_c = 10.0 * np.log10(zarr[zarr > 0])
            print(f"    {cname:8s}: {npos:>8d} pts > 0, max Z = {dbz_c.max():.1f} dBZ")
        else:
            print(f"    {cname:8s}: no contribution")

    if args.compare:
        compare_with_mpas(ds, dbz_diag, time_idx=args.time)

    if args.output:
        write_output(args.input, args.output, dbz_diag, components, time_idx=args.time)

    if args.inplace:
        ds.close()
        overwrite_inplace(args.input, dbz_diag, time_idx=args.time)
    else:
        ds.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
