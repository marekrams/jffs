import argparse
import numpy as np
from pathlib import Path
import os.path
import csv
import time
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sumLn2, measure_local_observables


def folder_temp(g, m, a, N, dt, D, tol, method):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/temp/{dt=:0.4f}/{D=}/{tol=:0.0e}/{method}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_gs(g, m, a, N):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/gs/")
    path.mkdir(parents=True, exist_ok=True)
    return path


# @ray.remote(num_cpus=9)
def run_gs(g, m, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    folder = folder_gs(g, m, a, N)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sumLn2(N, t=0, a=a, v=1, Q=1, ops=ops)
    #
    files = list(folder.glob("*.npy"))
    Ds = [int(f.stem.split("=")[1]) for f in files]
    if any(D <= D0 for D in Ds):
        D = max(D for D in Ds if D <= D0)
        print(f"Loading initial state with {D=}")
        old_data = np.load(folder / f"state_D={D}.npy", allow_pickle=True).item()
        psi_gs = mps.load_from_dict(ops.config, old_data["psi"])
    else:
        print(f"Random initial state.")
        psi_gs = mps.random_mps(H0, D_total=D0, n=(N // 2))
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_gs, [H0, H1], max_sweeps=200,
                     method='2site', opts_svd={"D_total": D0, "tol": 1e-6},
                     energy_tol=energy_tol, Schmidt_tol=Schmidt_tol, precompute=False)
    #
    data = {}
    data["psi"] = psi_gs.save_to_dict()
    data["bd"] = psi_gs.get_bond_dimensions()
    data["entropy"] = psi_gs.get_entropy()
    sch = psi_gs.get_Schmidt_values()
    data["schmidt"] = [x.data for x in sch]
    np.save(fname, data, allow_pickle=True)

    fieldnames = ["D", "energy", "sweeps", "denergy", "dSchmidt", "min_Schmidt"]
    out = {"D" : max(data["bd"]),
           "energy": info.energy,
           "sweeps": info.sweeps,
           "denergy": info.denergy,
           "dSchmidt": info.max_dSchmidt,
           "min_Schmidt": min(data["schmidt"][N // 2])}
    file_exists = os.path.isfile(finfo)
    with open(finfo, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)


def run_temp(g, m, a, N, D, dt, tol, method, snapshots, bmax):
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    e0 = a * g * g / 2
    folder = folder_temp(g, m, a, N, dt, D, tol, method)
    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sumLn2(N, t=0, a=a, v=0, Q=0, ops=ops)
    HH = H0 + H1

    data = {}
    data['entropy_1'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['entropy_2'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['entropy_3'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['Ln'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T00'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T11'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T01'] = np.zeros((snapshots + 1, N - 2), dtype=np.float64)
    data['j0'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['j1'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['nu'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['energy'] = np.zeros(snapshots + 1, dtype=np.float64)
    data['time'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0
    data['min_Schmidt'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0

    dt0 = 0.00001
    psi = mps.product_mpo(ops.I(), N=N)
    psi = psi - (dt0 * HH) + (dt0 ** 2 / 2) * (HH @ HH) - (dt0 ** 3 / 6) * (HH @ HH @ HH)
    psi.canonize_(to='last')
    psi.truncate_(to='first', opts_svd={'D_total': D})

    print(psi.get_bond_dimensions())

    times = np.linspace(0, bmax / 2, snapshots + 1)
    times[0] = dt0

    evol = mps.tdvp_(psi, HH, times,
                    method=method, dt=dt, u=1,
                    opts_svd={"D_total": D, "tol": tol},
                    yield_initial=True, subtract_E=True)

    tref0 = time.time()
    for ii, step in enumerate(evol):
        data['time'][ii] = step.tf
        data['entropy_1'][ii, :] = psi.get_entropy(alpha=1)
        data['entropy_2'][ii, :] = psi.get_entropy(alpha=2)
        data['entropy_3'][ii, :] = psi.get_entropy(alpha=3)
        data['energy'][ii] = mps.vdot(psi, H0, psi).real + mps.vdot(psi, H1, psi).real
        data["min_Schmidt"][ii] = min(psi.get_Schmidt_values()[N // 2].data)

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v=0, Q=0, ops=ops)
        data['T00'][ii, :] = T00
        data['T11'][ii, :] = T11
        data['T01'][ii, :] = T01
        data['j0'][ii, :] = j0
        data['j1'][ii, :] = j1
        data['nu'][ii, :] = nu
        data['Ln'][ii, :] = Ln
        np.save(folder / f"results.npy", data, allow_pickle=True)
        print(f"t={step.tf:0.2f}  st={time.time() - tref0:0.1f} sek.")


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=float, default=1.0)
    parser.add_argument("-D0", type=int, default=256)
    parser.add_argument("-mm", type=float, default=0.5)
    parser.add_argument("-N", type=int, default=128)
    parser.add_argument("-a", type=float, default=0.125)

    args = parser.parse_args()
    print(args)

    D, tol, method = args.D0, 1e-6, '2site'
    snapshots, bmax = 100, 5
    dt = 1 / 200
    tref0 = time.time()

    run_gs(args.g, args.mm, args.a, args.N, args.D0)
    run_temp(args.g, args.mm, args.a, args.N, args.D0, dt, tol, method, snapshots, bmax)
    print(f"Evolution finished in: {time.time() - tref0}")
