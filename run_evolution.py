import numpy as np
import csv
import os.path
from pathlib import Path
import ray
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sumLn2, measure_local_observables


def folder_gs(g, m, a, N):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/gs/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_evol(g, m, a, N, v, Q, D0, dt, D, tol):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/{v=:0.4f}/{Q=:0.4f}/{D0=}/{dt=:0.4f}/{D=}/{tol=:0.0e}/")
    path.mkdir(parents=True, exist_ok=True)
    return path


@ray.remote(num_cpus=2)
def run_gs(g, m, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    folder = folder_gs(g, m, a, N)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    ops = yastn.operators.SpinlessFermions(sym='U1')
    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sumLn2(N, t=0, a=a, v=1, Q=1, ops=ops)
    #
    files = list(folder.glob("*.npy"))
    Ds = [int(f.stem.split("=")[1]) for f in files]
    if any(D <= D0 for D in Ds):
        D = min(D for D in Ds if D <= D0)
        print(f"Loading initial state with {D=}")
        old_data = np.load(folder / f"state_D={D}.npy", allow_pickle=True).item()
        psi_gs = mps.load_from_dict(ops.config, old_data["psi"])
    else:
        print(f"Random initial state.")
        psi_gs = mps.random_mps(H0, D_total=D0, n=(N // 2))
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_gs, [H0, H1], max_sweeps=200,
                     method='2site', opts_svd={"D_total": D0},
                     energy_tol=energy_tol, Schmidt_tol=Schmidt_tol)
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

def save_psi(fname, psi):
    data = {}
    data["psi"] = psi.save_to_dict()
    data["bd"] = psi.get_bond_dimensions()
    np.save(fname, data, allow_pickle=True)

@ray.remote(num_cpus=4)
def run_evol(g, m, a, N, D0, v, Q, dt, D, tol, snapshots, snapshots_states):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    #
    try:
        fname = folder_gs(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = mps.load_from_dict(ops.config, data["psi"])
    except FileNotFoundError:
        return None
    #
    e0 = a * g * g / 2
    folder = folder_evol(g, m, a, N, v, Q, D0, dt, D, tol)
    H0 = HNN(N, a, m, ops=ops)
    Ht = lambda t: [H0, e0 * sumLn2(N, t, a, v, Q, ops=ops)]

    times = np.linspace(0, N * a / (2 * v), snapshots + 1)
    sps = snapshots // snapshots_states

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

    evol = mps.tdvp_(psi, Ht, times,
                    method='2site', dt=dt,
                    opts_svd={"D_total": D, "tol": tol},
                    yield_initial=True)

    for ii, step in enumerate(evol):
        data['time'][ii] = step.tf
        data['entropy_1'][ii, :] = psi.get_entropy(alpha=1)
        data['entropy_2'][ii, :] = psi.get_entropy(alpha=2)
        data['entropy_3'][ii, :] = psi.get_entropy(alpha=3)
        data['energy'][ii] = mps.vdot(psi, Ht(step.tf), psi).real
        data["min_Schmidt"][ii] = min(psi.get_Schmidt_values()[N // 2].data)

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v, Q, ops)
        data['T00'][ii, :] = T00
        data['T11'][ii, :] = T11
        data['T01'][ii, :] = T01
        data['j0'][ii, :] = j0
        data['j1'][ii, :] = j1
        data['nu'][ii, :] = nu
        data['Ln'][ii, :] = Ln

        if ii % sps == 0:
            np.save(folder / f"results.npy", data, allow_pickle=True)
            save_psi(folder / f"state_t={step.tf:0.4f}.npy", psi)


if __name__ == "__main__":
    #
    g = 1 / 5
    D0 = 64

    # refs = []
    # for m in [0 * g, 0.1 * g, 0.318309886 * g, 1 * g]:
    #     for N, a in [(256, 0.5), (512, 0.25), (1024, 0.125), (1024, 0.25)]:
    #         job = run_gs.remote(g, m, a, N, D0)
    #         refs.append(job)
    # ray.get(refs)

    refs = []
    v, Q = 1, 1
    D, tol = 64, 1e-6
    for m in [0 * g, 0.1 * g, 0.318309886 * g, 1 * g]:
        for N, a in [(1024, 0.25)]: #, (512, 0.25)]: #, (1024, 0.125), (1024, 0.25)]:
            snapshots = 2 * N
            dt = min(1/8, N * a / (2 * v *  snapshots))
            job = run_evol.remote(g, m, a, N, D0, v, Q, dt, D, tol, snapshots, 16)
            refs.append(job)
    ray.get(refs)
