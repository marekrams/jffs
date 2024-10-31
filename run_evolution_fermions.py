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


@ray.remote(num_cpus=4)
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


def add_line_to_file(fname, tf, data):
    with open(fname, "a") as f:
        f.write(f"{tf:0.3f}")
        for ee in data:
            f.write(f";{ee:0.8f}")
        f.write("\n")


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

    evol = mps.tdvp_(psi, Ht, times,
                    method='2site', dt=dt,
                    opts_svd={"D_total": D, "tol": tol},
                    yield_initial=True)

    for ii, step in enumerate(evol):
        ents1 = psi.get_entropy()
        ents2 = psi.get_entropy(alpha=2)
        ents3 = psi.get_entropy(alpha=3)
        total_eng = mps.vdot(psi, Ht(step.tf), psi).real

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v, Q, ops)

        if ii % sps == 0:
            data = {}
            data["psi"] = psi.save_to_dict()
            data["bd"] = psi.get_bond_dimensions()
            data["entropy"] = ents1
            sch = psi.get_Schmidt_values()
            data["schmidt"] = [x.data for x in sch]
            #
            np.save(folder / f"state_t={step.tf:0.2f}.npy", data, allow_pickle=True)
            #
            with open(folder / "min_schmidt.txt", "a") as f:
                f.write(f"{step.tf:0.2f};{min(data['schmidt'][N // 2]):12f}\n")

        add_line_to_file(folder / "total_eng.txt", step.tf, [total_eng])
        add_line_to_file(folder / "ents1.txt", step.tf, ents1)
        add_line_to_file(folder / "ents2.txt", step.tf, ents2)
        add_line_to_file(folder / "ents3.txt", step.tf, ents3)
        add_line_to_file(folder / "T00.txt", step.tf, T00)
        add_line_to_file(folder / "T01.txt", step.tf, T01)
        add_line_to_file(folder / "T11.txt", step.tf, T11)
        add_line_to_file(folder / "Ln.txt", step.tf, Ln)
        add_line_to_file(folder / "j0.txt", step.tf, j0)
        add_line_to_file(folder / "j1.txt", step.tf, j1)
        add_line_to_file(folder / "nu.txt", step.tf, nu)


if __name__ == "__main__":
    #
    g = 1 / 5
    D0 = 64

    refs = []
    for m in [0 * g, 0.1 * g, 0.318309886 * g, 1 * g]:
        for N, a in [(256, 0.5), (512, 0.25)]: #, (1024, 0.125), (1024, 0.25)]:
            job = run_gs.remote(g, m, a, N, D0)
            refs.append(job)
    ray.get(refs)

    refs = []
    v, Q = 1, 1
    dt, D, tol = 1/8, 64, 1e-6
    for m in [0 * g, 0.1 * g, 0.318309886 * g, 1 * g]:
        for N, a in [(256, 0.5), (512, 0.25)]: #, (1024, 0.125), (1024, 0.25)]:
            job = run_evol.remote(g, m, a, N, D0, v, Q, dt, D, tol, 4 * N, 16)
            refs.append(job)
    ray.get(refs)
