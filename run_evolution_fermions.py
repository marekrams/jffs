import numpy as np
import csv
import os.path
from pathlib import Path
import ray
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sumLn2, Ln, momentum_total, momentum_n
# , measure_energy_per_site, measure_T_per_site, measure_Ln, measure_currents


def folder_gs(g, m, a, N):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/gs/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_evol(g, m, a, N, v, Q, D):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/{v=:0.4f}/{Q=:0.4f}/{D=}/")
    path.mkdir(parents=True, exist_ok=True)
    return path


@ray.remote(num_cpus=1)
def run_gs(g, m, a, N, D0, energy_tol=1e-12, Schmidt_tol=1e-10, t=0):
    """ initial state at t=0 """
    #
    folder = folder_gs(g, m, a, N)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    e0 = a * g * g / 2
    t = 0
    #
    ops = yastn.operators.SpinlessFermions(sym='U1')
    H0 = HNN(N, a, m, ops=ops)
    H1 = e0 * sumLn2(N, t, a, v=1, Q=1, ops=ops)
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
        psi_gs = mps.random_mps(H0, D_total=D0, n=0)
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_gs, [H0, H1], max_sweeps=500,
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


@ray.remote(num_cpus=2)
def run_evol(g, m, a, N, D0, v, Q, dt, snapshots, snapshots_states):
    ops = yastn.operators.Spin12(sym='U1')
    #
    try:
        fname = folder_gs(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = mps.load_from_dict(ops.config, data["psi"])
    except FileNotFoundError:
        return None
    #
    e0 = a * g * g / 2
    t = 0
    engs_gs = measure_energy_per_site(psi, a, g, m, t, v, Q, ops)

    folder = folder_evol(g, m, a, N, v, Q, D0)

    ops = yastn.operators.Spin12(sym='U1')
    H0 = HXXZ(N, a, m, ops=ops)
    Ht = lambda t: [H0, e0 * sumLn2(N, t, a, v, Q, ops=ops)]

    times = np.linspace(0, N * a / (2 * v), snapshots + 1)
    sps = snapshots // snapshots_states

    ii = 0
    for step in mps.tdvp_(psi, Ht, times, dt=dt,
                          method='2site', opts_svd={"D_total": D0}, yield_ti=True):

        engs = measure_energy_per_site(psi, a, g, m, step.tf, v, Q, ops)
        ents = psi.get_entropy()
        T00, T11, T01 = measure_T_per_site(psi, a, g, m, ops)
        Ln = measure_Ln(psi, ops)
        j0, j1 = measure_currents(psi, a, ops)

        if ii % sps == 0:
            data = {}
            data["psi"] = psi.save_to_dict()
            data["bd"] = psi.get_bond_dimensions()
            data["entropy"] = ents
            sch = psi.get_Schmidt_values()
            data["schmidt"] = [x.data for x in sch]
            #
            np.save(folder / f"state_t={step.tf:0.2f}.npy", data, allow_pickle=True)
            #
            with open(folder / "min_schmidt.txt", "a") as f:
                f.write(f"{step.tf:0.2f};{min(data['schmidt'][N // 2]):12f}\n")

        add_line_to_file(folder / "engs.txt", step.tf, engs - engs_gs)
        add_line_to_file(folder / "ents.txt", step.tf, ents)
        add_line_to_file(folder / "T00.txt", step.tf, T00)
        add_line_to_file(folder / "T01.txt", step.tf, T01)
        add_line_to_file(folder / "T11.txt", step.tf, T11)
        add_line_to_file(folder / "Ln.txt", step.tf, Ln)
        add_line_to_file(folder / "j0.txt", step.tf, j0)
        add_line_to_file(folder / "j1.txt", step.tf, j1)
        ii += 1


if __name__ == "__main__":
    #
    g = 1 / 5
    D0 = 64

    refs = []
    for m in [0 * g, 0.1 * g, 0.318 * g, 1 * g]:
        for N, a in [(100, 1.0)]: #, (200, 0.5), (400, 0.25)]:
            job = run_gs.remote(g, m, a, N, D0)
            refs.append(job)
    ray.get(refs)

    # refs = []
    # for m in [0 * g, 0.1 * g, 0.318 * g, 1 * g]:
    #     for N, a in [(100, 1.0), (200, 0.5), (400, 0.25)]:
    #         job = run_evol.remote(g, m, a, N, D0, 1, 1, 1/8, 100, 20)
    #         refs.append(job)
    # ray.get(refs)
