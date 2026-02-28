"""
Raw data download script

Usage:
    python src/data/download.py              # download all sources
    python src/data/download.py --only fg    # FlavorGraph only
    python src/data/download.py --only fdb   # FlavorDB only
    TODO: python src/data/download.py --only r1m   # Recipe1M+ only
    TODO: python src/data/download.py --only mstm  # MoleculeSTM weights only
"""

import csv
import ssl
import time
import json
import shutil
import argparse
import subprocess
import urllib.error
import urllib.request
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
RAW = ROOT / "data" / "raw"

FLAVORGRAPH_DIR = RAW / "flavorgraph"
FLAVORDB_DIR = RAW / "flavordb"
RECIPE1M_DIR = RAW / "recipe1m"
MOLECULESTM_DIR = RAW / "moleculestm"

# Flavorgraph weights, nodes, edges (from GitHub)

FLAVORGRAPH_REPO = "https://github.com/lamypark/FlavorGraph.git"
FLAVORGRAPH_BRANCH = "master"


def download_flavorgraph():
    """Clone FlavorGraph repo and extract graph CSVs + pretrained weights."""
    print("\n[FlavorGraph] Cloning repo...")

    if FLAVORGRAPH_DIR.exists():
        print(f"  Already exists at {FLAVORGRAPH_DIR}.")
        print("  Delete the directory and re-run to re-download.")
        return

    FLAVORGRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Shallow clone to save bandwidth
    tmp_dir = RAW / "_fg_tmp"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            FLAVORGRAPH_BRANCH,
            FLAVORGRAPH_REPO,
            str(tmp_dir),
        ],
        check=True,
    )

    # Move relevant files — adjust paths based on actual repo structure
    # The repo structure may vary; inspect after first clone and update these paths
    for pattern in ["*.csv", "*.pkl", "*.pt", "*.npy"]:
        for f in tmp_dir.rglob(pattern):
            dest = FLAVORGRAPH_DIR / f.relative_to(tmp_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    shutil.rmtree(tmp_dir)

    # List what we got
    files = list(FLAVORGRAPH_DIR.rglob("*"))
    print(
        f"  Downloaded {len([f for f in files if f.is_file()])} files to {FLAVORGRAPH_DIR}"
    )
    for f in sorted(files):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.relative_to(FLAVORGRAPH_DIR)} ({size_mb:.1f} MB)")


# FlavorDB mappings (from API)
FLAVORDB_API_BASE = "https://cosylab.iiitd.edu.in/flavordb2"
FLAVORDB_MAX_ENTITY_ID = 1000  # Upper bound of number of entities in DB


def download_flavordb():
    """Download FlavorDB entities and molecule data via the API.

    Endpoint: GET /entity_details?id={entity_id}

    Each response contains:
      - entity_id, category, entity_alias_readable (the ingredient)
      - molecules[]: array of molecule objects, each containing:
          - pubchem_id, common_name, smile (SMILES string)
          - flavor_profile: "@"-delimited descriptor string (e.g. "sweet@fruity@green")
          - fema_flavor_profile, fooddb_flavor_profile: additional descriptor sources
          - molecular_weight, functional_groups, cas_id, etc.

    We cache each entity response as raw JSON and also extract structured CSVs
    for downstream use.
    """
    print("\n[FlavorDB] Downloading via API...")
    print(f"  Base URL: {FLAVORDB_API_BASE}/entities_json?id={{id}}")

    FLAVORDB_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = FLAVORDB_DIR / "entities_raw"
    raw_dir.mkdir(exist_ok=True)

    # Test with known good entity ID 0 (Egg)
    print("  Testing connectivity with entity ID 0...")
    test_url = f"{FLAVORDB_API_BASE}/entities_json?id=0"
    context = ssl._create_unverified_context()  # SSL bypass

    try:
        req = urllib.request.Request(
            test_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (FlavorManifold)",
            },
        )
        with urllib.request.urlopen(req, timeout=15, context=context) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw_bytes = resp.read()
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        # Check if we got JSON or HTML error page
        if (
            "text/html" in content_type
            or raw_text.strip().startswith("<!")
            or raw_text.strip().startswith("<html")
        ):
            print(f"  ERROR: Server returned HTML instead of JSON.")
            print(f"  Content-Type: {content_type}")
            print(f"  First 200 chars: {raw_text[:200]}")
            print(f"  The API may be down or the endpoint may have changed.")
            print(f"  Try opening {test_url} in your browser.")
            return

        test_data = json.loads(raw_text)

        # Print structure for debugging
        if isinstance(test_data, dict):
            print(f"  ✓ Got JSON dict. Keys: {list(test_data.keys())[:10]}")
            print(f"    entity_id: {test_data.get('entity_id', 'MISSING')}")
            print(
                f"    entity_alias_readable: {test_data.get('entity_alias_readable', 'MISSING')}"
            )
            print(f"    molecules count: {len(test_data.get('molecules', []))}")
        elif isinstance(test_data, list):
            print(f"  ✓ Got JSON list with {len(test_data)} items.")
            if test_data and isinstance(test_data[0], dict):
                print(f"    First item keys: {list(test_data[0].keys())[:10]}")
        else:
            print(f"  ✓ Got JSON {type(test_data).__name__}: {str(test_data)[:200]}")

    except urllib.error.HTTPError as e:
        print(f"  ERROR: HTTP {e.code} {e.reason}")
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"  Response body: {body}")
        return
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print(f"  Cannot reach {FLAVORDB_API_BASE}")
        print(f"  If this works in your browser, check proxy/firewall settings.")
        return

    # Response structure
    # The response might be the entity directly, or wrapped in something
    def extract_entity(data):
        """Return (entity_dict, is_valid) from whatever the API returns."""
        if isinstance(data, dict):
            if "entity_id" in data and "molecules" in data:
                return data, True
            # Maybe wrapped: check common wrapper keys
            for key in ["data", "entity", "result"]:
                if key in data and isinstance(data[key], dict):
                    inner = data[key]
                    if "entity_id" in inner:
                        return inner, True
        elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return extract_entity(data[0])
        return data, False

    # Validate extractor on the test data
    test_entity, test_valid = extract_entity(test_data)
    if not test_valid:
        print(f"  ERROR: Cannot find entity_id + molecules in response.")
        print(f"  Response structure: {type(test_data).__name__}")
        if isinstance(test_data, dict):
            print(f"  Top-level keys: {list(test_data.keys())}")
        print(
            f"  Saving raw test response to {raw_dir / 'TEST_RESPONSE.json'} for inspection."
        )
        with open(raw_dir / "TEST_RESPONSE.json", "w") as f:
            json.dump(test_data, f, indent=2)
        return

    print(f"  ✓ Preflight passed. Starting sweep of IDs 1–{FLAVORDB_MAX_ENTITY_ID}...")

    # Sweep all entity IDs
    downloaded = 0
    skipped = 0
    not_found = 0
    errors = []

    id_range = range(0, FLAVORDB_MAX_ENTITY_ID + 1)
    if tqdm:
        id_iter = tqdm(id_range, desc="  FlavorDB", unit="id", ncols=80)
    else:
        id_iter = id_range

    for eid in id_iter:
        detail_file = raw_dir / f"{eid}.json"
        if detail_file.exists():
            skipped += 1
            if tqdm:
                id_iter.set_postfix(dl=downloaded, skip=skipped, miss=not_found)
            continue

        url = f"{FLAVORDB_API_BASE}/entities_json?id={eid}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (FlavorManifold Research)",
                },
            )
            with urllib.request.urlopen(req, timeout=15, context=context) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")

            # Skip HTML error pages
            if raw_text.strip().startswith("<!") or raw_text.strip().startswith(
                "<html"
            ):
                not_found += 1
                if tqdm:
                    id_iter.set_postfix(dl=downloaded, skip=skipped, miss=not_found)
                continue

            data = json.loads(raw_text)
            entity, valid = extract_entity(data)

            if not valid:
                not_found += 1
                if tqdm:
                    id_iter.set_postfix(dl=downloaded, skip=skipped, miss=not_found)
                continue

            with open(detail_file, "w") as f:
                json.dump(entity, f, indent=2)
            downloaded += 1

            if tqdm:
                id_iter.set_postfix(dl=downloaded, skip=skipped, miss=not_found)
            elif downloaded % 50 == 0:
                print(f"  Fetched {downloaded} entities (at ID {eid})...")

            time.sleep(0.15)

        except urllib.error.HTTPError as e:
            if e.code in (404, 400, 500):
                not_found += 1
            else:
                errors.append((eid, f"HTTP {e.code}"))
                if len(errors) <= 3:
                    print(f"  ID {eid}: HTTP {e.code} {e.reason}")
        except json.JSONDecodeError:
            not_found += 1
        except Exception as e:
            errors.append((eid, str(e)))
            if len(errors) <= 3:
                print(f"  ID {eid}: {type(e).__name__}: {e}")
            # If we get 20+ consecutive errors, bail
            if len(errors) >= 20 and all(eid - errors[-20][0] < 25 for _ in [1]):
                print(f"  Too many errors. Stopping at ID {eid}.")
                break

    print(f"\n  Raw download complete:")
    print(f"    New downloads:  {downloaded}")
    print(f"    Already cached: {skipped}")
    print(f"    IDs not found:  {not_found}")
    print(f"    Errors:         {len(errors)}")
    if errors:
        print(f"    First few errors: {errors[:5]}")

    # Parse raw JSON into structured CSVs for the pipeline
    print("  Parsing into structured CSVs...")

    entities = []  # (entity_id, name, category)
    molecules = []  # (pubchem_id, common_name, smile, molecular_weight, functional_groups)
    entity_mol = []  # (entity_id, pubchem_id)  — ingredient-molecule edges
    mol_desc = set()  # (pubchem_id, descriptor)  — molecule-descriptor edges

    json_files = sorted(raw_dir.glob("*.json"))
    if tqdm:
        json_files = tqdm(json_files, desc="  Parsing", unit="file", ncols=80)

    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)

        eid = data["entity_id"]
        name = data.get(
            "entity_alias_readable", data.get("entity_alias", f"entity_{eid}")
        )
        category = data.get("category", "unknown")
        entities.append((eid, name, category))

        for mol in data.get("molecules", []):
            pid = mol.get("pubchem_id")
            if pid is None:
                continue

            molecules.append(
                (
                    pid,
                    mol.get("common_name", ""),
                    mol.get("smile", ""),
                    mol.get("molecular_weight", ""),
                    mol.get("functional_groups", ""),
                )
            )

            entity_mol.append((eid, pid))

            # Extract descriptors from flavor_profile (primary, "@"-delimited)
            fp = mol.get("flavor_profile", "")
            if fp:
                for desc in fp.split("@"):
                    desc = desc.strip().lower()
                    if desc:
                        mol_desc.add((pid, desc))

    # Deduplicate molecules
    seen_mols = {}
    for m in molecules:
        if m[0] not in seen_mols:
            seen_mols[m[0]] = m

    entities_csv = FLAVORDB_DIR / "entities.csv"
    with open(entities_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_id", "name", "category"])
        w.writerows(entities)

    molecules_csv = FLAVORDB_DIR / "molecules.csv"
    with open(molecules_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pubchem_id",
                "common_name",
                "smile",
                "molecular_weight",
                "functional_groups",
            ]
        )
        w.writerows(seen_mols.values())

    edges_entity_mol_csv = FLAVORDB_DIR / "edges_entity_molecule.csv"
    with open(edges_entity_mol_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_id", "pubchem_id"])
        w.writerows(entity_mol)

    edges_mol_desc_csv = FLAVORDB_DIR / "edges_molecule_descriptor.csv"
    with open(edges_mol_desc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pubchem_id", "descriptor"])
        w.writerows(sorted(mol_desc))

    print(f"  Entities:                {len(entities)}")
    print(f"  Unique molecules:        {len(seen_mols)}")
    print(f"  Entity-molecule edges:   {len(entity_mol)}")
    print(f"  Molecule-descriptor edges: {len(mol_desc)}")
    print(f"  CSVs written to {FLAVORDB_DIR}/")


SOURCES = {
    "fg":   ("FlavorGraph",  download_flavorgraph),
    "fdb":  ("FlavorDB",     download_flavordb),
    # TODO: "r1m":  ("Recipe1M+",    download_recipe1m),
    # TODO: "mstm": ("MoleculeSTM",  download_moleculestm),
}


def main():
    """Download data from sources"""
    parser = argparse.ArgumentParser(
        description="Download raw data for Flavor Manifold project."
    )
    parser.add_argument(
        "--only",
        choices=list(SOURCES.keys()),
        help="Download only a specific source.",
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if args.only:
        name, fn = SOURCES[args.only]
        print(f"Downloading: {name}")
        fn()
    else:
        print("Downloading all data sources...")
        for key, (name, fn) in SOURCES.items():
            fn()

    print("\n" + "=" * 60)
    print("Download summary:")
    print("=" * 60)
    for key, (name, _) in SOURCES.items():
        path = {
            "fg": FLAVORGRAPH_DIR,
            "fdb": FLAVORDB_DIR,
            "r1m": RECIPE1M_DIR,
            "mstm": MOLECULESTM_DIR,
        }[key]
        if path.exists():
            file_count = len([f for f in path.rglob("*") if f.is_file()])
            total_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            status = f"{file_count} files, {total_mb:.1f} MB"
        else:
            status = "NOT DOWNLOADED"
        print(f"  {name:20s} {status}")
    print()


if __name__ == "__main__":
    main()
