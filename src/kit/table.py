"""Tuned tables: CSV files of measured configs. ``kit.tune`` writes them, the dispatch
(``Op.pick``) reads them.

A row holds the key columns (dims), the bucket column, ``config``, then the
measurements: ``us`` and any metrics (e.g. ``cos``), i.e. every column after
``config``. The columns up to ``config`` are the row's identity: writing a row replaces
the row with the same identity. Rows are kept sorted by their identity without the
config (numbers as numbers), then by ``us``: the configs of a point fastest first.

    fields, rows = read(path)              # header, rows as {column: str}
    merge(path, rows)                      # replace / add rows, keep the order
    compare(old_path, new_path)            # per point: the winners of two tables
"""

import csv
import os

__all__ = ["read", "write", "merge", "identity", "compare"]


def read(path):
    """(fields, rows) of a table file: its header and its rows ({column: str})."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return tuple(reader.fieldnames or ()), rows


def identity(fields):
    """The identity columns of a header: those up to and including ``config``."""
    if "config" not in fields:
        raise ValueError(f"a tuned table has a config column; header {list(fields)}")
    return tuple(fields[:list(fields).index("config") + 1])


def _num(s):
    """Sort key of a cell: numbers before text, numbers by value."""
    for t in (int, float):
        try:
            return (0, t(s))
        except (TypeError, ValueError):
            pass
    return (1, str(s))


def _point_key(fields):
    """Sort key of a row: its identity without the config, then ``us``."""
    cols = [f for f in identity(fields) if f != "config"]
    timed = "us" in fields
    return lambda r: (tuple(_num(r.get(c, "")) for c in cols)
                      + ((_num(r.get("us", "")),) if timed else ()))


def write(path, rows, fields):
    """Write ``rows`` (sorted, see the module docstring) to ``path`` atomically."""
    rows = sorted(rows, key=_point_key(fields))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def merge(path, rows, fields=None):
    """Add ``rows`` to the table at ``path`` (created if missing): a row with the identity
    of an existing one replaces it. ``fields``: the header of a new file (default: the
    existing file's, else the first row's columns)."""
    rows = list(rows)
    old = []
    if os.path.exists(path):
        fields, old = read(path)
    elif fields is None:
        if not rows:
            raise ValueError(f"{path}: no rows and no fields for a new table")
        fields = tuple(rows[0])
    ident = identity(fields)
    new = {tuple(str(r[c]) for c in ident) for r in rows}
    keep = [r for r in old if tuple(r[c] for c in ident) not in new]
    write(path, keep + rows, fields)


def _points(path):
    """{point identity (all identity columns but config): {config: us}}."""
    fields, rows = read(path)
    cols = [f for f in identity(fields) if f != "config"]
    out = {}
    for r in rows:
        out.setdefault(tuple(r[c] for c in cols), {})[r["config"]] = float(r["us"])
    return cols, out


def compare(old_path, new_path, *, label=None, order=None, out=print):
    """Per point of the new table: its winner, the old table's winner and what keeping
    the old winner costs in the new table. ``label`` / ``order``: functions of the
    point ({column: value}) giving its printed name and its sort key (default: the
    columns joined, the file order). Returns the number of points whose winner changed."""
    cols, new = _points(new_path)
    _, old = _points(old_path)
    named = {k: dict(zip(cols, k)) for k in new}
    label = label or (lambda p: " ".join(f"{c}={v}" for c, v in p.items()))
    keys = sorted(new, key=lambda k: order(named[k])) if order else list(new)
    changed = 0
    for k in keys:
        name, nw = label(named[k]), min(new[k], key=new[k].get)
        if k not in old:
            out(f"{name}  new {nw:28s} {new[k][nw]:8.1f}  (no old row)")
            continue
        ow = min(old[k], key=old[k].get)
        keep = new[k].get(ow)
        gain = f"{(new[k][nw] / keep - 1) * 100:+5.1f}%" if keep else "  n/a"
        changed += ow != nw
        out(f"{name}  old {ow:28s} {old[k][ow]:8.1f}  new {nw:28s} "
            f"{new[k][nw]:8.1f}  old-cfg-now {keep or float('nan'):8.1f} {gain}"
            + ("  *" if ow != nw else ""))
    out(f"\n{changed} of {len(new)} points changed winner (*); old-cfg-now: the old winner's "
        "time in the new table, the % is what the new winner gains over it")
    return changed
